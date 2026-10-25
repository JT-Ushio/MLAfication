"""CPU numerical checks; the eight-GPU smoke test lives beside train_ep.py."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

PATH = Path(__file__).resolve().parents[1] / 'MLAfication/qknorm/src/mha2mla/ep_distill_utils.py'
spec = importlib.util.spec_from_file_location('ep_utils', PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


class DistillationTests(unittest.TestCase):
    def test_portable_export_roundtrip_and_interrupted_directory(self):
        config = Qwen3MoeConfig(vocab_size=32, hidden_size=32, intermediate_size=32,
                               moe_intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
                               num_key_value_heads=1, head_dim=16, num_experts=4, num_experts_per_tok=2)
        model = Qwen3MoeForCausalLM(config)
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        mlp = model.model.layers[0].mlp
        packed = torch.nn.Module()
        packed.ep_size = 1
        packed.router = torch.nn.Module()
        packed.router.gate = mlp.gate
        packed.experts = torch.nn.Module()
        for name, tensor in [('w1', mlp.experts.gate_up_proj[:, :16]),
                             ('w3', mlp.experts.gate_up_proj[:, 16:]), ('w2', mlp.experts.down_proj)]:
            setattr(packed.experts, name, torch.nn.Parameter(tensor.detach().clone()))
        model.model.layers[0].mlp = packed
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor_path = root / 'mla.pt'
            torch.save({}, tensor_path)
            model.config.mha2mla = {'mla_tensor_path': str(tensor_path)}
            partial = root / 'export.incomplete'
            partial.mkdir()
            (partial / 'old-partial').write_text('preserve me')
            dist.init_process_group('gloo', init_method=f'file://{root / "rendezvous"}', rank=0, world_size=1)
            try:
                for _ in range(2):
                    utils.export_hf_checkpoint(SimpleNamespace(module=model),
                        SimpleNamespace(save_pretrained=lambda path: None), root / 'export', 'out_in',
                        tensor_path, {'step': 2})
            finally:
                dist.destroy_process_group()
            self.assertTrue(list(root.glob('export.incomplete-previous-*/old-partial')))
            self.assertFalse(partial.exists())
            self.assertTrue((root / 'export/export_complete.json').exists())
            index = json.loads((root / 'export/model.safetensors.index.json').read_text())
            restored = {}
            for shard in set(index['weight_map'].values()):
                restored.update(load_file(str(root / 'export' / shard)))
            self.assertEqual(restored.keys(), expected.keys())
            for name, tensor in expected.items():
                torch.testing.assert_close(restored[name], tensor)

    def test_chunked_kl_matches_full_vocabulary_loss_and_gradient(self):
        torch.manual_seed(7)
        for temperature in [1.0, 2.0]:
            for block in [1, 4, 64]:
                student = torch.randn(2, 5, 17, dtype=torch.float64, requires_grad=True)
                teacher = torch.randn_like(student, requires_grad=True)
                reference = F.kl_div(F.log_softmax(student.reshape(-1, 17) / temperature, -1),
                                     F.softmax(teacher.detach().reshape(-1, 17) / temperature, -1),
                                     reduction='batchmean') * temperature ** 2
                expected_grad, = torch.autograd.grad(reference, student)
                result = utils.distillation_kl(student, teacher, temperature, block)
                result.backward()
                torch.testing.assert_close(result, reference)
                torch.testing.assert_close(student.grad, expected_grad)
                self.assertIsNone(teacher.grad)

    def test_global_batches_and_resume_are_invariant_to_micro_batch(self):
        expected = None
        for micro in [1, 2, 4, 8]:
            ranks = [list(utils.GlobalBatchSampler(256, 64, micro, rank, 8)) for rank in range(8)]
            accumulation = 8 // micro
            steps = [sorted(x for rank in ranks for batch in rank[s * accumulation:(s + 1) * accumulation]
                            for x in batch) for s in range(4)]
            if expected is None:
                expected = steps
            self.assertEqual(steps, expected)
            self.assertEqual(len(set(sum(steps, []))), 256)
            resumed = list(utils.GlobalBatchSampler(256, 64, micro, 3, 8, start_step=2))
            self.assertEqual(resumed, ranks[3][2 * accumulation:])

    def test_expert_export_preserves_gate_up_down_layout(self):
        w1, w3 = torch.randn(8, 12, 16), torch.randn(8, 12, 16)
        w2 = torch.randn(8, 16, 12)
        for layout in ['out_in', 'in_out']:
            gate_up, down = utils.pack_hf_experts(w1, w2, w3, layout)
            if layout == 'in_out':
                gate_up, down = gate_up.transpose(1, 2), down.transpose(1, 2)
            torch.testing.assert_close(gate_up[:, :12], w1)
            torch.testing.assert_close(gate_up[:, 12:], w3)
            torch.testing.assert_close(down, w2)

    def test_teacher_has_no_optimizer_and_student_uses_zero2(self):
        config = dict(global_batch_size=64, ep_size=8, learning_rate=1e-4)
        teacher = utils.engine_config(config, 4, 8, teacher=True)
        student = utils.engine_config(config, 4, 8)
        self.assertNotIn('optimizer', teacher)
        self.assertEqual(teacher['zero_optimization']['stage'], 0)
        self.assertEqual(student['zero_optimization']['stage'], 2)
        self.assertEqual(student['gradient_accumulation_steps'], 2)
        with self.assertRaises(ValueError):
            utils.accumulation_steps(64, 3, 8)


if __name__ == '__main__':
    unittest.main()
