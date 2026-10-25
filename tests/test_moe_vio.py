"""Check that Vio measures the experts selected by current Transformers."""
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'MLAfication/qknorm/src/mha2mla'))
spec = importlib.util.spec_from_file_location('moe_vio', ROOT / 'MLAfication/qknorm/eval/collect_moe_vio.py')
vio = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vio
spec.loader.exec_module(vio)


class MoeVioTests(unittest.TestCase):
    def test_bf16_evaluation_preserves_long_position_rotary_values(self):
        config = Qwen3MoeConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                               moe_intermediate_size=32, num_hidden_layers=1,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                               num_experts=8, num_experts_per_tok=4)
        model = Qwen3MoeForCausalLM(config)
        hidden = torch.zeros(1, 3, 32, dtype=torch.bfloat16)
        positions = torch.tensor([[0, 1024, 4096]])
        expected = model.model.rotary_emb(hidden, positions)
        converted = vio.move_model(model, device='cpu', dtype=torch.bfloat16)
        actual = converted.model.rotary_emb(hidden, positions)
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in converted.parameters()))
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    def test_router_hooks_count_tokens_and_identical_routes(self):
        torch.manual_seed(123)
        config = Qwen3MoeConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                               moe_intermediate_size=32, num_hidden_layers=2,
                               num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                               num_experts=8, num_experts_per_tok=4)
        model = Qwen3MoeForCausalLM(config).eval()
        tokens = torch.tensor([[1, 2, 3, 4, 5]])
        baseline = vio.MoeStatsCollector(model, 'base', keep_routes=True)
        baseline.register()
        with torch.no_grad():
            model(tokens, use_cache=False)
        baseline.remove()
        summary, layers = baseline.finish()
        self.assertEqual(len(layers), 2)
        for layer, stats in baseline.layer_stats.items():
            self.assertEqual(stats['tokens'], 5)
            self.assertEqual(stats['counts'].sum().item(), 20)
            expected = torch.bincount(baseline.route_cache[layer][0].long().flatten(), minlength=8)
            torch.testing.assert_close(stats['counts'], expected)
        repeated = vio.MoeStatsCollector(model, 'same', baseline_routes=baseline.route_cache)
        repeated.register()
        with torch.no_grad():
            model(tokens, use_cache=False)
        repeated.remove()
        result, _ = repeated.finish()
        self.assertEqual(result['route_exact_match'], 1.0)
        self.assertEqual(result['route_average_vio'], 0.0)
        self.assertEqual(result['max_vio'], summary['max_vio'])


if __name__ == '__main__':
    unittest.main()
