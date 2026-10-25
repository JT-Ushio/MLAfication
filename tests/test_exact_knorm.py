"""Exact-KNorm controls, using tiny CPU Qwen3 models and no downloaded weights."""
import copy
from dataclasses import fields
import importlib.util
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoConfig, Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import rotate_half

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "MLAfication/qknorm"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conversion = load_module("gen/patch_func.py", "exact_conversion")
patching = load_module("src/mha2mla/patching_mlafication.py", "exact_patching")
forward = load_module("src/mha2mla/patching_mla_fwd.py", "exact_forward")
latent = load_module("src/mha2mla/patching_mla_latent_fwd.py", "exact_latent")
helpers = load_module("src/mha2mla/helpers.py", "exact_helpers")
arguments = load_module("src/mha2mla/arguments.py", "exact_arguments")


class ExactKNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = Qwen3Config(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, attention_dropout=0.0, pad_token_id=0,
        )
        config._attn_implementation = "eager"
        self.base = Qwen3ForCausalLM(config).eval()
        for layer in self.base.model.layers:
            layer.self_attn.q_norm.weight.data.copy_(torch.linspace(.6, 1.4, 8))
            layer.self_attn.k_norm.weight.data.copy_(torch.linspace(1.5, .5, 8))
        # Conversion artifacts use BF16 storage. Match that storage in the oracle,
        # while doing CPU arithmetic in FP32; retain FP32 rotary buffers.
        for parameter in self.base.parameters():
            parameter.data = parameter.data.bfloat16().float()
        self.ranks = torch.tensor([[3, 1], [1, 2], [0, 3], [2, 0]])
        self.topk = 2
        self.down = torch.randn(5, 32).bfloat16().float() * .125
        self.up_k = torch.randn(8, 5).bfloat16().float() * .125
        self.up_v = torch.randn(16, 5).bfloat16().float() * .125
        self.rms = torch.tensor([[.75], [1.5]])
        self.tokens = torch.tensor([[1, 4, 7, 11, 2], [3, 8, 9, 2, 6]])

    def tensors(self, mode):
        layers = [conversion.build_mla_layer_tensors(
            layer.self_attn, self.ranks, self.topk, self.down, self.up_k, self.up_v,
            knorm_mode=mode, calibrated_rms=self.rms,
        ) for layer in self.base.model.layers]
        return {key: torch.stack([layer[key] for layer in layers]) for key in layers[0]}

    def model(self, mode="exact", requested=None, config=None):
        tensor_path = self.root / f"{mode}.pt"
        torch.save(self.tensors(mode), tensor_path)
        model = copy.deepcopy(self.base)
        args = SimpleNamespace(mla_tensor_path=str(tensor_path), knorm_mode=requested)
        patching.patch_mlafication(model, config or model.config, args)
        forward.patch_mla_fwd_instance(model)
        model.config.mha2mla = {
            "model_name_or_path": "unused-tiny-base", "mla_tensor_path": str(tensor_path),
            "is_baseline": False, "knorm_mode": args.knorm_mode,
        }
        return model

    def test_calibrated_export_matches_legacy_formula(self):
        attn = self.base.model.layers[0].self_attn
        output = self.tensors("calibrated")
        self.assertNotIn("gamma_k", output)
        wkr, _ = conversion.seperate_nope_rope(attn.k_proj.weight, self.ranks, self.topk)
        gkr, gkc = conversion.seperate_nope_rope(
            attn.k_norm.weight.repeat(2), self.ranks, self.topk)
        expected_kr = (gkr.view(2, -1) / self.rms).view(-1, 1) * wkr
        expected_uk = self.up_k * gkc[:, None]
        expected_uk = (expected_uk.view(2, -1, 5) / self.rms[:, :, None]).view(-1, 5)
        torch.testing.assert_close(output["k_r_proj"][0], expected_kr.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(output["k_up_weight"][0], expected_uk.bfloat16(), rtol=0, atol=0)
        exact = self.tensors("exact")
        for key in ("q_proj", "gamma_q", "kv_down_weight", "v_up_weight", "rope_index"):
            torch.testing.assert_close(output[key], exact[key], rtol=0, atol=0)
        torch.testing.assert_close(exact["k_up_weight"][0], self.up_k.bfloat16(), rtol=0, atol=0)
        torch.testing.assert_close(exact["k_r_proj"][0], wkr.bfloat16(), rtol=0, atol=0)

    def original_order_reference(self):
        """Independent oracle: reconstruct K in its original order, then HF KNorm.

        This does not use the patched forward, its reordered gains, or its
        per-head partial-RoPE gather function.
        """
        reference = copy.deepcopy(self.base)
        down, up_k, up_v = self.down, self.up_k, self.up_v
        frequency_mask = torch.zeros(2, 4, dtype=torch.bool)
        for head in range(2):
            frequency_mask[head, self.ranks[:self.topk, head]] = True
        key_mask = frequency_mask.repeat(1, 2)
        query_mask = key_mask.repeat_interleave(2, dim=0)

        def reference_attention(attn, hidden_states, position_embeddings, attention_mask, **kwargs):
            b, length, _ = hidden_states.shape
            q = attn.q_norm(attn.q_proj(hidden_states).view(b, length, 4, 8)).transpose(1, 2)
            k = attn.k_proj(hidden_states).clone()
            c = F.linear(hidden_states, down)
            k[..., ~key_mask.flatten()] = F.linear(c, up_k)
            k = attn.k_norm(k.view(b, length, 2, 8)).transpose(1, 2)
            v = F.linear(c, up_v).view(b, length, 2, 8).transpose(1, 2)
            cos, sin = (value.unsqueeze(1) for value in position_embeddings)
            q = torch.where(query_mask[None, :, None], q * cos + rotate_half(q) * sin, q)
            k = torch.where(key_mask[None, :, None], k * cos + rotate_half(k) * sin, k)
            k, v = k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)
            scores = q @ k.transpose(-1, -2) * attn.scaling
            if attention_mask is not None:
                scores = scores + attention_mask[..., :k.shape[-2]]
            weights = scores.softmax(dim=-1)
            result = (weights @ v).transpose(1, 2).reshape(b, length, 32)
            return attn.o_proj(result), weights

        for layer in reference.model.layers:
            layer.self_attn.forward = MethodType(reference_attention, layer.self_attn)
        return reference

    def test_exact_logits_and_input_gradients_match_original_order_reference(self):
        model, reference = self.model(), self.original_order_reference()
        # Include long positions so incorrect rotary-dimension permutations show up.
        positions = torch.tensor([[0, 127, 1024, 4096, 8192]]).expand(2, -1)
        x = self.base.get_input_embeddings()(self.tokens).detach().requires_grad_()
        y = x.detach().clone().requires_grad_()
        actual = model(inputs_embeds=x, position_ids=positions, use_cache=False).logits
        expected = reference(inputs_embeds=y, position_ids=positions, use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=3e-4, atol=2e-5)

    def test_cached_decoding_matches_full_sequence(self):
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend), torch.no_grad():
                model = self.model()
                model.config._attn_implementation = backend
                expected = model(self.tokens, use_cache=False).logits
                prefix = model(self.tokens[:, :3], use_cache=True)
                cache = prefix.past_key_values
                chunks = [prefix.logits]
                for i in range(3, self.tokens.shape[1]):
                    output = model(self.tokens[:, i:i + 1], past_key_values=cache, use_cache=True)
                    cache = output.past_key_values
                    chunks.append(output.logits)
                torch.testing.assert_close(torch.cat(chunks, dim=1), expected, rtol=2e-5, atol=2e-6)
                if hasattr(cache, "layers"):
                    keys, values = cache.layers[0].keys, cache.layers[0].values
                else:
                    keys, values = cache.key_cache[0], cache.value_cache[0]
                self.assertEqual(tuple(keys.shape), (2, 2, 5, 8))
                self.assertEqual(tuple(values.shape), (2, 2, 5, 8))

    def test_legacy_cache_keyword_is_consumed(self):
        model = self.model().eval()
        attn = model.model.layers[0].self_attn
        hidden = torch.randn(2, 3, 32)
        position_embeddings = (torch.ones(2, 3, 8), torch.zeros(2, 3, 8))
        cache = unittest.mock.Mock()
        cache.update.side_effect = lambda key, value, *args: (key, value)
        attn(hidden, position_embeddings, None, past_key_value=cache)
        cache.update.assert_called_once()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            attn(hidden, position_embeddings, None,
                 past_key_value=cache, past_key_values=unittest.mock.Mock())

    def test_qkv_distillation_updates_key_norm_and_keeps_backbone_frozen(self):
        model = self.model().train()
        with patch.object(helpers.dist, "get_rank", return_value=0):
            helpers.freeze_non_attn_weights_v2(model)
        helpers.enable_input_require_grads(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        teacher = copy.deepcopy(self.base).eval().requires_grad_(False)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        old_gamma = model.model.layers[0].self_attn.k_norm.weight.detach().clone()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        student_logits = model(self.tokens, use_cache=False).logits.float()
        with torch.no_grad():
            teacher_logits = teacher(self.tokens, use_cache=False).logits.float()
        loss = F.kl_div(student_logits.log_softmax(-1), teacher_logits.softmax(-1), reduction="sum")
        loss.backward()
        for suffix in ("k_norm.weight", "kv_proj.down_kv.weight", "kv_proj.up_kc.weight", "k_r_proj.weight"):
            grad = dict(model.named_parameters())[f"model.layers.0.self_attn.{suffix}"].grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.norm().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(old_gamma, model.model.layers[0].self_attn.k_norm.weight))
        for name, value in frozen.items():
            torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_checkpoint_roundtrips_preserve_exact_mode_and_trained_gamma(self):
        model = self.model()
        with patch.object(helpers.dist, "get_rank", return_value=0):
            helpers.freeze_non_attn_weights_v2(model)
        with torch.no_grad():
            model.model.layers[0].self_attn.k_norm.weight.mul_(1.25)
            expected = model(self.tokens, use_cache=False).logits
        tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
        for partial in (True, False):
            with self.subTest(partial=partial):
                folder = self.root / str(partial)
                model.config.mha2mla["checkpoint_type"] = "attn_only" if partial else "full"
                helpers.save_checkpoint(model, folder, tokenizer, trainable_only=partial)
                config = AutoConfig.from_pretrained(folder)
                self.assertEqual(config.mla_knorm_mode, "exact")
                self.assertEqual(config.mha2mla["mla_tensor_path"], "mla_tensors.pt")
                # Reload using only the bundled artifact, as an evaluation loader does.
                restored = copy.deepcopy(self.base)
                args = SimpleNamespace(**config.mha2mla)
                args.mla_tensor_path = str(folder / args.mla_tensor_path)
                patching.patch_mlafication(restored, config, args)
                forward.patch_mla_fwd_instance(restored)
                result = helpers.load_checkpoint(restored, folder, strict=not partial)
                self.assertFalse(result.unexpected_keys)
                self.assertNotIn("model.layers.0.self_attn.k_norm.weight", result.missing_keys)
                with torch.no_grad():
                    torch.testing.assert_close(restored(self.tokens, use_cache=False).logits,
                                               expected, rtol=0, atol=0)

    def test_mode_mismatches_and_latent_cache_are_rejected(self):
        for mode, requested in (("calibrated", "exact"), ("exact", "calibrated")):
            with self.assertRaisesRegex(ValueError, "KNorm mode mismatch"):
                self.model(mode=mode, requested=requested)
        config = copy.deepcopy(self.base.config)
        config.mla_knorm_mode = "exact"
        with self.assertRaisesRegex(ValueError, "KNorm mode mismatch"):
            self.model(mode="calibrated", config=config)
        exact = self.model()
        with self.assertRaisesRegex(ValueError, "latent-cache/MQA"):
            latent.patch_mla_fwd_instance(exact)
        calibrated = self.model("calibrated")
        self.assertFalse(hasattr(calibrated.model.layers[0].self_attn, "k_norm"))
        latent.patch_mla_fwd_instance(calibrated)
        folder = self.root / "wrong-mode"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps({"mla_knorm_mode": "calibrated"}))
        with self.assertRaisesRegex(ValueError, "Checkpoint KNorm mode"):
            helpers.load_checkpoint(exact, folder, strict=False)

    def test_training_recipe_uses_supported_arguments_and_matched_budget(self):
        config = yaml.safe_load((SOURCE / "cfgs/qwen3-8b-rank512-exact-knorm.example.yml").read_text())
        supported = {field.name for cls in (arguments.MHA2MLAModelArguments,
                     arguments.MHA2MLADataArguments, arguments.MHA2MLATrainingArguments)
                     for field in fields(cls)}
        self.assertFalse(set(config) - supported)
        self.assertEqual(config["knorm_mode"], "exact")
        self.assertEqual(8 * config["per_device_train_batch_size"] * config["gradient_accumulation_steps"], 64)
        self.assertEqual(config["max_steps"], 2400)
        self.assertEqual(config["max_steps"] * 64 * config["sequence_length"], 629145600)
        self.assertEqual(config["data_sampling_steps"], 4800)
        self.assertEqual(config["data_seed"], 42)
        self.assertTrue(config["accelerator_config"]["use_seedable_sampler"])
        self.assertEqual(config["dataset_weights"], [0.7, 0.15, 0.08, 0.06, 0.01])
        # The main-table checkpoint was saved halfway through a 4800-step run.
        # Match its observed LR history, not a rescaled 2400-step schedule.
        optimizer = torch.optim.AdamW(
            [torch.nn.Parameter(torch.zeros(1))], lr=config["learning_rate"]
        )
        # Scheduler math is independent of Nanotron's distributed-only logger.
        with patch.dict("sys.modules", {
            "nanotron.logging": SimpleNamespace(log_rank=lambda *args, **kwargs: None),
        }):
            scheduler = helpers.lr_scheduler_builder(
                optimizer, SimpleNamespace(**config["lr_scheduler_kwargs"]), config["max_steps"]
            )
        for update in range(1, config["max_steps"] + 1):
            self.assertAlmostEqual(
                optimizer.param_groups[0]["lr"],
                1e-4 * min((update - 1) / 144, 1.0), places=12,
            )
            optimizer.step()
            scheduler.step()

    def test_early_stop_keeps_historical_sampler_population(self):
        class FakeNanoset(torch.utils.data.Dataset):
            def __init__(self, **kwargs):
                self.settings = kwargs

            def __len__(self):
                return self.settings["train_split_num_samples"]

            def __getitem__(self, index):
                return index

        class Tokenizer:
            def __len__(self):
                return 151936

        data_args = arguments.MHA2MLADataArguments(
            is_nanoset=True, sequence_length=4096, data_sampling_steps=4800
        )
        train_args = SimpleNamespace(
            per_device_train_batch_size=2, gradient_accumulation_steps=4, max_steps=2400
        )
        with patch.dict("os.environ", {"WORLD_SIZE": "8"}), patch.dict("sys.modules", {
            "nanotron.data.nanoset": SimpleNamespace(Nanoset=FakeNanoset),
        }):
            dataset = helpers.load_dataset(data_args, train_args, Tokenizer())
            self.assertEqual(len(dataset), 307200)
            self.assertEqual(dataset.settings["random_seed"], 42)
            self.assertEqual(dataset.settings["token_size"], 4)
            historical = torch.randperm(307200, generator=torch.Generator().manual_seed(42))
            observed = torch.tensor(list(torch.utils.data.RandomSampler(
                dataset, generator=torch.Generator().manual_seed(42)
            )))
            self.assertTrue(torch.equal(observed[:153600], historical[:153600]))
            data_args.data_sampling_steps = None
            short = helpers.load_dataset(data_args, train_args, Tokenizer())
            self.assertEqual(len(short), 153600)
            wrong = torch.tensor(list(torch.utils.data.RandomSampler(
                short, generator=torch.Generator().manual_seed(42)
            )))
            self.assertFalse(torch.equal(wrong, historical[:153600]))
            for sampling_steps, offset in ((2399, 0), (0, 0), (4800, 1)):
                data_args.data_sampling_steps = sampling_steps
                data_args.data_offset_steps = offset
                with self.assertRaisesRegex(ValueError, "data_sampling_steps"):
                    helpers.load_dataset(data_args, train_args, Tokenizer())


if __name__ == "__main__":
    unittest.main()
