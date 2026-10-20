"""CPU regression checks; run with python -m unittest discover -s tests."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


def load_cache():
    # The optional CUDA kernels are absent here so the real tensor fallback runs on CPU.
    spec = importlib.util.spec_from_file_location(
        "drama3_test_kv_cache", Path(__file__).resolve().parents[1] / "drama3" / "kv_cache.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: module, "comfy_kitchen": types.ModuleType("comfy_kitchen")}):
        spec.loader.exec_module(module)
    return module.FixedKVBias


class FixedKVBiasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_type = load_cache()

    def make_cache(self, capacity=32, dtype=torch.float32, head_dim=8, shared=None):
        if shared is None:
            shared = self.cache_type.shared(capacity, "cpu", dtype)
        return self.cache_type.zeros(2, 2, capacity, head_dim, "cpu", dtype, shared)

    def assert_decode_matches(self, cache, width):
        batch, kv_heads, _, dim = cache.key.shape
        heads = 3 * kv_heads
        generator = torch.Generator().manual_seed(100 + cache.index + width)
        q = torch.randn(batch, heads, width, dim, generator=generator).to(cache.key.dtype)
        k = torch.randn(batch, kv_heads, width, dim, generator=generator).to(cache.key.dtype)
        v = torch.randn(batch, kv_heads, width, dim, generator=generator).to(cache.key.dtype)
        start = cache.index
        cache.prepare(width)
        out = cache.decode(q, k, v, kv_heads)
        end = start + width
        keys = cache.key[:, :, :end].repeat_interleave(heads // kv_heads, dim=1)
        values = cache.value[:, :, :end].repeat_interleave(heads // kv_heads, dim=1)
        visible = torch.arange(end)[None, :] <= torch.arange(start, end)[:, None]
        expected = F.scaled_dot_product_attention(q.float(), keys.float(), values.float(), attn_mask=visible)
        expected = expected.transpose(1, 2).reshape(batch, width, heads * dim).to(out.dtype)
        torch.testing.assert_close(out, expected, rtol=2e-2 if out.dtype == torch.bfloat16 else 1e-5,
                                   atol=2e-2 if out.dtype == torch.bfloat16 else 1e-6)
        cache.advance(width)

    def test_prefill_then_variable_width_decode(self):
        cache = self.make_cache(capacity=40)
        k, v = torch.randn(2, 2, 9, 8), torch.randn(2, 2, 9, 8)
        cache.prepare(9)
        keys, values = cache.append(k, v)
        torch.testing.assert_close(keys, k)
        torch.testing.assert_close(values, v)
        cache.advance(9)
        for width in (1, 2, 3, 4, 5, 6, 1):
            with self.subTest(width=width, index=cache.index):
                self.assert_decode_matches(cache, width)

    def test_short_prompt_and_capacity_boundary(self):
        for width in range(1, 7):
            with self.subTest(width=width):
                cache = self.make_cache(capacity=width + 1)
                self.assert_decode_matches(cache, width)
                self.assert_decode_matches(cache, 1)

    def test_rollback_masks_rejected_tokens(self):
        cache = self.make_cache(capacity=16)
        self.assert_decode_matches(cache, 6)
        self.assert_decode_matches(cache, 6)
        cache.rollback(5)
        self.assertEqual(cache.index, 7)
        self.assert_decode_matches(cache, 2)
        cache.rollback()
        self.assert_decode_matches(cache, 1)

    def test_shared_state_across_layers(self):
        shared = self.cache_type.shared(16, "cpu", torch.float32)
        caches = [self.make_cache(capacity=16, shared=shared) for _ in range(2)]
        self.assertIs(caches[0].position, caches[1].position)
        self.assertIs(caches[0].bias, caches[1].bias)
        self.assertIsNot(caches[0].key, caches[1].key)
        for width in (6, 1, 3, 1):
            for cache in caches:
                self.assert_decode_matches(cache, width)
        torch.testing.assert_close(caches[0].key, caches[1].key)

    def test_bfloat16_without_optional_flash_kernels(self):
        cache = self.make_cache(capacity=12, dtype=torch.bfloat16, head_dim=128)
        self.assertIsNone(cache.seqlen)
        self.assert_decode_matches(cache, 6)
        self.assert_decode_matches(cache, 1)


if __name__ == "__main__":
    unittest.main()
