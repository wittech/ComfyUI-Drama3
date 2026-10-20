"""Legacy fixed-capacity attention cache, kept local as ComfyUI's cache API evolves.

Adapted from ComfyUI v0.41.0, comfy/text_encoders/llama.py (GPL-3.0).
https://github.com/Comfy-Org/ComfyUI/blob/v0.41.0/comfy/text_encoders/llama.py
"""
from dataclasses import dataclass

import torch
import comfy_kitchen


@dataclass
class FixedKVBias:
    key: torch.Tensor
    value: torch.Tensor
    index: int
    position: torch.Tensor
    seqlen: torch.Tensor
    # The last `seq` rows of this shared [1, 1, rows, capacity] bias serve the queries.
    bias: torch.Tensor = None
    tracker: dict = None

    def prepare(self, num_tokens):
        if num_tokens == 1 and self.seqlen is not None:
            self.seqlen.fill_(self.index + num_tokens)
        if self.tracker["step"] == (self.index, num_tokens):
            return
        self.tracker["step"] = (self.index, num_tokens)
        i = self.index
        rows = self.bias.shape[-2]
        if num_tokens <= rows:
            torch.arange(i, i + rows, out=self.position)
            window = self.tracker[num_tokens]
            start = i - rows + 1
            skip = max(-start, 0)
            end = min(i + rows, self.bias.shape[-1])
            self.bias[..., :, start + skip:end] = window[:, skip:end - start]
        else:
            self.bias[..., :, i:i + num_tokens] = 0

    def advance(self, num_tokens):
        self.index += num_tokens

    def rollback(self, discard=1):
        # The next prepare/write masks and overwrites the rejected tail.
        self.index -= discard

    @staticmethod
    def shared(capacity, device, dtype):
        # Layers on the same device advance in lockstep and share position/bias updates.
        rows = 6
        position = torch.empty((rows,), device=device, dtype=torch.int64)
        bias = torch.full((1, 1, rows, capacity), torch.finfo(dtype).min, device=device, dtype=dtype)
        tracker = {"step": -1}
        for n in range(1, rows + 1):
            window = torch.full((rows, 2 * rows - 1), torch.finfo(dtype).min, device=device, dtype=dtype)
            for r in range(rows):
                window[r, :rows + max(r - (rows - n), 0)] = 0
            tracker[n] = window
        return position, bias, tracker

    @classmethod
    def zeros(cls, batch, kv_heads, capacity, head_dim, device, dtype, shared):
        # Decode reads full capacity; zeroed masked tails keep the result finite.
        key = torch.zeros((batch, kv_heads, capacity, head_dim), device=device, dtype=dtype)
        seqlen = None
        flash_available = getattr(comfy_kitchen, "flash_attention_decode_is_available", None)
        if (dtype == torch.bfloat16 and head_dim in (128, 256) and flash_available is not None
                and hasattr(comfy_kitchen, "flash_attention_decode") and flash_available(key.device)):
            seqlen = torch.empty((batch,), device=device, dtype=torch.int32)
        return cls(key, torch.zeros_like(key), 0, shared[0], seqlen, shared[1], shared[2])

    def append(self, xk, xv):
        seq = xk.shape[2]
        self.key[:, :, self.index:self.index + seq] = xk
        self.value[:, :, self.index:self.index + seq] = xv
        return self.key[:, :, :self.index + seq], self.value[:, :, :self.index + seq]

    def decode(self, xq, xk, xv, num_kv_heads):
        # Device-side positions let a captured CUDA graph replay at a new cache offset.
        batch_size, num_heads, seq, head_dim = xq.shape
        self.key.index_copy_(2, self.position[:seq], xk)
        self.value.index_copy_(2, self.position[:seq], xv)
        if seq == 1 and self.seqlen is not None:
            out = comfy_kitchen.flash_attention_decode(
                xq.transpose(1, 2), self.key.transpose(1, 2), self.value.transpose(1, 2), self.seqlen)
            return out.reshape(batch_size, seq, num_heads * head_dim)
        groups = num_heads // num_kv_heads
        q = xq.reshape(batch_size, num_kv_heads, groups, seq, head_dim) * head_dim ** -0.5
        bias = self.bias[..., self.bias.shape[-2] - seq:, :].unsqueeze(1)
        scores = (q @ self.key.transpose(-1, -2).unsqueeze(2)).add_(bias)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(xq.dtype)
        out = probs @ self.value.unsqueeze(2)
        return out.permute(0, 3, 1, 2, 4).reshape(batch_size, seq, num_heads * head_dim)
