"""Timing models: how long does one engine step take?

A step processes `prefill_tokens` (prompt tokens, possibly a chunk) plus one
decode token for each of `decode_seqs` sequences, as a single fused forward pass
(this is how vLLM does chunked prefill).

RooflineTiming: step_time = max(compute_time, memory_time) + fixed overhead.
  compute: ~2 * params FLOPs per token (attention FLOPs ignored: small at these lengths)
  memory : read all weights once per step + read the KV cache of every scheduled
           sequence + write new KV entries.
Decode is memory-bound (weights + KV reads); large prefill chunks are compute-bound.

Efficiency factors are *assumptions* until calibrated against real measurements.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: float
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2

    @property
    def weight_bytes(self) -> float:
        return self.params * self.dtype_bytes

    @property
    def kv_bytes_per_token(self) -> int:
        # K and V, per layer, per KV head
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes


@dataclass(frozen=True)
class GPUSpec:
    name: str
    mem_bw: float        # bytes/s (peak)
    flops: float         # dense FLOP/s at serving dtype (peak)
    mem_bytes: float     # HBM capacity
    bw_eff: float = 0.75     # achievable fraction of peak bandwidth
    flops_eff: float = 0.5   # achievable fraction of peak FLOPs (MFU)


LLAMA3_8B = ModelSpec("llama3-8b", params=8.03e9, layers=32, kv_heads=8, head_dim=128)
H100_SXM = GPUSpec("h100-sxm", mem_bw=3.35e12, flops=989e12, mem_bytes=80e9)


class TimingModel(Protocol):
    def step_time(self, prefill_tokens: int, decode_seqs: int, kv_tokens_read: int) -> float: ...


@dataclass
class RooflineTiming:
    model: ModelSpec
    gpu: GPUSpec
    step_overhead_s: float = 1.0e-3   # scheduler + kernel launch + sampling

    def step_time(self, prefill_tokens: int, decode_seqs: int, kv_tokens_read: int) -> float:
        tokens = prefill_tokens + decode_seqs
        if tokens == 0:
            return 0.0
        m, g = self.model, self.gpu
        compute_s = 2.0 * m.params * tokens / (g.flops * g.flops_eff)
        mem_bytes = m.weight_bytes + (kv_tokens_read + tokens) * m.kv_bytes_per_token
        memory_s = mem_bytes / (g.mem_bw * g.bw_eff)
        return max(compute_s, memory_s) + self.step_overhead_s


def kv_capacity_blocks(model: ModelSpec, gpu: GPUSpec, block_size: int = 16,
                       gpu_mem_util: float = 0.90) -> int:
    """How many KV blocks fit after weights (ignores activation workspace)."""
    free = gpu.mem_bytes * gpu_mem_util - model.weight_bytes
    return max(0, int(free // (model.kv_bytes_per_token * block_size)))
