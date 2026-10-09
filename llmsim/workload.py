"""Traffic profile: Poisson arrivals, lognormal prompt/output lengths."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Request:
    id: int
    arrival: float
    prompt_len: int
    output_len: int
    # --- mutable simulation state ---
    computed: int = 0            # tokens whose KV is resident
    generated: int = 0
    blocks: int = 0
    first_token: float | None = None
    finish: float | None = None
    dropped: bool = False
    preemptions: int = 0
    prefill_target: int = 0      # tokens to (re)compute before decoding; set on init/preempt

    def __post_init__(self):
        self.prefill_target = self.prompt_len


@dataclass
class WorkloadSpec:
    qps: float
    duration_s: float = 60.0
    prompt_mean: float = 512
    prompt_sigma: float = 0.8     # lognormal shape
    output_mean: float = 200
    output_sigma: float = 0.8
    max_prompt: int = 8192
    max_output: int = 2048
    seed: int = 0


def _lognormal(rng, mean, sigma, n, lo, hi):
    mu = np.log(mean) - sigma ** 2 / 2   # so that E[X] == mean (before clipping)
    return np.clip(rng.lognormal(mu, sigma, n).round().astype(int), lo, hi)


def generate(spec: WorkloadSpec) -> list[Request]:
    rng = np.random.default_rng(spec.seed)
    n = int(rng.poisson(spec.qps * spec.duration_s))
    arrivals = np.sort(rng.uniform(0, spec.duration_s, n))  # Poisson process given count
    prompts = _lognormal(rng, spec.prompt_mean, spec.prompt_sigma, n, 1, spec.max_prompt)
    outputs = _lognormal(rng, spec.output_mean, spec.output_sigma, n, 1, spec.max_output)
    return [Request(i, float(a), int(p), int(o))
            for i, (a, p, o) in enumerate(zip(arrivals, prompts, outputs))]
