"""Capacity planner: Erlang-C queueing theory meeting vLLM continuous batching.

Telecom trunk sizing (Erlang-C / M/M/c) assumes load-independent service times.
In LLM continuous batching, service time depends on batch size (memory bandwidth
dominates decode steps, while prefill FLOPs compete for compute).

This module calculates:
1. First-order Erlang-C capacity requirements (traffic intensity in Erlangs).
2. Continuous batching simulation across candidate cluster sizes.
3. The empirical divergence: where load-dependent batching requires additional headroom.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

from .timing import ModelSpec, GPUSpec, RooflineTiming, kv_capacity_blocks
from .workload import WorkloadSpec, generate
from .engine import Engine, EngineConfig
from .metrics import SLO


def log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def erlang_c_probability(c: int, A: float) -> float:
    """
    Computes Erlang-C probability of queuing P(Wait > 0) for c servers with traffic intensity A.
    Uses log-space arithmetic to prevent overflow with large cluster sizes.
    """
    if c <= A:
        return 1.0  # Overloaded system: queue grows without bound
    if A <= 0:
        return 0.0

    rho = A / c
    # log(A^k / k!) for k from 0 to c-1
    log_terms = [k * math.log(A) - log_factorial(k) for k in range(c)]
    max_log = max(log_terms)
    sum_denom = sum(math.exp(t - max_log) for t in log_terms)

    # Last term: (A^c / c!) / (1 - rho)
    last_log_term = c * math.log(A) - log_factorial(c) - math.log(1.0 - rho)
    last_term_scaled = math.exp(last_log_term - max_log)

    p_wait = last_term_scaled / (sum_denom + last_term_scaled)
    return min(max(p_wait, 0.0), 1.0)


def erlang_c_queue_time_percentile(c: int, A: float, mu: float, percentile: float = 0.99) -> float:
    """
    Computes the queue waiting time for a given percentile (e.g. 0.99 for P99).
    mu: service rate per server (1 / mean_service_time)
    """
    if c <= A:
        return float("inf")
    pw = erlang_c_probability(c, A)
    if pw <= 0.0:
        return 0.0

    # P(W > t) = P_W * exp(-(c*mu - lambda)*t)
    # For target percentile p (e.g. 0.99), we solve 1 - P(W > t) = p  =>  P(W > t) = 1 - p
    unmet = 1.0 - percentile
    if unmet >= pw:
        return 0.0  # Probability of waiting is smaller than 1 - p

    service_margin = (c * mu) - (A * mu)  # c*mu - lambda
    if service_margin <= 0:
        return float("inf")

    wait_t = math.log(pw / unmet) / service_margin
    return max(0.0, wait_t)


@dataclass
class SizingReport:
    qps: float
    prompt_len: int
    output_len: int
    slo_ttft_s: float
    slo_tpot_s: float
    traffic_erlangs: float
    isolated_service_time_s: float
    erlang_min_gpus: int
    simulated_min_gpus: int
    batching_efficiency: float
    divergence_overhead_pct: float
    cluster_results: list[dict]


class CapacityPlanner:
    def __init__(self, model: ModelSpec, gpu: GPUSpec):
        self.model = model
        self.gpu = gpu
        self.timing = RooflineTiming(model, gpu)
        self.nb = kv_capacity_blocks(model, gpu)

    def single_request_service_time(self, prompt_len: int, output_len: int) -> tuple[float, float, float]:
        """
        Calculates baseline service time for an isolated request with no queuing or concurrency.
        Returns: (prefill_time, decode_time, total_time)
        """
        # 1. Isolated prefill time
        t_prefill = self.timing.step_time(prompt_len, 0, 0)
        # 2. Isolated decode time (cumulative KV reads)
        t_decode = sum(
            self.timing.step_time(0, 1, prompt_len + k)
            for k in range(max(1, output_len - 1))
        )
        return t_prefill, t_decode, t_prefill + t_decode

    def erlang_c_sizing(self, qps: float, prompt_len: int, output_len: int, slo_ttft_s: float = 0.5) -> tuple[int, float, float]:
        """
        First-order telecom sizing:
        Calculates traffic intensity A = lambda * E[S] (Erlangs)
        and finds minimum servers c such that P99 TTFT <= slo_ttft_s.
        Returns: (min_c, A, mean_service_time)
        """
        t_prefill, _, t_total = self.single_request_service_time(prompt_len, output_len)
        mu = 1.0 / t_total
        A = qps * t_total  # Offered load in Erlangs

        # Smallest integer servers c > A
        c = int(math.ceil(A))
        if c == int(A):
            c += 1

        allowed_queue_wait = max(0.0, slo_ttft_s - t_prefill)

        while True:
            p99_wait = erlang_c_queue_time_percentile(c, A, mu, percentile=0.99)
            if p99_wait <= allowed_queue_wait:
                break
            c += 1
            if c > 256:  # Safety boundary
                break

        return c, A, t_total

    def plan(
        self,
        qps: float,
        prompt_len: int = 512,
        output_len: int = 200,
        slo: SLO = SLO(ttft_s=0.5, tpot_s=0.05),
        duration_s: float = 20.0,
        candidate_sizes: list[int] | None = None
    ) -> SizingReport:
        """
        Runs both Erlang-C mathematical modeling and continuous batching simulation
        across candidate cluster sizes, reporting exact divergence.
        """
        t_prefill, t_decode, t_total = self.single_request_service_time(prompt_len, output_len)
        erlang_min, erlangs, _ = self.erlang_c_sizing(qps, prompt_len, output_len, slo.ttft_s)

        # Evaluate candidate GPU cluster sizes
        if candidate_sizes is None:
            max_cand = max(4, int(math.ceil(qps / 12.0)) + 1)
            candidate_sizes = list(range(1, max_cand + 1))

        cluster_results = []
        sim_min_gpus = None

        mu = 1.0 / t_total

        for num_gpus in candidate_sizes:
            # Under homogeneous load balancing, each GPU processes qps / num_gpus
            per_gpu_qps = qps / num_gpus

            # Erlang-C prediction for 1 slot/GPU
            erlang_p99_wait = erlang_c_queue_time_percentile(num_gpus, erlangs, mu, percentile=0.99)
            erlang_p99_ttft = erlang_p99_wait + t_prefill

            # Slotted Erlang-C prediction assuming 32 concurrency slots per GPU
            slotted_c = 32 * num_gpus
            slotted_wait = erlang_c_queue_time_percentile(slotted_c, erlangs, mu, percentile=0.99)
            slotted_ttft = slotted_wait + t_prefill

            # Continuous batching simulation on one representative GPU
            spec = WorkloadSpec(
                qps=per_gpu_qps,
                duration_s=duration_s,
                prompt_mean=prompt_len,
                output_mean=output_len,
                seed=42
            )
            reqs = generate(spec)
            cfg = EngineConfig(mode="continuous", num_blocks=self.nb, max_num_seqs=256)
            eng = Engine(self.timing, cfg)
            m = eng.run(reqs, slo)

            meets_slo = (m["ttft_p99"] <= slo.ttft_s) and (m["tpot_p99"] <= slo.tpot_s) and (m["n_dropped"] == 0)

            if meets_slo and sim_min_gpus is None:
                sim_min_gpus = num_gpus

            cluster_results.append({
                "num_gpus": num_gpus,
                "per_gpu_qps": per_gpu_qps,
                "naive_erlang_p99_ms": erlang_p99_ttft * 1000 if math.isfinite(erlang_p99_ttft) else 99999.0,
                "slotted_erlang_p99_ms": slotted_ttft * 1000 if math.isfinite(slotted_ttft) else 99999.0,
                "sim_p99_ttft_ms": m["ttft_p99"] * 1000,
                "sim_p99_tpot_ms": m["tpot_p99"] * 1000,
                "goodput_per_gpu": m["goodput_req_s"],
                "total_goodput": m["goodput_req_s"] * num_gpus,
                "kv_util_max": m["kv_util_max"],
                "meets_slo": meets_slo
            })

        if sim_min_gpus is None:
            sim_min_gpus = candidate_sizes[-1]

        batching_efficiency = erlang_min / sim_min_gpus
        divergence_pct = ((erlang_min - sim_min_gpus) / erlang_min) * 100.0

        return SizingReport(
            qps=qps,
            prompt_len=prompt_len,
            output_len=output_len,
            slo_ttft_s=slo.ttft_s,
            slo_tpot_s=slo.tpot_s,
            traffic_erlangs=erlangs,
            isolated_service_time_s=t_total,
            erlang_min_gpus=erlang_min,
            simulated_min_gpus=sim_min_gpus,
            batching_efficiency=batching_efficiency,
            divergence_overhead_pct=divergence_pct,
            cluster_results=cluster_results
        )
