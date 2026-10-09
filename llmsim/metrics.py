"""Metrics: TTFT, TPOT, throughput, goodput under an SLO, KV utilization."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SLO:
    ttft_s: float = 0.5
    tpot_s: float = 0.05     # 50 ms/token ~ 20 tok/s per user


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def summarize(requests, slo: SLO, step_log, sim_end: float) -> dict:
    done = [r for r in requests if r.finish is not None]
    dropped = [r for r in requests if r.dropped]
    ttft = np.array([r.first_token - r.arrival for r in done])
    tpot = np.array([(r.finish - r.first_token) / (r.output_len - 1)
                     for r in done if r.output_len > 1])
    e2e = np.array([r.finish - r.arrival for r in done])
    ok = [r for r in done
          if (r.first_token - r.arrival) <= slo.ttft_s
          and (r.output_len == 1 or (r.finish - r.first_token) / (r.output_len - 1) <= slo.tpot_s)]
    dur = max(sim_end, 1e-9)

    # time-weighted KV utilization from per-step log: (dt, util)
    if step_log:
        dts = np.array([s[0] for s in step_log]); ut = np.array([s[1] for s in step_log])
        kv_mean = float((dts * ut).sum() / dts.sum()); kv_max = float(ut.max())
    else:
        kv_mean = kv_max = 0.0

    return {
        "n_requests": len(requests), "n_done": len(done), "n_dropped": len(dropped),
        "ttft_p50": _pct(ttft, 50), "ttft_p99": _pct(ttft, 99),
        "tpot_p50": _pct(tpot, 50), "tpot_p99": _pct(tpot, 99),
        "e2e_p99": _pct(e2e, 99),
        "throughput_tok_s": sum(r.output_len for r in done) / dur,
        "throughput_req_s": len(done) / dur,
        "goodput_req_s": len(ok) / dur,
        "slo_attainment": len(ok) / len(requests) if requests else float("nan"),
        "kv_util_mean": kv_mean, "kv_util_max": kv_max,
        "preemptions": sum(r.preemptions for r in requests),
        "sim_end_s": sim_end,
    }
