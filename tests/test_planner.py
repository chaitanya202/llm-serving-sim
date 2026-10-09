"""Tests for Telecom Erlang-C Queueing Theory and CapacityPlanner."""
import pytest
import math
from llmsim import (
    LLAMA3_8B, H100_SXM, CapacityPlanner,
    erlang_c_probability, erlang_c_queue_time_percentile, SLO
)


def test_erlang_c_math_properties():
    # 1. Overloaded system (c <= A) must yield P(Wait) = 1.0
    assert erlang_c_probability(c=5, A=5.0) == 1.0
    assert erlang_c_probability(c=4, A=5.0) == 1.0
    assert erlang_c_queue_time_percentile(c=4, A=5.0, mu=1.0) == float("inf")

    # 2. Zero traffic yields zero wait
    assert erlang_c_probability(c=5, A=0.0) == 0.0

    # 3. Known analytical textbook value: c=3, A=2
    # P_W = (2^3 / 3! / (1 - 2/3)) / (1 + 2 + 2^2/2 + 2^3 / 3! / (1 - 2/3))
    # Numerator = 8 / 6 / (1/3) = 4.0
    # Denominator = 1 + 2 + 2 + 4.0 = 9.0
    # P_W = 4 / 9 = 0.4444...
    pw = erlang_c_probability(c=3, A=2.0)
    assert pw == pytest.approx(4.0 / 9.0, rel=1e-4)

    # 4. Monotonicity: adding servers strictly reduces queuing probability
    assert erlang_c_probability(c=4, A=2.0) < pw
    assert erlang_c_probability(c=5, A=2.0) < erlang_c_probability(c=4, A=2.0)


def test_isolated_service_time_sanity():
    planner = CapacityPlanner(LLAMA3_8B, H100_SXM)
    t_prefill, t_decode, t_total = planner.single_request_service_time(prompt_len=512, output_len=200)

    # Prefill 512 tokens: ~10 to 30 ms
    assert 0.005 < t_prefill < 0.050
    # Decode 200 tokens: 200 tokens * ~7.38 ms = ~1.47 s
    assert 1.0 < t_decode < 2.0
    assert t_total == pytest.approx(t_prefill + t_decode)


def test_capacity_planner_recommendation():
    planner = CapacityPlanner(LLAMA3_8B, H100_SXM)
    # Plan for 50 QPS
    report = planner.plan(
        qps=50.0,
        prompt_len=512,
        output_len=150,
        slo=SLO(ttft_s=0.5, tpot_s=0.05),
        duration_s=10.0
    )

    assert report.erlang_min_gpus > 0
    assert report.traffic_erlangs > 0
    assert report.simulated_min_gpus >= 1
    # Continuous batching achieves dramatic consolidation vs unbatched Erlang-C
    assert report.simulated_min_gpus < report.erlang_min_gpus
    assert report.batching_efficiency > 1.0
    assert len(report.cluster_results) > 0
    assert report.cluster_results[0]["num_gpus"] == 1
