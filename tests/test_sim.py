import pytest
from llmsim import (Engine, EngineConfig, RooflineTiming, LLAMA3_8B, H100_SXM,
                    WorkloadSpec, Request, generate, SLO, kv_capacity_blocks)
from llmsim.kvcache import BlockAllocator

T = RooflineTiming(LLAMA3_8B, H100_SXM)
NB = kv_capacity_blocks(LLAMA3_8B, H100_SXM)


def run(mode, qps, dur=20, **kw):
    eng = Engine(T, EngineConfig(mode=mode, num_blocks=kw.pop("num_blocks", NB), **kw))
    reqs = generate(WorkloadSpec(qps=qps, duration_s=dur))
    return eng, reqs, eng.run(reqs)


def test_allocator_conservation():
    a = BlockAllocator(10, 16)
    assert a.blocks_for(1) == 1 and a.blocks_for(16) == 1 and a.blocks_for(17) == 2
    a.allocate(4); a.release(4)
    assert a.free == 10
    with pytest.raises(MemoryError):
        a.allocate(11)


def test_kv_capacity_sane():
    # 80GB*0.9 - 16GB weights = ~56GB / (128KiB*16) ~ 26-27k blocks
    assert 25_000 < NB < 29_000
    assert LLAMA3_8B.kv_bytes_per_token == 131072


def test_single_request_matches_analytic():
    r = Request(0, 0.0, prompt_len=512, output_len=11)
    eng = Engine(T, EngineConfig(mode="continuous", num_blocks=NB))
    eng.run([r])
    ttft_expected = T.step_time(512, 0, 0)
    assert r.first_token == pytest.approx(ttft_expected)
    assert r.finish > r.first_token
    # 10 decode steps follow, each reading growing context
    expected = ttft_expected + sum(T.step_time(0, 1, 512 + k) for k in range(10))
    assert r.finish == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["continuous", "static"])
def test_all_finish_and_kv_released(mode):
    eng, reqs, m = run(mode, qps=5)
    assert m["n_done"] == len(reqs) and m["n_dropped"] == 0
    assert eng.kv.used == 0
    for r in reqs:
        assert r.generated == r.output_len
        assert r.first_token >= r.arrival and r.finish >= r.first_token


def test_preemption_under_memory_pressure_still_completes():
    eng, reqs, m = run("continuous", qps=20, num_blocks=1500)   # tiny KV cache
    assert m["preemptions"] > 0
    assert eng.kv.used == 0
    assert m["n_done"] + m["n_dropped"] == len(reqs)
    for r in reqs:
        if r.finish is not None:
            assert r.generated == r.output_len


def test_admission_control_sheds_load():
    _, _, m = run("continuous", qps=400, dur=10, max_queue=50)
    assert m["n_dropped"] > 0


def test_continuous_beats_static_under_load():
    _, _, mc = run("continuous", qps=40, dur=30)
    _, _, ms = run("static", qps=40, dur=30)
    assert mc["ttft_p99"] < ms["ttft_p99"]
    assert mc["goodput_req_s"] > ms["goodput_req_s"]


def test_chunk_budget_tpot_tradeoff():
    # Smaller chunk budget enforces smoother decode steps (lower TPOT)
    _, _, m_small = run("continuous", qps=20, dur=15, max_num_batched_tokens=256)
    _, _, m_large = run("continuous", qps=20, dur=15, max_num_batched_tokens=4096)
    assert m_small["tpot_p99"] < m_large["tpot_p99"]

