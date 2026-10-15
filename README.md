# llmsim — LLM serving simulator & capacity planner

A discrete-event simulator of a vLLM-style inference engine (continuous batching,
paged KV cache, chunked prefill, prefix caching, preemption) with a hardware
roofline timing model. Built to model serving dynamics (compute FLOPs, HBM3
bandwidth, KV cache memory pressure, queueing) and evaluate GPU cluster sizing
under latency SLOs.

## Documentation
- [Concepts & Hardware Intuition](docs/CONCEPTS.md): Notes on Tensor Cores, FLOPs, HBM3 memory bandwidth, KV caching, and continuous batching.
- [High-Level Design (HLD)](docs/HLD.md): Subsystem roles, component contracts, and request lifecycle flows.
- [Low-Level Design (LLD)](docs/LLD.md): Block layout, $O(1)$ LRU eviction pool, scheduler loop, and log-space Erlang-C math.

---

## Quick start
```bash
pip install numpy matplotlib pytest
PYTHONPATH=. python experiments/latency_vs_load.py       # Static vs Continuous batching
PYTHONPATH=. python experiments/chunk_budget_sweep.py    # TTFT vs TPOT chunk sweep
PYTHONPATH=. python experiments/prefix_caching_eval.py   # Prefix caching evaluation
PYTHONPATH=. python experiments/capacity_planner_demo.py # Erlang-C vs Continuous batching
python -m pytest tests/ -v                              # Run all 15 unit tests
```

## System Architecture

```
                                  ┌───────────────────────────┐
                                  │   Workload Generator      │
                                  │   (Poisson + Lognormal)   │
                                  └─────────────┬─────────────┘
                                                │ Request Stream
                                                ▼
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│     Capacity Planner      │────▶│ Continuous Batching Engine│◀────│      KV Cache Manager     │
│   (Erlang-C vs. vLLM)     │     │  (Chunked Prefill + Sched)│     │(PagedAttention + O(1) LRU)│
└───────────────────────────┘     └─────────────┬─────────────┘     └───────────────────────────┘
                                                │ Iteration Steps
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Roofline Timing Model   │
                                  │ (FLOPs Peak vs HBM3 BW)   │
                                  └─────────────┬─────────────┘
                                                │ Execution Times
                                                ▼
                                  ┌───────────────────────────┐
                                  │    Metrics & SLO Engine   │
                                  │ (TTFT, TPOT, Goodput P99) │
                                  └───────────────────────────┘
```

| Component | File | Responsibility |
|---|---|---|
| **Workload** | `llmsim/workload.py` | Poisson arrivals, lognormal prompt/output token distributions, shared prefixes. |
| **Timing** | `llmsim/timing.py` | Roofline physics: $\max(\text{FLOPs/peak}, \text{bytes/BW}) + \text{overhead}$; weights + KV reads dominate decode. |
| **KV Cache** | `llmsim/kvcache.py` | PagedAttention allocator (16 tok/block), 64-bit block hashing, $O(1)$ `OrderedDict` LRU pool. |
| **Engine** | `llmsim/engine.py` | Continuous batching runtime, token-budget chunked prefill, recompute preemption. |
| **Metrics** | `llmsim/metrics.py` | Telemetry tracker: TTFT, TPOT, tail latencies (P50, P90, P99), and **goodput under SLO**. |
| **Planner** | `llmsim/planner.py` | Telecom Erlang-C ($M/M/c$) solver, traffic intensity ($A = \lambda \cdot E[S]$), cluster sizing divergence. |

## First result: static vs continuous batching
Llama-3-8B, 1×H100, prompt≈512, output≈200, SLO: TTFT<500 ms and TPOT<50 ms.

![latency vs load](results/latency_vs_load.png)

- Continuous batching meets the TTFT SLO up to ~30 QPS and saturates near ~40 QPS.
- Static batching (batch=32) misses the SLO even at 2 QPS: a request that arrives
  mid-batch waits for the longest sequence in the running batch to finish.
- Beyond saturation, queueing delay grows without bound (TTFT p99 in the tens of seconds).
- Mean KV utilization stays below ~35%: on this workload the engine is limited by
  `max_num_seqs` and step time, not KV memory. Longer contexts would flip that.

## Second result: chunked-prefill budget sweep (TTFT vs TPOT)
Sweep of `max_num_batched_tokens` from 256 to 8192 (Llama-3-8B, 1x H100, prompt≈512, output≈200, QPS=15 and 30).
Run: `PYTHONPATH=. python experiments/chunk_budget_sweep.py`

![chunk budget sweep](results/chunk_budget_sweep.png)

### Key findings:
| Chunk Budget | P99 TTFT (QPS=30) | P99 TPOT (QPS=30) | P99 Step Forward-Pass Latency | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **256** | 341.1 ms | **10.1 ms** | **10.1 ms** | Prompts split across multiple steps; decode stays fast. |
| **512** | 215.0 ms | 15.8 ms | 17.6 ms | Balanced prefill and decode stepping. |
| **1024** | **187.1 ms** | 20.6 ms | 34.3 ms | Lowest TTFT in this setup; step duration remains bounded. |
| **2048** | 198.7 ms | 24.8 ms | 67.5 ms | Heavier prefill steps begin delaying concurrent decodes. |
| **4096+** | 211.0 ms | 25.3 ms | 75.5 ms | Large prefill bursts delay running decodes, raising TPOT 2.5x. |

- For Llama-3-8B on an H100 with ~512-token prompts, `max_num_batched_tokens = 512..1024` gave the lowest TTFT without causing decode jitter.

## Third result: prefix caching evaluation (shared system prompt / RAG)
Evaluation of PagedAttention prefix caching with $O(1)$ LRU block eviction across shared prefix ratios (0% to 100%, 384-token shared system prompt, QPS=25).
Run: `PYTHONPATH=. python experiments/prefix_caching_eval.py`

![prefix caching evaluation](results/prefix_caching_eval.png)

### Key findings:
| Shared Prefix Ratio | Prefix Hit Rate | Prefill Tokens Saved | P99 TTFT (No Cache) | P99 TTFT (With Cache) | TTFT Reduction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0%** | 0.0% | 0 | 144.6 ms | 144.6 ms | 0.0% |
| **25%** | 99.4% | 49,200 | 144.6 ms | 134.0 ms | **-7.3%** |
| **50%** | 99.7% | 98,592 | 144.6 ms | 109.5 ms | **-24.2%** |
| **75%** | 99.8% | 145,872 | 144.6 ms | 84.4 ms | **-41.6%** |
| **100%** | 99.8% | 191,328 | 144.6 ms | 83.0 ms | **-42.6%** |

- Prefix caching reduced P99 TTFT by 42.6% on workloads with a shared system prompt and eliminated redundant prefill FLOPs.
- Unreferenced cached blocks sit in an $O(1)$ `OrderedDict` LRU pool and evict when physical KV capacity fills.

## Fourth result: capacity planner (Erlang-C vs continuous batching)

Compares Erlang-C ($M/M/c$) sizing against the continuous-batching simulation.
Llama-3-8B on H100 SXM, prompt≈512, output≈200, 100 QPS target,
SLO: TTFT<500 ms and TPOT<50 ms. **Simulation only, constants unvalidated
against real hardware (see Known limitations).**
Config: `max_num_seqs = 256`, chunk budget = 512, routing = uniform Poisson splitting ($\lambda / N$).
Run: `PYTHONPATH=. python experiments/capacity_planner_demo.py`

![capacity planner](results/capacity_planner.png)

| Sizing model | Min GPUs | P99 latency metric | P99 TPOT | Notes |
| --- | --- | --- | --- | --- |
| **Single-request-per-GPU Erlang-C ($c = N$)** | 161 | queue wait < 500 ms | ~7.4 ms | Strawman for contrast, not how anyone serves LLMs. Offered load 149.5 Erlangs. ~$4.8M at an assumed ~$30k per GPU. |
| **Slotted Erlang-C (32 slots/GPU, $c = 32N$)** | 5 | queue wait 500.2 ms | not modeled | Asks for more GPUs than the simulation needs (5 vs 3). Depends on the slot count, which is a free parameter. |
| **Simulation (continuous batching)** | 3 | TTFT 262.6 ms | 24.7 ms | Sizing knee: 2 GPUs fail the SLO. Runs near saturation with no redundancy or burst headroom. |

### Cluster scaling (100 QPS workload)

| GPUs in Cluster | Offered QPS/GPU | Slotted Erlang-C P99 queue wait (ms) | Sim P99 TTFT (ms) | Sim P99 TPOT (ms) | Peak KV Util | SLO Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 50.0 | Overloaded | 2,206.4 | 49.3 | 38.9% | FAIL (TTFT Exceeded) |
| **3** | 33.3 | Overloaded | **262.6** | **24.7** | 23.9% | **PASS (Sizing Knee)** |
| **4** | 25.0 | Overloaded | 131.9 | 18.7 | 12.5% | PASS |
| **5** | 20.0 | 500.2 | 117.1 | 14.4 | 9.4% | PASS |
| **6** | 16.7 | 17.6 | 135.2 | 13.8 | 6.7% | PASS |
| **8** | 12.5 | 17.6 | 125.8 | 12.2 | 6.3% | PASS |
| **10** | 10.0 | 17.6 | 126.8 | 11.5 | 5.4% | PASS |

*(Note: Slotted Erlang-C reports queue wait alone, whereas simulation reports TTFT which includes queue wait plus prefill).*

### Findings

- **Against the single-request-per-GPU baseline** the simulation needs 3 GPUs instead of 161 (53.7x). That baseline is a strawman. Against slotted Erlang-C the gap is 5 vs 3 GPUs (1.7x), and it depends on the slot count.
- **Service time is not load-independent.** P99 TPOT is 11.5 ms at 10 GPUs, 24.7 ms at the passing 3-GPU knee (2.1x), and 49.3 ms at 2 GPUs, which fails the TTFT SLO (4.3x). Whether the rise comes from KV/weight reads or from prefill chunks sharing steps with decode has not been decomposed yet (see Roadmap).
- **Fixed slots are arbitrary.** 96 slots < 149.5 Erlangs, so slotted Erlang-C calls 3 GPUs overloaded while the simulation passes at 23.9% peak KV utilization. A different slot count would change this result.
- **Batching trades per-token latency for cost.** The 161-GPU cluster has ~7.4 ms TPOT vs 24.7 ms at 3 GPUs.
- **At 3 GPUs (33.3 QPS per GPU)**, the workload operates right at the passing knee ($\approx 262\text{ ms} \le 500\text{ ms}$ SLO) before the steep queueing cliff observed at 40+ QPS on single-GPU runs.

## Known limitations (read before quoting numbers)
- Results are simulated and not yet validated against real vLLM.
- Arrivals are Poisson only. Real traffic is burstier, which raises the GPU count needed for the same tail latency.
- Sizing finds the minimum GPUs that pass the SLO. It includes no N+1 redundancy and no utilization headroom.
- Slotted Erlang-C results depend on the chosen slot count (32 here).
- Erlang-C reports queue wait. The simulation reports TTFT (queue wait plus prefill). They are not the same metric.
- The ~$30k per GPU figure is an assumption, and excludes host, networking, and power.
- Roofline efficiencies (`bw_eff=0.75`, `flops_eff=0.5`, 1 ms step overhead) are assumptions.
- Attention FLOPs, tensor-parallel communication, and activation memory are ignored.
- The static-batching baseline is deliberately classic (padding until the longest sequence finishes); it is a lower bound on what naive serving looks like.
- Rates (goodput, throughput) divide by total simulated time including queue drain, so overloaded points are pessimistic.
- Preemption is recompute-style only (no swap-to-CPU).

## Roadmap
- [x] Prefix caching (block hashing + LRU) and its hit-rate/TTFT effect
- [x] Sweep `max_num_batched_tokens` (chunk size) — TTFT vs TPOT tradeoff
- [x] Capacity planner: roofline + queueing model (Erlang-C first-order, show where it diverges)
- [ ] Decompose P99 TPOT into decode-only steps vs steps shared with prefill chunks, to attribute the 11.5 -> 49.3 ms rise
- [ ] Sweep slotted Erlang-C slot count (16, 32, 64, 128, 256) vs simulation
- [ ] Reconcile the first-result saturation (~30-40 QPS/GPU) with the 33.3 QPS/GPU pass at 3 GPUs in the capacity planner
- [ ] Headroom-aware sizing (N+1, target utilization <= 70-80%)
- [ ] Bursty arrivals (gamma / MMPP) in addition to Poisson
- [ ] Calibrate timing model against llama.cpp on CPU (validates *method*, not H100 constants)
- [ ] One-hour real-vLLM run on a rented GPU to validate p99 TTFT prediction
- [ ] Compare against Vidur and cite differences


