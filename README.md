# llmsim — LLM Serving Simulator & Capacity Planner

A discrete-event simulator and capacity planning engine of a modern GPU inference serving stack (**vLLM**-style: continuous batching, paged KV cache, chunked prefill, prefix caching, preemption) with a hardware roofline timing model.

**Core Mission**: Answer *"How many GPUs do I need to serve X requests/sec under strict latency SLOs?"* and demonstrate *why* (GPU compute FLOPs, HBM3 memory bandwidth, KV cache fragmentation, and queueing dynamics) without needing physical GPUs to run trials.

---

## 📚 Deep Dive Documentation

For detailed architectural, low-level, and intuitive walk-throughs:
- **[📖 Concepts & Hardware Intuition Guide](docs/CONCEPTS.md)**: A complete, beginner-friendly explanation of GPU Tensor Cores, FLOPs, HBM3 Bandwidth, Model Weights, Prefill vs. Decode, KV Caching, and Continuous Batching.
- **[🏛️ High-Level Design (HLD)](docs/HLD.md)**: System architecture decomposition, component contracts, end-to-end request lifecycle, and Mermaid sequence diagrams.
- **[⚙️ Low-Level Design (LLD)](docs/LLD.md)**: Algorithmic mechanics, data structures (`Block`, `BlockAllocator`, `Request`, `Engine`), $O(1)$ LRU eviction pool, and log-space Erlang-C mathematics.

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
| **KV Cache** | `llmsim/kvcache.py` | PagedAttention allocator (16 tok/block), 64-bit cryptographic hashing, $O(1)$ `OrderedDict` LRU pool. |
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
| Chunk Budget | P99 TTFT (QPS=30) | P99 TPOT (QPS=30) | P99 Step Forward-Pass Latency | Trade-Off Dynamic |
| :---: | :---: | :---: | :---: | :--- |
| **256** | 341.1 ms | **10.1 ms** | **10.1 ms** | **Smooth Decode, High TTFT**: 512-tok prompts split across multiple steps. |
| **512** | 215.0 ms | 15.8 ms | 17.6 ms | Balanced prefill and decode stepping. |
| **1024** | **187.1 ms** | 20.6 ms | 34.3 ms | **Optimal TTFT Sweet Spot**: Prompt finishes in 1-2 steps; step time remains moderate. |
| **2048** | 198.7 ms | 24.8 ms | 67.5 ms | Heavy compute prefill steps begin delaying concurrent decode tokens. |
| **4096+** | 211.0 ms | 25.3 ms | 75.5 ms | **Decode Jitter Penalty**: Massive forward-pass bursts inflate TPOT by **2.5x**. |

- **The Sweet Spot**: For Llama-3-8B on an H100 with ~512 token prompts, `max_num_batched_tokens = 512..1024` yields the Pareto optimal trade-off between prompt responsiveness (TTFT) and decode smoothness (TPOT).

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

- **Compute & Latency Payoff**: Prefix caching slashes P99 TTFT by over **42%** on multi-turn conversations while completely eliminating redundant prefill FLOPs for shared system prompts.
- **Memory Safety**: Unreferenced cached blocks reside in an $O(1)$ `OrderedDict` LRU pool, evicting seamlessly when physical KV capacity saturates.

## Fourth result: capacity planner (Erlang-C vs continuous batching divergence)
Capacity planning comparing classical telecom Erlang-C ($M/M/c$) queueing theory against continuous batching simulation (Llama-3-8B on H100 SXM, prompt≈512, output≈200, 100 QPS target, SLO: TTFT<500 ms & TPOT<50 ms).
Run: `PYTHONPATH=. python experiments/capacity_planner_demo.py`

![capacity planner](results/capacity_planner.png)

### Key findings:
| Sizing Model | Target QPS | Offered Traffic | Minimum GPUs | P99 TTFT | P99 TPOT | Sizing Dynamic |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Naive Erlang-C ($M/M/c$)** | 100 | 149.5 Erlangs | **161 GPUs** | < 500 ms | ~7.4 ms | **53.7x Over-provisioning**: Assumes 1 isolated request per server ($c=1$). Ignores batching concurrency. |
| **Slotted Erlang-C (32 slots/GPU)** | 100 | 149.5 Erlangs | **5 GPUs** | 500.2 ms | Assumes constant | **Under-estimates Capacity**: Assumes service time is load-independent; misses dynamic KV multiplexing. |
| **Continuous Batching (vLLM)** | 100 | 149.5 Erlangs | **3 GPUs** | **262.6 ms** | **24.7 ms** | **Optimal Real-World Sizing**: PagedAttention multiplexes sequences dynamically, easily beating static slot limits. |

### Cluster Scaling & Memory Bandwidth Inflation (100 QPS Workload):
| GPUs in Cluster | Offered QPS/GPU | Slotted Erlang-C P99 | Sim P99 TTFT | Sim P99 TPOT | Peak KV Util | SLO Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 50.0 | Overloaded | 2,206.4 ms | 49.3 ms | 38.9% | FAIL (TTFT Exceeded) |
| **3** | 33.3 | Overloaded | **262.6 ms** | **24.7 ms** | 23.9% | **PASS (Sizing Knee)** |
| **4** | 25.0 | Overloaded | 131.9 ms | 18.7 ms | 12.5% | PASS |
| **5** | 20.0 | 500.2 ms | 117.1 ms | 14.4 ms | 9.4% | PASS |
| **6** | 16.7 | 17.6 ms | 135.2 ms | 13.8 ms | 6.7% | PASS |
| **8** | 12.5 | 17.6 ms | 125.8 ms | 12.2 ms | 6.3% | PASS |
| **10** | 10.0 | 17.6 ms | 126.8 ms | 11.5 ms | 5.4% | PASS |

- **The Consolidation Dividend**: Continuous batching delivers a **53.7x hardware reduction** (161 GPUs down to 3 GPUs) compared to classical single-channel queueing theory.
- **Why Erlang-C Breaks Down on GPUs**:
  1. *Under-provisioning via Load-Independent Assumptions*: Classical $M/M/c$ assumes service time $E[S]$ is constant regardless of how many slots are occupied. On GPUs, decode step time is bound by HBM3 memory bandwidth reading weights and active KV blocks. As cluster size shrinks from 10 to 2 GPUs, concurrent batch load inflates P99 TPOT by **4.3x** (from 11.5 ms up to 49.3 ms).
  2. *Over-provisioning via Fixed Slotting*: Slotted Erlang-C declared 3 and 4 GPUs "overloaded" because 96 static slots $< 149.5$ Erlangs. In reality, continuous batching dynamically interleaves sequences beyond static batch slots, operating stably at 23.9% peak KV cache utilization.

## Known limitations (read before quoting numbers)
- Roofline efficiencies (`bw_eff=0.75`, `flops_eff=0.5`, 1 ms step overhead) are assumptions.
- Attention FLOPs, tensor-parallel communication, and activation memory are ignored.
- The static-batching baseline is deliberately classic (padding until the longest
  sequence finishes); it is a lower bound on what naive serving looks like.
- Rates (goodput, throughput) divide by total simulated time including queue drain,
  so overloaded points are pessimistic.
- Preemption is recompute-style only (no swap-to-CPU).

## Roadmap
- [x] Prefix caching (block hashing + LRU) and its hit-rate/TTFT effect
- [x] Sweep `max_num_batched_tokens` (chunk size) — TTFT vs TPOT tradeoff
- [x] Capacity planner: roofline + queueing model (Erlang-C first-order, show where it diverges)
- [ ] Calibrate timing model against llama.cpp on CPU (validates *method*, not H100 constants)
- [ ] One-hour real-vLLM run on a rented GPU to validate p99 TTFT prediction
- [ ] Compare against Vidur and cite differences

