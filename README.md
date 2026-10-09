# llmsim — LLM serving simulator & capacity planner

A discrete-event simulator of a vLLM-style inference engine (continuous batching,
paged KV cache, chunked prefill, preemption, admission control) with a roofline
timing model. Goal: answer *"how many GPUs for X QPS under an SLO?"* and show
*why* (batching, KV memory, queueing), without needing a GPU to run.

> **Status: Week 1 (core simulator).** Timing constants are **uncalibrated**
> roofline estimates. Results show *relative* policy behavior, not validated
> H100 numbers. Calibration and real-vLLM validation are planned (see Roadmap).

## Quick start
```bash
pip install numpy matplotlib pytest
PYTHONPATH=. python experiments/latency_vs_load.py   # writes results/*.png/csv
python -m pytest -q
```

## What's modeled
| Component | File | Notes |
|---|---|---|
| Workload | `workload.py` | Poisson arrivals, lognormal prompt/output lengths |
| Timing | `timing.py` | `max(FLOPs/peak, bytes/BW) + overhead` per fused step; weights + KV reads dominate decode |
| KV cache | `kvcache.py` | Block allocator (16 tok/block); capacity derived from GPU mem − weights |
| Engine | `engine.py` | `continuous` (token budget, chunked prefill, recompute preemption) and `static` (padded batches) |
| Metrics | `metrics.py` | TTFT, TPOT, throughput, **goodput under SLO**, KV utilization |

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
- [ ] Capacity planner: roofline + queueing model (Erlang-C first-order, show where it diverges)
- [ ] Calibrate timing model against llama.cpp on CPU (validates *method*, not H100 constants)
- [ ] One-hour real-vLLM run on a rented GPU to validate p99 TTFT prediction
- [ ] Compare against Vidur and cite differences
