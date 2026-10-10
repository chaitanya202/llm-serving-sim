# Multi-Part Engineering Series: LLM Serving Architecture & Capacity Planning

A 4-part technical series designed for LinkedIn, Substack, or engineering blogs. It demonstrates staff-level systems depth, intellectual honesty, hardware-grounded reasoning, and empirical rigor without buzzwords or unverified claims.

---

## 📌 Post 1: Why Classical Queueing Theory Fails for LLMs ($M/M/c$ vs. Continuous Batching)

**Best Visual to Attach**: `results/capacity_planner.png` (specifically Panel 1 & Panel 3)

### Draft:
When systems engineers transition from telecom, networking, or distributed backend systems into AI infrastructure, the first natural instinct for capacity planning is queueing theory—specifically the classical Erlang-C ($M/M/c$) model that has sized telephone trunks and web server pools for decades.

I recently built an open-source discrete-event simulator (`llm-serving-sim`) to model GPU inference under load. One of the first questions I wanted to answer mathematically was:  
*How accurate is classical Erlang-C when sizing an LLM serving cluster?*

The answer: It diverges by more than **50x**, for two fundamental reasons.

### 1. The 1-Request-per-Server Trap (53.7x Over-provisioning)
In an M/M/c queue, each server handles strictly one customer at a time.  
For Llama-3-8B on an NVIDIA H100 SXM, an isolated request (512 prompt tokens, 200 output tokens) takes ~1.495 seconds of GPU time.
- If your target ingress is 100 QPS, the offered traffic intensity is:  
  A = lambda * E[S] = 100 * 1.495 = 149.5 Erlangs
- Under classical Erlang-C, maintaining a P99 queue wait under 500 ms requires **at least 161 GPUs** (~$4.8M in hardware).
- However, modern serving engines (like vLLM) implement **Continuous Batching** and **PagedAttention**. A single GPU doesn't process 1 sequence at a time—it dynamically interleaves dozens of sequences concurrently.
- Simulating the identical workload with continuous batching reveals that **3 GPUs** easily handle the 100 QPS load while maintaining TTFT P99 <= 262 ms and TPOT P99 <= 24.7 ms.  
- That is a **53.7x hardware consolidation factor** unlocked purely by continuous batching.

### 2. The Constant-Service-Time Fallacy (Under-estimating Latency)
What if you try adapting Erlang-C by assuming each GPU provides 32 static concurrency slots (c = 32 * N_GPUs)?
Classical queueing theory breaks down here as well because it assumes service time is **load-independent**.

On a physical GPU, that assumption violates hardware physics:
- **Prefill** is compute-bound (Tensor Core FLOPs).
- **Decode** is memory-bound (HBM3 bandwidth). For every token generated, the GPU must stream model weights plus all active conversational histories (KV Cache).
- As cluster size drops from 10 to 2 GPUs, concurrent batch density increases. Reading active KV caches from HBM3 memory inflates Time-Per-Output-Token (TPOT) by **4.3x** (from 11.5 ms up to 49.3 ms/token).

Erlang-C assumes processing speed is invariant to load; GPU memory bandwidth dictates that it isn't.

Repo, HLD/LLD docs, and full simulation suite linked below. What queueing models or simulation tools is your team using for GPU capacity planning?

#MLSystems #AIInfrastructure #vLLM #GPU #QueueingTheory #SystemsEngineering

---

## 📌 Post 2: Slicing the Forward Pass: The Chunked-Prefill Pareto Frontier (TTFT vs. TPOT)

**Best Visual to Attach**: `results/chunk_budget_sweep.png`

### Draft:
In LLM serving, user experience is governed by two competing metrics:
1. **TTFT (Time-To-First-Token)**: How fast the model begins speaking (prompt responsiveness).
2. **TPOT (Time-Per-Output-Token)**: How smoothly tokens stream to the screen without stuttering.

The fundamental architectural dilemma:  
When a user submits a long prompt (e.g., 2,000 tokens), processing it all at once hogs the Tensor Cores for 60+ ms. During that prefill computation, every other user currently receiving streamed words is frozen waiting for the GPU—a phenomenon known as **decode jitter**.

To quantify this trade-off, I ran an empirical sweep on `llm-serving-sim` across token budgets (`max_num_batched_tokens` from 256 to 8192) under sustained load (Llama-3-8B on H100 SXM, 30 QPS, ~512 prompt tokens):

Here is what the empirical frontier revealed:

- **At Budget = 256 tokens**:  
  Decode steps are ultra-fast (10.1 ms), but prompts are sliced into too many micro-chunks across multiple iterations. P99 TTFT degrades to **341.1 ms**.
- **At Budget = 4096+ tokens**:  
  Large prefill bursts monopolize forward passes. Forward-pass step latency spikes to 75.5 ms, inflating P99 TPOT by **2.5x** (25.3 ms/token). Users notice the streaming text stutter.
- **The Pareto Sweet Spot (512 to 1024 tokens)**:  
  Prompts finish in 1–2 iterations while step duration remains bounded (< 35 ms). This achieves the minimal P99 TTFT (**187.1 ms**) while keeping decode streaming responsive.

**Takeaway for Platform Engineers**:  
Setting `max_num_batched_tokens` too high destroys decode smoothness; setting it too low adds unnecessary scheduling overhead and queuing latency. Sizing your chunk budget around your median prompt length is critical for balancing responsiveness against streaming jitter.

Code and sweep script in the repo comments.

#LLMServing #vLLM #PerformanceEngineering #LatencyOptimization #MachineLearning

---

## 📌 Post 3: PagedAttention & Prefix Caching: Lessons from Building an O(1) LRU Memory Manager

**Best Visual to Attach**: `results/prefix_caching_eval.png`

### Draft:
Every token an LLM generates requires reading its conversation history from GPU memory. For Llama-3-8B (GQA with 8 KV heads, 32 layers), that translates to **128 KB of KV Cache per token**.

A single request with a 512-token prompt and 200-token output consumes ~91 MB. 100 concurrent requests consume nearly 10 GB of physical VRAM.

In multi-turn chat and RAG pipelines, prompts often share identical system instructions (e.g., a 384-token system prompt). Recomputing those 384 tokens on every request wastes valuable Tensor Core FLOPs and duplicates memory.

While implementing PagedAttention with prefix caching and block hashing for `llm-serving-sim`, I hit a classic systems engineering bottleneck that is worth sharing:

### The Bug: The O(N) Linear Scan Trap
In our initial implementation, the block allocator partitioned memory into 16-token pages (28,785 physical blocks on an H100). When allocating memory under pressure, the engine scanned all 28,000 blocks to find unreferenced pages (`ref_count == 0`) available for eviction.

At 25 QPS, running a linear scan over 28,000 objects thousands of times per simulated second caused the simulation to freeze.

### The Fix: O(1) LRU Eviction with OrderedDict
We refactored the allocator into a strict three-tier layout:
1. A **Free Set** (`Set[int]`) for unallocated blocks (O(1) pop/push).
2. A **Hash Table** (`Dict[int, Block]`) mapping cryptographic block content hashes for instant prefix matching.
3. An **Evictable LRU Pool** (`collections.OrderedDict[int, Block]`):
   - When a request completes, if `ref_count == 0`, the block is moved to the back of the OrderedDict in O(1).
   - When physical memory is exhausted, the oldest unreferenced page is popped from the front (`popitem(last=False)`) in O(1).

Simulation time dropped from minutes to under **2 seconds**.

### The Empirical Result:
Evaluating prefix caching across shared prefix ratios (0% to 100%, 384-token shared context at 25 QPS):
- **P99 TTFT dropped by 42.6%** (from 144.6 ms down to 83.0 ms).
- Over **191,000 redundant prefill tokens** were completely eliminated from the GPU pipeline.

Systems takeaway: Whether in a production C++ engine or a discrete-event Python simulator, data structure layout and asymptotic time complexity on the allocation hot path dominate runtime performance.

#SoftwareEngineering #DataStructures #vLLM #PagedAttention #Python #SystemsArchitecture

---

## 📌 Post 4: Building `llm-serving-sim`: An Open-Source Discrete-Event Simulator & Capacity Planner

**Best Visual to Attach**: System Architecture Diagram from `docs/HLD.md`

### Draft:
How many H100 GPUs do you need to serve 100 QPS of Llama-3-8B under a 500 ms TTFT SLO?

Until recently, answering this required either renting a GPU cluster for trial benchmarks or relying on static approximations that don't account for continuous batching, memory bandwidth saturation, or KV cache preemption.

Over the past few weeks, I built and verified **`llm-serving-sim`**: a zero-dependency discrete-event simulator and capacity planning engine.

### What is modeled under the hood:
1. **Roofline Hardware Physics**: Fused step durations calculated from Tensor Core FLOP limits (compute-bound prefills) and HBM3 memory bandwidth (memory-bound decodes).
2. **vLLM-Style Runtime**:
   - PagedAttention block allocation (16 tok/block).
   - O(1) LRU prefix caching with block hash chaining.
   - Chunked prefill scheduling (`max_num_batched_tokens`).
   - Recompute-style preemption under memory pressure.
3. **Queueing Theory Integration**:
   - Offered traffic calculations in Erlangs (A = lambda * E[S]).
   - Numerically stable log-space Erlang-C (M/M/c) solver using Log-Gamma and log-sum-exp arithmetic.
   - Empirical divergence tracking between classical queuing models and GPU serving.

### Code Quality & Artifacts:
- **Comprehensive Docs**: Full [Concepts Guide], [High-Level Design (HLD)], and [Low-Level Design (LLD)] documentation.
- **100% Test Coverage**: 15 unit tests covering mathematical invariants, memory conservation, preemption recovery, and scheduling policies.
- **Zero Heavy Dependencies**: Pure Python with only `numpy` and `matplotlib` for reproducibility.

The entire codebase, benchmark sweeps, and design documentation are available on GitHub [link in comments].

I'd love feedback from ML infra and serving engineers: What features would be most valuable to add next—tensor parallel all-reduce communication overhead, disaggregated prefill/decode (split clusters), or timing calibration against real vLLM traces?

#OpenSource #AIInfra #MachineLearningSystems #CapacityPlanning #SoftwareArchitecture
