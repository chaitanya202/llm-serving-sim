**Classical Erlang-C says serving 100 QPS of Llama-3-8B takes 161 H100s. My simulator says 3.**

The interesting part is that the textbook model is wrong in two different ways.

I built an open-source discrete-event simulator (llm-serving-sim) that models vLLM-style serving: continuous batching, paged KV cache, chunked prefill, and a roofline timing model. Then I asked how far off telecom-style Erlang-C sizing is.

Setup: Llama-3-8B, H100 SXM, 512 prompt / 200 output tokens, 100 QPS, SLO of TTFT < 500 ms and TPOT < 50 ms.

**1. One request per GPU → 161 GPUs**
An isolated request takes ~1.5 s of GPU time, so offered load is 100 × 1.495 = 149.5 Erlangs. Treating each GPU as a single server needs 161 GPUs (~$4.8M at ~$30k each). Continuous batching interleaves many sequences per GPU, and the simulation meets the SLO with 3 (P99 TTFT 262 ms).

Nobody serves this way, so the 54x is a strawman. Point 2 is the real lesson.

**2. Adding batching "slots" doesn't fix it**
Give each GPU 32 slots and Erlang-C asks for 5 GPUs. The slot count is a guess, and service time isn't constant. Decode steps get heavier as batches grow, so P99 TPOT went from 11.5 ms at 10 GPUs to 24.7 ms at the 3-GPU knee. (The naive 161-GPU cluster has ~7 ms TPOT, so batching trades per-token latency for cost.)

**Caveats:** this is simulation only, and the roofline constants aren't validated against real hardware yet. Arrivals are Poisson, and 3 GPUs runs near saturation with no redundancy or burst headroom. Real-vLLM validation is next.

If you do capacity planning for GPU inference, what do you use today?

#AIInfrastructure #vLLM #QueueingTheory #MLSystems

Attach: results/capacity_planner.png. Repo link in the first comment.
