Part 2: Same GPU, same load. Changing one scheduler setting made P99 token latency 2.5x worse.

In Part 1 I argued Erlang-C treats service time as a fixed property of the server, and a GPU doesn't work that way. This is the cleanest example I found in my simulator.

I thought chunk size was just a knob to tune TTFT. I didn't expect that letting a large prefill chunk through would make active decodes stutter by 2.5x.

Setup: Llama-3-8B, 1x H100, ~512-token prompts, ~200-token outputs, 30 QPS. The only change is the chunked-prefill token budget (max_num_batched_tokens), meaning how much prompt work can share a step with running decodes.

- Budget 256: P99 TTFT 341 ms, P99 TPOT 10.1 ms
- Budget 512: 215 ms, 15.8 ms
- Budget 1024: 187 ms, 20.6 ms
- Budget 4096+: 211 ms, 25.3 ms

A small budget splits each prompt across many steps. Decode stays smooth, but the prompt waits longer for its first token. A large budget lands a whole prompt in one big step, and every running decode waits behind it. P99 step time went from 10 ms to 75 ms. In this setup, 512-1024 was the sweet spot.

Why this matters for capacity planning: the "service time" you'd feed a queueing model isn't fixed by the hardware. It depends on batch composition and scheduler config. Same GPU, same traffic, 2.5x different per-token latency.

Caveats: simulation only, roofline constants not yet validated on real hardware, and one workload (512 in / 200 out). The best budget will move with prompt length and real kernels.

If you run vLLM in production, what max_num_batched_tokens do you use, and did you tune it?

Part 1: [link to your first post]
Repo: https://github.com/chaitanya202/llm-serving-sim

#vLLM #LLMInference #AIInfrastructure
