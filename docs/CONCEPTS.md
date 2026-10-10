# LLM Serving & Systems Engineering Concepts Guide

This document explains the foundational concepts behind modern Large Language Model (LLM) serving, GPU architecture physics, and capacity planning. It assumes no prior deep systems engineering background.

---

## 1. GPU Hardware Fundamentals

A Graphics Processing Unit (GPU) like the **NVIDIA H100 SXM** is essentially a massive parallel calculator optimized for matrix math.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA H100 GPU (80 GB)                         │
│                                                                        │
│   ┌───────────────────────────┐      High-Bandwidth     ┌────────────┐ │
│   │   HBM3 High-Speed RAM     │   Memory Bus (HBM3)     │Tensor Cores│ │
│   │         (80 GB)           │ ═══════════════════════▶│(Math Engine│ │
│   │ Stores Weights + KV Cache │      3.35 TB/s          │ 989 TFLOP/s│ │
│   └───────────────────────────┘                         └────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### A. Compute: FLOPs and Tensor Cores
- **FLOP**: Floating-Point Operation (e.g., $1.414 \times 3.141$).
- **TFLOP/s**: Tera-FLOPs per second ($10^{12}$ math operations per second).
- **Tensor Cores**: Specialized on-chip hardware circuits designed strictly to multiply and add small matrices together in a single clock cycle.
- **Peak Compute Capacity**: The absolute maximum arithmetic operations the GPU can physically perform per second. For an H100 in FP16, this is **$989\text{ TFLOP/s}$** (realistic kernel efficiency is around $\sim 50\%$, or $\approx 495\text{ TFLOP/s}$).

### B. Memory: HBM3 and Bandwidth
- **HBM3 (High Bandwidth Memory)**: Specialized 3D-stacked RAM chips connected directly to the GPU silicon via a silicon interposer.
- **Memory Bandwidth (BW)**: How fast bytes can travel between HBM3 RAM and the GPU cores.
- **Bandwidth Rating**: The H100 can stream data at up to **$3.35\text{ TB/s}$** ($3,350\text{ GB/s}$). Realistic kernel efficiency is $\sim 75\%$, or $\approx 2,510\text{ GB/s}$.
- **SRAM vs. HBM3**:
  - **SRAM (On-Chip Register/Shared Memory)**: Super fast ($\sim 50\text{ TB/s}$), but tiny ($\sim 50\text{ MB}$).
  - **HBM3 (Main GPU VRAM)**: Fast ($\sim 3.35\text{ TB/s}$), and large ($80\text{ GB}$).
  - Every calculation requires loading data from HBM3 into SRAM, running the math on Tensor Cores, and storing results back into HBM3.

---

## 2. Model Physics: Weights and Parameters

### Where does the "16 GB" model weight come from?
- A model like **Llama-3-8B** has approximately **8 Billion parameters** ($8,030,000,000$ numbers).
- In standard deep learning precision (**FP16** or **BF16**), each parameter is a 16-bit float:
  $$16\text{ bits} = 2\text{ bytes}$$
- Total size of the model weights:
  $$8,030,000,000 \times 2\text{ bytes} \approx 16.06\text{ GB}$$
- **Key Insight**: These 16 GB of weights are **static and read-only**. They are loaded into GPU memory once at server startup and are shared by all incoming user requests.

---

## 3. The Two Lifecycle Phases: Prefill vs. Decode

Every single LLM interaction consists of two completely different algorithmic phases:

```
User Prompt: "Explain quantum mechanics in 100 words"
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   Phase 1: PREFILL        Phase 2: DECODE
  (Prompt Evaluation)    (Autoregressive Generation)
  - Process all 8 tokens  - 1 token generated per step
  - Parallel matrix math  - Sequential (100 steps)
  - Compute-Bound         - Memory-Bound
```

### Phase 1: Prefill (Prompt Processing)
- The user sends a prompt of $N$ tokens (e.g., 512 tokens).
- The GPU processes **all 512 tokens simultaneously**.
- Math required: $2 \times \text{Params} \times \text{Tokens} = 2 \times 8\text{B} \times 512 \approx 8.19\text{ TFLOPs}$.
- Because all 512 tokens are processed in one giant matrix multiplication, the math cores are fully saturated.
- **Bottleneck**: **Compute-Bound (FLOPs)**.

### Phase 2: Decode (Token Generation)
- Generating words is auto-regressive: word $K$ depends on words $1 \dots K-1$.
- Tokens are produced **strictly one-by-one**. To generate 100 words, the GPU must execute **100 consecutive forward passes**.
- For every single word generated, the GPU must stream the entire **16 GB model weights** through its memory bus!
- Moving 16 GB takes $\sim 6.4\text{ ms}$, while the math for 1 token takes $\sim 0.03\text{ ms}$.
- **Bottleneck**: **Memory-Bound (HBM3 Bandwidth)**.

---

## 4. The KV Cache: What It Is and Why It Grows

### Why do we need a KV Cache?
In the Transformer architecture, computing the next token requires calculating Attention across all previous tokens.
- Each token generates a **Key (K)** vector and a **Value (V)** vector.
- Without a cache, generating token #100 would require recalculating tokens 1 through 99 from scratch ($O(N^2)$ recomputation!).
- Instead, the GPU saves the K and V vectors in memory: this is the **KV Cache**.

### How big is the KV Cache?
For Llama-3-8B with Grouped Query Attention (GQA):
- Layers: $32$
- KV Heads: $8$
- Head Dimension: $128$
- Data Type: 16-bit ($2\text{ bytes}$)
$$\text{KV Bytes per Token} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes} = 2 \times 32 \times 8 \times 128 \times 2 = 131,072\text{ bytes } (\mathbf{128\text{ KB/token}})$$

### The Memory Footprint:
- 1 request with 512 prompt tokens + 200 output tokens ($712\text{ tokens}$ total):
  $$712 \times 128\text{ KB} \approx \mathbf{91.1\text{ MB}}$$
- 100 concurrent requests:
  $$100 \times 91.1\text{ MB} \approx \mathbf{9.11\text{ GB}}$$
- 500 concurrent requests:
  $$500 \times 91.1\text{ MB} \approx \mathbf{45.5\text{ GB}}$$

As conversations grow longer or concurrent users increase, the KV Cache quickly consumes all available GPU memory!

---

## 5. PagedAttention & Prefix Caching

### Traditional Memory Allocation vs. PagedAttention
- **Naive Allocation**: Early engines pre-allocated contiguous memory for the user's maximum possible output length (e.g., 4,096 tokens = 512 MB). If the user only asked a 10-word question, $98\%$ of that memory sat idle.
- **PagedAttention (vLLM)**: Breaks the KV cache into fixed-size physical pages or **blocks** (e.g., 16 tokens = 2 MB per block). Blocks are allocated dynamically on-demand, virtually eliminating internal memory waste.

```
User Prompt (36 tokens):
Block 0: [Tokens 1-16]  --> Physical Block #402 (Full)
Block 1: [Tokens 17-32] --> Physical Block #109 (Full)
Block 2: [Tokens 33-36] --> Physical Block #850 (4/16 used)
```

### Prefix Caching
If 100 users chat with a customer support bot, every prompt begins with the exact same system instructions (e.g., 384 tokens):
- **Without Prefix Caching**: The GPU redundantly runs prefill FLOPs and allocates 24 duplicate blocks for every user.
- **With Prefix Caching**: The engine hashes block tokens (`hash(tokens)`). All 100 users point their logical block tables to the **same physical block** (`ref_count = 100`).
- **Result**: Eliminates prefill compute time and saves gigabytes of VRAM.

---

## 6. Static Batching vs. Continuous Batching

```
STATIC BATCHING (Naive):
Request 1 (10 tokens):  ██ (Waits idle for 90 steps...)           │ Done
Request 2 (100 tokens): ████████████████████████████████████████ │ Done
New Request 3 Arrives:  ──────(BLOCKED OUTSIDE BATCH)───────────▶ │ Finally Starts
                        0                                        100 steps

CONTINUOUS BATCHING (Iteration-Level):
Request 1 (10 tokens):  ██ (Leaves at step 10)
Request 3 (Joins):        ██████████████ (Starts immediately at step 11!)
Request 2 (100 tokens): ████████████████████████████████████████
                        0                                        100 steps
```

- **Static Batching**: Waits for the longest sequence in the batch to complete before accepting new requests. Results in massive queue times and low GPU utilization.
- **Continuous Batching**: Operates at the iteration step level. Finished requests leave immediately, and newly arrived requests join the next forward pass.

---

## 7. Serving Metrics & SLOs

When serving models in production, teams define **Service Level Objectives (SLOs)**:

| Metric | Full Name | Definition | Target Goal |
|---|---|---|---|
| **TTFT** | Time-To-First-Token | Time from when the user clicks "Send" until the first word appears on their screen. | $\le 500\text{ ms}$ (P99) |
| **TPOT** | Time-Per-Output-Token | Time between consecutive words during streaming. Dictates reading smoothness. | $\le 50\text{ ms}$ (P99) ($20\text{ tok/s}$) |
| **E2E** | End-to-End Latency | Total time from request submission to final output completion ($\text{TTFT} + \text{Output} \times \text{TPOT}$). | Workload dependent |
| **Goodput** | Goodput under SLO | Requests per second that successfully meet **both** TTFT and TPOT SLO thresholds. | Maximize |
| **Tail Latency** | P50 / P90 / P99 | The 50th, 90th, and 99th percentile latencies. P99 measures worst-case user experience. | Keep bounded |
