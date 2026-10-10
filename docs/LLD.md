# Low-Level Design (LLD): LLM Serving Simulator & Capacity Planner

This document details the low-level data structures, algorithms, mathematical equations, and execution mechanics implemented across the codebase.

---

## 1. Data Structures & Memory Layout

### 1.1. Physical KV Block & Paged Memory (`kvcache.py`)

Memory is structured into fixed-size physical blocks to eliminate external fragmentation and enable prefix sharing.

```python
@dataclass
class Block:
    block_id: int          # Unique physical page index [0 ... N-1]
    tokens: tuple[int, ...] # Sequence of token IDs in this block (len <= block_size)
    hash: int              # Chained 64-bit cryptographic hash of block contents
    ref_count: int         # Number of active sequences currently referencing this page
```

#### Key Properties:
- **`block_size`**: Default 16 tokens ($16 \times 128\text{ KB} = 2\text{ MB}$ per block for Llama-3-8B).
- **`ref_count`**:
  - `ref_count > 0`: Actively used by one or more running requests. Cannot be evicted.
  - `ref_count == 0`: Inactive, residing in the LRU eviction pool. Available for immediate reuse if another request has the same prefix, or subject to eviction if memory is exhausted.

---

### 1.2. The $O(1)$ LRU Eviction Pool (`BlockAllocator`)

To prevent performance degradation during large-scale simulation, block allocation and eviction are implemented in strictly $O(1)$ time complexity using a two-tier data structure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BlockAllocator Layout                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  [ Free Pool ]          Set of unallocated block IDs: Set[int]             │
│                         - O(1) pop and push                                 │
│                                                                             │
│  [ Hash Map ]           Hash to Block Mapping: Dict[int, Block]             │
│                         - O(1) prefix lookup                                │
│                                                                             │
│  [ Evictable LRU Pool ] collections.OrderedDict[int, Block]                 │
│                         - O(1) insert to back on ref_count == 0             │
│                         - O(1) popitem(last=False) for oldest eviction      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Algorithmic Invariants:
1. **Allocation with Hash Match ($O(1)$)**:
   - Check if `block_hash in self.cached_blocks`.
   - If present, increment `ref_count += 1`. If it was in `evictable_lru`, remove it from `evictable_lru` in $O(1)$.
2. **Fresh Block Allocation ($O(1)$)**:
   - If `self.free_blocks` is non-empty, pop an unused block ID in $O(1)$.
   - If `self.free_blocks` is empty, evict the oldest entry from `evictable_lru`:
     ```python
     evicted_id, evicted_block = self.evictable_lru.popitem(last=False)
     del self.cached_blocks[evicted_block.hash]
     ```
3. **Deallocation ($O(1)$)**:
   - Decrement `ref_count -= 1`.
   - If `ref_count == 0`, append to `self.evictable_lru[block_id] = block` (least recently used entries stay at the front).

---

### 1.3. Request State Machine (`workload.py` & `engine.py`)

A request transitions through a strict discrete lifecycle:

```
[ WAITING ] ──▶ [ PREFILLING ] ──▶ [ DECODING ] ──▶ [ COMPLETED ]
     ▲                 │                 │
     │                 ▼                 │
     └───────── [ PREEMPTED ] ◀──────────┘
```

```python
@dataclass
class Request:
    request_id: int
    arrival_time: float       # Epoch seconds of arrival
    prompt_len: int           # Total prompt token count
    output_len: int           # Target output token count
    tokens_to_prefill: int    # Remaining prompt tokens needing forward-pass
    tokens_decoded: int       # Tokens generated so far
    allocated_blocks: list[int] # Physical block IDs currently holding this sequence
    first_token_time: float | None # Time first token was generated (TTFT calculation)
    finish_time: float | None      # Time final token was generated (E2E calculation)
```

---

## 2. Low-Level Execution Mechanics & Mathematical Models

### 2.1. The Roofline Timing Equation (`timing.py`)

For every fused iteration step on the GPU, execution duration is computed based on arithmetic work vs. memory traffic:

$$\text{Step Time} = \max\left( t_{\text{compute}}, t_{\text{memory}} \right) + t_{\text{overhead}}$$

#### Compute Time ($t_{\text{compute}}$):
$$t_{\text{compute}} = \frac{2 \times N_{\text{params}} \times (N_{\text{prefill\_tokens}} + N_{\text{decode\_tokens}})}{\text{Peak FLOPS} \times \eta_{\text{flops}}}$$
- $N_{\text{params}} = 8.03 \times 10^9$ parameters.
- $\text{Peak FLOPS} = 989 \times 10^{12}\text{ FLOP/s}$ (H100 FP16).
- $\eta_{\text{flops}} = 0.50$ (realistic Tensor Core sustained efficiency).

#### Memory Time ($t_{\text{memory}}$):
$$t_{\text{memory}} = \frac{\text{Bytes}_{\text{weights}} + \text{Bytes}_{\text{KV\_reads}}}{\text{Peak Bandwidth} \times \eta_{\text{bw}}}$$
$$\text{Bytes}_{\text{weights}} = 16.03 \times 10^9\text{ bytes (read once per step)}$$
$$\text{Bytes}_{\text{KV\_reads}} = N_{\text{decode}} \times \sum_{i=1}^{N_{\text{decode}}} (\text{context\_len}_i) \times \text{KV\_bytes\_per\_token}$$
- $\text{Peak Bandwidth} = 3.35 \times 10^{12}\text{ bytes/s}$ (H100 HBM3).
- $\eta_{\text{bw}} = 0.75$ (realistic memory controller sustained efficiency).
- $\text{KV\_bytes\_per\_token} = 131,072\text{ bytes}$ ($128\text{ KB}$).
- $t_{\text{overhead}} = 1.0\text{ ms}$ (CUDA kernel launch & scheduling overhead).

---

### 2.2. The Continuous Batching Scheduler Loop (`engine.py`)

The heart of the runtime is the discrete iteration loop:

```python
def step(self):
    # 1. Initialize Step Budget
    budget = self.config.max_num_batched_tokens  # e.g., 512 or 1024 tokens
    step_decodes = []
    step_prefills = []

    # 2. Schedule Running Decodes First (Smooth Token Streaming)
    for req in list(self.active_decode):
        if budget <= 0 or len(step_decodes) >= self.config.max_num_seqs:
            break
        # Allocate next token block if sequence crosses 16-token page boundary
        if (req.prompt_len + req.tokens_decoded) % 16 == 0:
            if not self.allocator.can_allocate(1):
                self.preempt(req)  # Reclaim memory under pressure
                continue
            self.allocator.allocate(req, 1)

        step_decodes.append(req)
        budget -= 1

    # 3. Fill Remaining Budget with Chunked Prefills
    while budget > 0 and self.waiting_queue:
        req = self.waiting_queue[0]
        chunk_tokens = min(budget, req.tokens_to_prefill)

        # Allocate blocks required for this chunk
        needed_blocks = math.ceil(chunk_tokens / 16)
        if not self.allocator.can_allocate(needed_blocks):
            break  # Cannot fit next prompt; wait for decodes to complete

        self.allocator.allocate(req, needed_blocks)
        req.tokens_to_prefill -= chunk_tokens
        budget -= chunk_tokens
        step_prefills.append((req, chunk_tokens))

        if req.tokens_to_prefill == 0:
            # Whole prompt evaluated! Promote to active decode pool
            self.waiting_queue.pop(0)
            self.active_decode.append(req)

    # 4. Execute Fused Step via Hardware Roofline
    dt = self.timing.step_time(
        n_prefill=sum(c for _, c in step_prefills),
        n_decode=len(step_decodes),
        total_context=sum(r.prompt_len + r.tokens_decoded for r in step_decodes)
    )

    # 5. Advance Simulation Clock and Update Metrics
    self.t_now += dt
    for req in step_decodes:
        req.tokens_decoded += 1
        if req.first_token_time is None:
            req.first_token_time = self.t_now  # TTFT Recorded
        if req.tokens_decoded >= req.output_len:
            self.finalize_request(req)         # Finished! Release memory
```

---

### 2.3. Log-Space Numerically Stable Erlang-C Math (`planner.py`)

Classical Erlang-C calculates the delay probability $P(\text{Wait} > 0)$ for $c$ servers handling traffic intensity $A$ (Erlangs):

$$P_W = \frac{\frac{A^c}{c!} \frac{1}{1 - A/c}}{\sum_{k=0}^{c-1} \frac{A^k}{k!} + \frac{A^c}{c!} \frac{1}{1 - A/c}}$$

#### The Numerical Overflow Problem:
When $c = 161$, $161! \approx 10^{287}$, causing standard IEEE 754 64-bit floating point numbers to overflow to `inf`.

#### The Log-Space Solution:
We express the terms in natural logarithm space using the Log-Gamma function:
$$\ln\left(\frac{A^k}{k!}\right) = k \ln(A) - \ln\Gamma(k + 1)$$
To sum terms without underflow or overflow, we apply the **Log-Sum-Exp Trick**:
$$M = \max_{k} \left[ \ln\left(\frac{A^k}{k!}\right) \right]$$
$$\sum_{k=0}^{c-1} \frac{A^k}{k!} = \exp(M) \sum_{k=0}^{c-1} \exp\left( \ln\left(\frac{A^k}{k!}\right) - M \right)$$
Dividing both numerator and denominator by $\exp(M)$ ensures every term resides safely between $10^{-12}$ and $1.0$, enabling flawless evaluation up to thousands of servers.

#### P99 Queuing Wait Time:
Given probability of wait $P_W$ and per-server service rate $\mu = 1 / E[S]$:
$$P(W > t) = P_W \cdot e^{-(c\mu - \lambda)t}$$
Solving for the 99th percentile ($P(W > t_{99}) = 0.01$):
$$t_{99} = \frac{\ln(P_W / 0.01)}{c\mu - \lambda} = \frac{\ln(100 \cdot P_W)}{(c - A)\mu}$$
If $c \le A$, the system is unstable and queue delay approaches $\infty$.

---

## 3. Edge Cases & Fault Recovery

| Scenario | Architectural Consequence | Mitigation Mechanism |
|---|---|---|
| **Memory Exhaustion** | GPU runs out of physical KV blocks during a step. | **Recompute Preemption**: Lowest-priority request is evicted to free blocks; recomputes from prompt when memory frees. |
| **100% Prefix Hit** | Whole prompt matches cached prefix (`tokens_to_prefill == 0`). | **Fast-Path Promotion**: Bypasses prefill scheduler immediately into `active_decode` with 1-token budget deduction. |
| **Decode Jitter** | Massive 4,000-token prompt evaluates in single step, stalling decodes. | **Chunked Prefill**: Limits prompt evaluation to `max_num_batched_tokens = 512`, distributing prompt work across consecutive steps. |
| **Numerical Overflow** | Cluster size $> 100$ in Erlang-C math. | **Log-Gamma Arithmetic**: All factorial terms computed via `math.lgamma(n + 1)` and scaled by max exponent. |
