"""Discrete-event serving engine: static vs continuous batching, with chunked prefill & prefix caching.

Time advances step by step. Each step the scheduler picks work, the timing model
says how long it takes, then request state is updated. When idle, the clock jumps
to the next arrival.

continuous: vLLM-style. Every step = 1 decode token for all running sequences +
            prefill (chunked, under a token budget) for newly admitted ones.
            KV blocks are allocated on demand; on exhaustion the youngest
            running request is preempted (recompute).
            Supports optional prefix caching with LRU block eviction.
static    : classic batching. Take up to max_batch requests, prefill them all,
            then decode until the *longest* finishes. Finished sequences keep
            their slot (padding), and arrivals wait for the next batch.
            KV is reserved up front for prompt + max output.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Literal

from .kvcache import BlockAllocator
from .metrics import SLO, summarize
from .timing import TimingModel


@dataclass
class EngineConfig:
    mode: Literal["continuous", "static"] = "continuous"
    num_blocks: int = 28000
    block_size: int = 16
    max_num_seqs: int = 256            # continuous: max concurrent sequences
    max_num_batched_tokens: int = 2048  # continuous: per-step token budget (chunked prefill)
    max_batch: int = 32                # static: batch size
    max_queue: int | None = None       # admission control: shed arrivals when queue is this long
    enable_prefix_caching: bool = False # vLLM-style prefix caching with LRU eviction


class Engine:
    def __init__(self, timing: TimingModel, cfg: EngineConfig):
        self.timing, self.cfg = timing, cfg
        self.kv = BlockAllocator(cfg.num_blocks, cfg.block_size, enable_prefix_caching=cfg.enable_prefix_caching)
        self.step_log: list[tuple[float, float]] = []   # (dt, kv_util)
        self.t = 0.0

    # ------------------------------------------------------------------ public
    def run(self, requests, slo: SLO = SLO()) -> dict:
        reqs = sorted(requests, key=lambda r: r.arrival)
        if self.cfg.mode == "continuous":
            self._run_continuous(reqs)
        else:
            self._run_static(reqs)
        return summarize(reqs, slo, self.step_log, self.t, kv=self.kv)

    # ----------------------------------------------------------------- helpers
    def _admit(self, reqs, i, waiting):
        """Move arrivals with arrival <= now into the waiting queue (or shed)."""
        while i < len(reqs) and reqs[i].arrival <= self.t:
            r = reqs[i]; i += 1
            too_big = self.kv.blocks_for(r.prompt_len + r.output_len) > self.kv.num_blocks
            full = self.cfg.max_queue is not None and len(waiting) >= self.cfg.max_queue
            if too_big or full:
                r.dropped = True
            else:
                waiting.append(r)
        return i

    def _log(self, dt):
        self.step_log.append((dt, self.kv.utilization))

    def _allocate_for_request(self, r, need: int) -> list[int]:
        """Allocates 'need' blocks, tagging with prefix keys if within the prefix range."""
        cfg, kv = self.cfg, self.kv
        bids = []
        for _ in range(need):
            prefix_key = None
            if cfg.enable_prefix_caching and r.prefix_id is not None:
                curr_blk_idx = len(r.block_ids) + len(bids)
                if (curr_blk_idx + 1) * kv.block_size <= r.prefix_len:
                    prefix_key = (r.prefix_id, curr_blk_idx)
            bid = kv.allocate_one_block(prefix_key=prefix_key, current_time=self.t)
            bids.append(bid)
        return bids

    def _free(self, r):
        if r.block_ids:
            for bid in r.block_ids:
                self.kv.release_block(bid, current_time=self.t)
            r.block_ids.clear()
            r.blocks = 0
        elif r.blocks > 0:
            self.kv.release(r.blocks)
            r.blocks = 0

    def _preempt(self, r, waiting):
        """Recompute-style preemption: drop KV, redo prompt + generated tokens later."""
        self._free(r)
        # resident KV covered prompt + (generated - 1) tokens; recompute exactly that
        r.prefill_target = r.prompt_len + max(r.generated - 1, 0)
        r.computed = 0
        r.cached_tokens = 0
        r.preemptions += 1
        waiting.appendleft(r)

    # -------------------------------------------------------------- continuous
    def _run_continuous(self, reqs):
        cfg, kv = self.cfg, self.kv
        waiting: deque = deque()
        running: list = []
        i = 0
        while i < len(reqs) or waiting or running:
            i = self._admit(reqs, i, waiting)
            if not waiting and not running:
                self.t = reqs[i].arrival          # idle: jump to next arrival
                continue

            budget = cfg.max_num_batched_tokens
            decode, prefill = [], []              # prefill: (req, chunk)

            # 1) decode: every fully-prefilled running seq gets 1 token; ensure KV room
            for r in list(running):
                if r.computed < r.prefill_target:
                    continue
                need = kv.blocks_for(r.computed + 1) - r.blocks
                while need > kv.free and running:
                    victim = running[-1]           # preempt youngest
                    if victim is r and len(running) == 1:
                        break
                    running.remove(victim)
                    self._preempt(victim, waiting)
                    if victim in decode:
                        decode.remove(victim); budget += 1
                    if victim is r:
                        break
                if r not in running:
                    continue
                if need > kv.free:                 # alone and still can't grow: give up
                    running.remove(r); self._free(r); r.dropped = True
                    continue
                if need > 0:
                    new_bids = self._allocate_for_request(r, need)
                    r.block_ids.extend(new_bids)
                    r.blocks = len(r.block_ids)
                decode.append(r); budget -= 1

            # 2) continue chunked prefills already in progress
            for r in running:
                if r.computed < r.prefill_target and budget > 0:
                    chunk = min(r.prefill_target - r.computed, budget)
                    need = kv.blocks_for(r.computed + chunk) - r.blocks
                    if need <= kv.free:
                        if need > 0:
                            new_bids = self._allocate_for_request(r, need)
                            r.block_ids.extend(new_bids)
                            r.blocks = len(r.block_ids)
                        prefill.append((r, chunk)); budget -= chunk

            # 3) admit new requests (FIFO) while budget, slots and KV allow
            while waiting and budget > 0 and len(running) < cfg.max_num_seqs:
                r = waiting[0]
                # Match prefix cache on initial admission
                if cfg.enable_prefix_caching and r.prefix_id is not None and r.computed == 0 and r.prefix_len > 0:
                    matched_bids, matched_tokens = kv.match_prefix(r.prefix_id, r.prefix_len, self.t)
                    if matched_bids:
                        r.block_ids.extend(matched_bids)
                        r.blocks = len(r.block_ids)
                        r.computed = matched_tokens
                        r.cached_tokens = matched_tokens

                chunk = min(r.prefill_target - r.computed, budget)
                need = kv.blocks_for(r.computed + max(chunk, 1)) - r.blocks
                if need > kv.free:
                    break                          # head-of-line: wait for memory
                waiting.popleft()
                if need > 0:
                    new_bids = self._allocate_for_request(r, need)
                    r.block_ids.extend(new_bids)
                    r.blocks = len(r.block_ids)
                running.append(r)
                if chunk > 0:
                    prefill.append((r, chunk))
                    budget -= chunk
                else:
                    # 100% prefix cache hit: immediately scheduled into decode step
                    decode.append(r)
                    budget -= 1

            if not decode and not prefill:         # nothing schedulable (memory stall)
                if running:
                    self._preempt(running.pop(), waiting)
                    continue
                r = waiting.popleft(); r.dropped = True
                continue

            p_tokens = sum(c for _, c in prefill)
            kv_read = sum(r.computed for r in decode) + sum(r.computed for r, _ in prefill)
            dt = self.timing.step_time(p_tokens, len(decode), kv_read)
            self._log(dt)
            self.t += dt

            for r in decode:
                r.computed += 1; r.generated += 1
                if r.first_token is None:
                    r.first_token = self.t
                if r.generated >= r.output_len:
                    r.finish = self.t; running.remove(r); self._free(r)
            for r, chunk in prefill:
                r.computed += chunk
                if r.computed >= r.prefill_target and r.generated == 0:
                    r.generated = 1                # final prompt chunk emits the first token
                    r.first_token = self.t
                    if r.output_len == 1:
                        r.finish = self.t; running.remove(r); self._free(r)

    # ------------------------------------------------------------------ static
    def _run_static(self, reqs):
        cfg, kv = self.cfg, self.kv
        waiting: deque = deque()
        i = 0
        while i < len(reqs) or waiting:
            i = self._admit(reqs, i, waiting)
            if not waiting:
                self.t = reqs[i].arrival
                continue

            batch = []
            while waiting and len(batch) < cfg.max_batch:
                r = waiting[0]
                need = kv.blocks_for(r.prompt_len + r.output_len)   # reserve worst case
                if need > kv.free:
                    break
                waiting.popleft()
                new_bids = self._allocate_for_request(r, need)
                r.block_ids.extend(new_bids)
                r.blocks = len(r.block_ids)
                batch.append(r)
            if not batch:
                waiting.popleft().dropped = True
                continue

            # prefill all prompts in one pass
            p_tokens = sum(r.prompt_len for r in batch)
            dt = self.timing.step_time(p_tokens, 0, p_tokens)
            self._log(dt); self.t += dt
            for r in batch:
                r.computed = r.prompt_len; r.generated = 1; r.first_token = self.t
                if r.output_len == 1:
                    r.finish = self.t

            # decode until longest finishes; finished seqs still occupy slots (padding)
            for _ in range(max(r.output_len for r in batch) - 1):
                kv_read = sum(r.computed for r in batch)
                dt = self.timing.step_time(0, len(batch), kv_read)
                self._log(dt); self.t += dt
                for r in batch:
                    if r.generated < r.output_len:
                        r.computed += 1; r.generated += 1
                        if r.generated >= r.output_len:
                            r.finish = self.t
            for r in batch:
                self._free(r)
