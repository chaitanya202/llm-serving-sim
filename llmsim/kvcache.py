"""Paged KV-cache block allocator (PagedAttention-style, with O(1) Prefix Caching & LRU eviction)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    prefix_key: Optional[tuple[int, int]] = None  # (prefix_id, block_index)
    last_accessed: float = 0.0


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16, enable_prefix_caching: bool = False):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching

        self.blocks = [Block(i) for i in range(num_blocks)]
        self.free_pool: list[int] = list(range(num_blocks))
        self.cache_table: dict[tuple[int, int], int] = {}  # (prefix_id, block_idx) -> block_id

        # O(1) LRU eviction queue: block_id -> None (oldest at front, newest at back)
        self.evictable_lru: OrderedDict[int, None] = OrderedDict()

        # Telemetry
        self.lookups: int = 0
        self.hits: int = 0
        self.tokens_saved: int = 0
        self.evictions: int = 0

    def blocks_for(self, tokens: int) -> int:
        return -(-tokens // self.block_size)  # ceil

    @property
    def free(self) -> int:
        return len(self.free_pool) + len(self.evictable_lru)

    def can_allocate(self, n: int) -> bool:
        return n <= self.free

    def match_prefix(self, prefix_id: int, num_prefix_tokens: int, current_time: float) -> tuple[list[int], int]:
        """
        Check how many consecutive blocks of prefix_id are cached.
        Returns (matched_block_ids, matched_tokens).
        """
        if not self.enable_prefix_caching or prefix_id is None or num_prefix_tokens < self.block_size:
            return [], 0

        self.lookups += 1
        num_prefix_blocks = num_prefix_tokens // self.block_size
        matched_blocks = []

        for blk_idx in range(num_prefix_blocks):
            key = (prefix_id, blk_idx)
            if key in self.cache_table:
                bid = self.cache_table[key]
                blk = self.blocks[bid]
                blk.ref_count += 1
                blk.last_accessed = current_time
                self.evictable_lru.pop(bid, None)  # active: remove from evictable LRU
                matched_blocks.append(bid)
            else:
                break

        matched_tokens = len(matched_blocks) * self.block_size
        if matched_blocks:
            self.hits += 1
            self.tokens_saved += matched_tokens

        return matched_blocks, matched_tokens

    def allocate_one_block(self, prefix_key: Optional[tuple[int, int]] = None, current_time: float = 0.0) -> int:
        """Allocates a single block in O(1), evicting the LRU unreferenced cached block if needed."""
        if self.free_pool:
            bid = self.free_pool.pop()
        elif self.enable_prefix_caching and self.evictable_lru:
            # O(1) eviction of the oldest unreferenced block
            bid, _ = self.evictable_lru.popitem(last=False)
            evicted_blk = self.blocks[bid]
            if evicted_blk.prefix_key in self.cache_table:
                del self.cache_table[evicted_blk.prefix_key]
            evicted_blk.prefix_key = None
            self.evictions += 1
        else:
            raise MemoryError("Out of KV memory: no free or evictable blocks")

        blk = self.blocks[bid]
        blk.ref_count = 1
        blk.last_accessed = current_time
        blk.prefix_key = prefix_key
        if prefix_key is not None and self.enable_prefix_caching:
            self.cache_table[prefix_key] = bid

        return bid

    def allocate(self, n: int) -> list[int]:
        """Allocate n anonymous blocks (backward-compatible)."""
        if n > self.free:
            raise MemoryError(f"need {n} blocks, only {self.free} free")
        return [self.allocate_one_block() for _ in range(n)]

    def release_block(self, bid: int, current_time: float = 0.0) -> None:
        blk = self.blocks[bid]
        assert blk.ref_count > 0, f"double free on block {bid}"
        blk.ref_count -= 1
        blk.last_accessed = current_time
        if blk.ref_count == 0:
            if not self.enable_prefix_caching or blk.prefix_key is None:
                blk.prefix_key = None
                self.free_pool.append(bid)
            else:
                # Retain in cache table, mark evictable in O(1) LRU queue
                self.evictable_lru.pop(bid, None)
                self.evictable_lru[bid] = None

    def release(self, n: int) -> None:
        """Release n active blocks (backward-compatible)."""
        count = 0
        for b in reversed(self.blocks):
            if b.ref_count > 0:
                self.release_block(b.block_id)
                count += 1
                if count == n:
                    break
        assert count == n, f"could not release {n} blocks (found {count})"

    @property
    def used(self) -> int:
        return sum(1 for b in self.blocks if b.ref_count > 0)

    @property
    def cached_blocks(self) -> int:
        return len(self.cache_table)

    @property
    def utilization(self) -> float:
        return self.used / self.num_blocks if self.num_blocks else 0.0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups > 0 else 0.0
