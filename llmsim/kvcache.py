"""Paged KV-cache block allocator (PagedAttention-style, accounting only)."""
from __future__ import annotations


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free = num_blocks

    def blocks_for(self, tokens: int) -> int:
        return -(-tokens // self.block_size)  # ceil

    def can_allocate(self, n: int) -> bool:
        return n <= self.free

    def allocate(self, n: int) -> None:
        if n > self.free:
            raise MemoryError(f"need {n} blocks, only {self.free} free")
        self.free -= n

    def release(self, n: int) -> None:
        self.free += n
        assert self.free <= self.num_blocks, "double free"

    @property
    def used(self) -> int:
        return self.num_blocks - self.free

    @property
    def utilization(self) -> float:
        return self.used / self.num_blocks if self.num_blocks else 0.0
