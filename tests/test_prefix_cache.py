"""Tests for Prefix Caching with Block Hashing and LRU Eviction."""
import pytest
from llmsim import (
    Engine, EngineConfig, RooflineTiming, LLAMA3_8B, H100_SXM,
    WorkloadSpec, generate, SLO, kv_capacity_blocks
)
from llmsim.kvcache import BlockAllocator

T = RooflineTiming(LLAMA3_8B, H100_SXM)
NB = kv_capacity_blocks(LLAMA3_8B, H100_SXM)


def test_prefix_cache_matching_and_conservation():
    a = BlockAllocator(num_blocks=32, block_size=16, enable_prefix_caching=True)
    
    # Prefix 0 has 256 tokens = 16 blocks
    # 1. Initially nothing cached
    matched_bids, matched_tokens = a.match_prefix(prefix_id=0, num_prefix_tokens=256, current_time=0.0)
    assert matched_bids == []
    assert matched_tokens == 0
    assert a.lookups == 1 and a.hits == 0

    # 2. Allocate 16 blocks for prefix 0
    bids = [a.allocate_one_block(prefix_key=(0, i), current_time=1.0) for i in range(16)]
    assert len(bids) == 16
    assert a.cached_blocks == 16
    assert a.used == 16

    # 3. Release blocks (active request finishes)
    for bid in bids:
        a.release_block(bid, current_time=2.0)
    # Blocks remain in cache with ref_count = 0, so used = 0 but cached_blocks = 16
    assert a.used == 0
    assert a.cached_blocks == 16

    # 4. Next request arrives with the same prefix 0: 100% Cache Hit!
    matched_bids_2, matched_tokens_2 = a.match_prefix(prefix_id=0, num_prefix_tokens=256, current_time=3.0)
    assert matched_bids_2 == bids
    assert matched_tokens_2 == 256
    assert a.hits == 1
    assert a.hit_rate == 0.5  # 1 hit out of 2 lookups
    assert a.used == 16       # Referenced by the new request


def test_prefix_cache_lru_eviction():
    # Allocator with only 4 blocks
    a = BlockAllocator(num_blocks=4, block_size=16, enable_prefix_caching=True)
    
    # Allocate 4 blocks for prefix A at t=1.0
    bids_a = [a.allocate_one_block(prefix_key=(10, i), current_time=1.0) for i in range(4)]
    for b in bids_a:
        a.release_block(b, current_time=1.5)
    
    assert a.cached_blocks == 4
    assert a.free == 4  # Evictable

    # Now allocate 2 blocks for prefix B at t=5.0
    # Must evict the 2 oldest blocks from prefix A
    bids_b = [a.allocate_one_block(prefix_key=(20, i), current_time=5.0) for i in range(2)]
    assert a.evictions == 2
    assert a.cached_blocks == 4  # 2 from A remaining, 2 from B


def test_prefix_caching_slashes_ttft_under_load():
    spec = WorkloadSpec(
        qps=15,
        duration_s=8,
        prompt_mean=512,
        output_mean=100,
        shared_prefix_len=384,
        shared_prefix_ratio=0.8,
        seed=42
    )
    reqs_no_cache = generate(spec)
    reqs_with_cache = generate(spec)

    cfg_no_cache = EngineConfig(mode="continuous", num_blocks=NB, enable_prefix_caching=False)
    cfg_with_cache = EngineConfig(mode="continuous", num_blocks=NB, enable_prefix_caching=True)

    m_no_cache = Engine(T, cfg_no_cache).run(reqs_no_cache)
    m_with_cache = Engine(T, cfg_with_cache).run(reqs_with_cache)

    assert m_with_cache["prefix_hits"] > 0
    assert m_with_cache["tokens_saved"] > 0
    assert m_with_cache["ttft_p50"] < m_no_cache["ttft_p50"]
