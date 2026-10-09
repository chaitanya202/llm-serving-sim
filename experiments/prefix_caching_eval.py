"""Evaluate Prefix Caching with Block Hashing and LRU Eviction.

Compares serving performance with and without prefix caching across
varying shared prefix ratios (0%, 25%, 50%, 75%, 100%) for multi-turn/RAG workloads.

Run:
    python experiments/prefix_caching_eval.py
"""
import csv
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llmsim import (
    Engine, EngineConfig, RooflineTiming, LLAMA3_8B, H100_SXM,
    WorkloadSpec, generate, SLO, kv_capacity_blocks
)

def run_experiment():
    timing = RooflineTiming(LLAMA3_8B, H100_SXM)
    NB = kv_capacity_blocks(LLAMA3_8B, H100_SXM)
    slo = SLO(ttft_s=0.5, tpot_s=0.05)

    prefix_ratios = [0.0, 0.25, 0.50, 0.75, 1.0]
    shared_prefix_len = 384  # 384 tokens shared system prompt/context
    qps = 25

    rows = []
    print("=" * 80)
    print("PREFIX CACHING EVALUATION (SHARED SYSTEM PROMPT / MULTI-TURN RAG)")
    print("=" * 80)

    for ratio in prefix_ratios:
        spec = WorkloadSpec(
            qps=qps,
            duration_s=25,
            prompt_mean=512,
            output_mean=150,
            shared_prefix_len=shared_prefix_len,
            shared_prefix_ratio=ratio,
            seed=42
        )
        reqs_no_cache = generate(spec)
        reqs_with_cache = generate(spec)

        # 1. Baseline: Without Prefix Caching
        t0 = time.time()
        cfg_no = EngineConfig(mode="continuous", num_blocks=NB, enable_prefix_caching=False)
        m_no = Engine(timing, cfg_no).run(reqs_no_cache, slo)
        t_no = time.time() - t0

        # 2. With Prefix Caching (Block Hashing + LRU)
        t1 = time.time()
        cfg_yes = EngineConfig(mode="continuous", num_blocks=NB, enable_prefix_caching=True)
        m_yes = Engine(timing, cfg_yes).run(reqs_with_cache, slo)
        t_yes = time.time() - t1

        rows.append({
            "prefix_ratio": ratio,
            "policy": "without_prefix_caching",
            **m_no
        })
        rows.append({
            "prefix_ratio": ratio,
            "policy": "with_prefix_caching",
            **m_yes
        })

        hit_rate = m_yes.get("prefix_hit_rate", 0.0) * 100
        saved = m_yes.get("tokens_saved", 0)
        ttft_drop = (1.0 - (m_yes["ttft_p99"] / m_no["ttft_p99"])) * 100 if m_no["ttft_p99"] > 0 else 0.0

        print(
            f"Ratio={int(ratio*100):3d}% | "
            f"Hit Rate={hit_rate:5.1f}% | "
            f"Tokens Saved={saved:7d} | "
            f"NoCache TTFT p99={m_no['ttft_p99']*1e3:6.1f}ms | "
            f"Cache TTFT p99={m_yes['ttft_p99']*1e3:6.1f}ms (-{ttft_drop:4.1f}%) | "
            f"Goodput={m_yes['goodput_req_s']:5.2f} req/s",
            flush=True
        )

    # Save CSV
    csv_path = "results/prefix_caching_eval.csv"
    all_keys = list({k: None for r in rows for k in r.keys()}.keys())
    for r in rows:
        for k in all_keys:
            r.setdefault(k, 0)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results to {csv_path}")

    # Generate visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ratios_pct = [int(r * 100) for r in prefix_ratios]
    no_cache = [r for r in rows if r["policy"] == "without_prefix_caching"]
    with_cache = [r for r in rows if r["policy"] == "with_prefix_caching"]

    # 1. P99 TTFT vs Shared Prefix Ratio
    ax = axes[0]
    ax.plot(ratios_pct, [r["ttft_p99"] * 1000 for r in no_cache], marker="o", color="#d62728", linewidth=2, label="Without Prefix Caching")
    ax.plot(ratios_pct, [r["ttft_p99"] * 1000 for r in with_cache], marker="s", color="#2ca02c", linewidth=2, label="With Prefix Caching (vLLM-style)")
    ax.set_title("P99 TTFT vs. Shared Prefix Ratio (Lower is Better)")
    ax.set_xlabel("% of Requests Sharing System Prompt (384 tok)")
    ax.set_ylabel("P99 TTFT (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Token Hit Rate & Tokens Saved
    ax = axes[1]
    tokens_saved_k = [r.get("tokens_saved", 0) / 1000 for r in with_cache]
    hit_rates = [r.get("prefix_hit_rate", 0.0) * 100 for r in with_cache]
    color = "#1f77b4"
    ax.set_xlabel("% of Requests Sharing System Prompt")
    ax.set_ylabel("Prefix Cache Hit Rate (%)", color=color)
    line1 = ax.plot(ratios_pct, hit_rates, marker="^", color=color, linewidth=2, label="Hit Rate (%)")
    ax.tick_params(axis="y", labelcolor=color)

    ax2 = ax.twinx()
    color2 = "#ff7f0e"
    ax2.set_ylabel("Total Prefill Tokens Saved (Thousands)", color=color2)
    line2 = ax2.plot(ratios_pct, tokens_saved_k, marker="D", color=color2, linewidth=2, linestyle="--", label="Tokens Saved (k)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax.set_title("Cache Hit Rate & Compute Savings")
    ax.grid(True, alpha=0.3)

    # 3. Goodput under SLO
    ax = axes[2]
    ax.plot(ratios_pct, [r["goodput_req_s"] for r in no_cache], marker="o", color="#d62728", linewidth=2, label="Without Prefix Caching")
    ax.plot(ratios_pct, [r["goodput_req_s"] for r in with_cache], marker="s", color="#2ca02c", linewidth=2, label="With Prefix Caching")
    ax.set_title("Goodput Under SLO (TTFT < 500ms & TPOT < 50ms)")
    ax.set_xlabel("% of Requests Sharing System Prompt")
    ax.set_ylabel("Goodput (req/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Prefix Caching Performance: Llama-3-8B on 1x H100 SXM (QPS=25, System Prompt=384 tokens)", fontsize=13, y=1.03)
    fig.tight_layout()
    chart_path = "results/prefix_caching_eval.png"
    fig.savefig(chart_path, dpi=140, bbox_inches="tight")
    print(f"Saved chart to {chart_path}")

if __name__ == "__main__":
    run_experiment()
