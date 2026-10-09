"""Sweep max_num_batched_tokens (chunked prefill budget) to analyze the TTFT vs TPOT trade-off.

Large chunk budget -> fast prompt prefill (lower TTFT), but heavy compute steps inflate decode step time (higher TPOT).
Small chunk budget -> smooth, low-latency decode steps (lower TPOT), but prompts are split across multiple cycles (higher TTFT).

Run:
    python experiments/chunk_budget_sweep.py
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
    slo = SLO(ttft_s=0.5, tpot_s=0.05)  # TTFT < 500ms, TPOT < 50ms

    chunk_budgets = [256, 512, 1024, 2048, 4096, 8192]
    qps_levels = [15, 30]  # 15 QPS (moderate load), 30 QPS (heavy load near saturation)

    rows = []
    print("=" * 80)
    print("CHUNKED PREFILL BUDGET SWEEP (TTFT vs TPOT TRADE-OFF)")
    print("=" * 80)

    for q in qps_levels:
        for budget in chunk_budgets:
            t0 = time.time()
            cfg = EngineConfig(
                mode="continuous",
                num_blocks=NB,
                max_num_batched_tokens=budget,
                max_num_seqs=256
            )
            # Consistent seed so arrival sequences and lengths are identical across budgets
            reqs = generate(WorkloadSpec(qps=q, duration_s=30, seed=42))
            eng = Engine(timing, cfg)
            m = eng.run(reqs, slo)
            
            # Step timing stats
            dts = [s[0] * 1000 for s in eng.step_log] if eng.step_log else [0]
            step_mean_ms = float(np.mean(dts))
            step_p99_ms = float(np.percentile(dts, 99))

            row = {
                "qps": q,
                "chunk_budget": budget,
                "step_mean_ms": step_mean_ms,
                "step_p99_ms": step_p99_ms,
                **m
            }
            rows.append(row)

            print(
                f"QPS={q:2d} | Budget={budget:4d} | "
                f"TTFT p99={m['ttft_p99']*1e3:6.1f}ms | "
                f"TPOT p99={m['tpot_p99']*1e3:5.1f}ms | "
                f"Step p99={step_p99_ms:5.1f}ms | "
                f"Goodput={m['goodput_req_s']:5.2f} req/s | "
                f"Elapsed={time.time()-t0:.1f}s",
                flush=True
            )

    # Save CSV
    csv_path = "results/chunk_budget_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results to {csv_path}")

    # Generate 4-panel visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {15: "#1f77b4", 30: "#ff7f0e"}
    markers = {15: "o", 30: "s"}

    # 1. P99 TTFT vs Chunk Budget
    ax = axes[0, 0]
    for q in qps_levels:
        rs = [r for r in rows if r["qps"] == q]
        ax.plot(
            [r["chunk_budget"] for r in rs],
            [r["ttft_p99"] * 1000 for r in rs],
            marker=markers[q], color=colors[q], linewidth=2, label=f"QPS = {q}"
        )
    ax.axhline(slo.ttft_s * 1000, color="red", linestyle="--", alpha=0.8, label="TTFT SLO (500 ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunk_budgets)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("P99 TTFT vs. Chunk Budget (Lower is Better)")
    ax.set_xlabel("Max Batched Tokens (Chunk Budget)")
    ax.set_ylabel("P99 TTFT (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. P99 TPOT vs Chunk Budget
    ax = axes[0, 1]
    for q in qps_levels:
        rs = [r for r in rows if r["qps"] == q]
        ax.plot(
            [r["chunk_budget"] for r in rs],
            [r["tpot_p99"] * 1000 for r in rs],
            marker=markers[q], color=colors[q], linewidth=2, label=f"QPS = {q}"
        )
    ax.axhline(slo.tpot_s * 1000, color="red", linestyle="--", alpha=0.8, label="TPOT SLO (50 ms)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunk_budgets)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("P99 TPOT vs. Chunk Budget (Lower is Better)")
    ax.set_xlabel("Max Batched Tokens (Chunk Budget)")
    ax.set_ylabel("P99 TPOT (ms/token)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. P99 Step Time vs Chunk Budget
    ax = axes[1, 0]
    for q in qps_levels:
        rs = [r for r in rows if r["qps"] == q]
        ax.plot(
            [r["chunk_budget"] for r in rs],
            [r["step_p99_ms"] for r in rs],
            marker=markers[q], color=colors[q], linewidth=2, label=f"QPS = {q}"
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunk_budgets)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("P99 Step Forward-Pass Latency vs. Chunk Budget")
    ax.set_xlabel("Max Batched Tokens (Chunk Budget)")
    ax.set_ylabel("P99 Step Time (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Goodput (Slo Attainment) vs Chunk Budget
    ax = axes[1, 1]
    for q in qps_levels:
        rs = [r for r in rows if r["qps"] == q]
        ax.plot(
            [r["chunk_budget"] for r in rs],
            [r["goodput_req_s"] for r in rs],
            marker=markers[q], color=colors[q], linewidth=2, label=f"QPS = {q}"
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(chunk_budgets)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("Goodput Under SLO (TTFT < 500ms & TPOT < 50ms)")
    ax.set_xlabel("Max Batched Tokens (Chunk Budget)")
    ax.set_ylabel("Goodput (req/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        "Chunked Prefill Trade-Off Analysis: Llama-3-8B on 1x H100 SXM\n"
        "(Prompt ~512, Output ~200, Roofline Timing Model)",
        fontsize=13, y=0.99
    )
    fig.tight_layout()
    chart_path = "results/chunk_budget_sweep.png"
    fig.savefig(chart_path, dpi=140, bbox_inches="tight")
    print(f"Saved chart to {chart_path}")

if __name__ == "__main__":
    run_experiment()
