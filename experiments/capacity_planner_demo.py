"""Capacity Planner: Telecom Erlang-C vs. vLLM Continuous Batching.

Demonstrates cluster capacity sizing for LLM serving:
1. Calculates traffic intensity in Erlangs (A = lambda * E[S]).
2. Sizing under naive classical Erlang-C (single-stream server assumption).
3. Sizing under slotted Erlang-C (static concurrency slots).
4. Sizing under continuous batching simulation (memory bandwidth roofline + PagedAttention).
5. Quantifies where Erlang-C diverges from real GPU serving.

Run:
    python experiments/capacity_planner_demo.py
"""
import csv
import math
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llmsim import (
    LLAMA3_8B, H100_SXM, CapacityPlanner, SLO
)


def run_demo():
    print("=" * 85)
    print("CAPACITY PLANNER: TELECOM ERLANG-C VS. CONTINUOUS BATCHING SIMULATION")
    print("=" * 85)

    target_qps = 100.0
    prompt_len = 512
    output_len = 200
    slo = SLO(ttft_s=0.5, tpot_s=0.05)  # 500ms TTFT, 50ms TPOT
    duration_s = 15.0
    candidate_sizes = [2, 3, 4, 5, 6, 8, 10]

    planner = CapacityPlanner(LLAMA3_8B, H100_SXM)
    t_prefill, t_decode, t_total = planner.single_request_service_time(prompt_len, output_len)

    print(f"Workload: {target_qps:.0f} QPS | Prompt: {prompt_len} tok | Output: {output_len} tok")
    print(f"Isolated Service Time: E[S_prefill]={t_prefill*1000:.1f}ms | E[S_decode]={t_decode*1000:.1f}ms | E[S_total]={t_total:.3f}s")
    print(f"SLO Targets: TTFT P99 <= {slo.ttft_s*1000:.0f}ms | TPOT P99 <= {slo.tpot_s*1000:.0f}ms")
    print("-" * 85)

    report = planner.plan(
        qps=target_qps,
        prompt_len=prompt_len,
        output_len=output_len,
        slo=slo,
        duration_s=duration_s,
        candidate_sizes=candidate_sizes
    )

    print(f"Offered Load: {report.traffic_erlangs:.2f} Erlangs")
    print(f"Naive Erlang-C Sizing (1 request/GPU): {report.erlang_min_gpus} GPUs")
    print(f"Continuous Batching Sizing (vLLM):   {report.simulated_min_gpus} GPUs")
    print(f"Continuous Batching Consolidation:   {report.batching_efficiency:.1f}x hardware reduction")
    print("-" * 85)
    print(f"{'GPUs':>4} | {'QPS/GPU':>7} | {'Slotted M/M/c':>14} | {'Sim TTFT P99':>12} | {'Sim TPOT P99':>12} | {'Goodput':>9} | {'KV Util':>7} | {'Status'}")
    print("-" * 85)

    for r in report.cluster_results:
        slotted_str = f"{r['slotted_erlang_p99_ms']:8.1f} ms" if r['slotted_erlang_p99_ms'] < 10000 else "   Overloaded"
        status_str = "PASS (SLO Met)" if r["meets_slo"] else "FAIL (SLO Exceeded)"
        print(
            f"{r['num_gpus']:4d} | "
            f"{r['per_gpu_qps']:7.1f} | "
            f"{slotted_str:>14} | "
            f"{r['sim_p99_ttft_ms']:10.1f} ms | "
            f"{r['sim_p99_tpot_ms']:10.2f} ms | "
            f"{r['total_goodput']:7.1f}/s | "
            f"{r['kv_util_max']*100:6.1f}% | "
            f"{status_str}"
        )
    print("=" * 85)

    # Save CSV
    csv_path = "results/capacity_planner.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "num_gpus", "per_gpu_qps", "naive_erlang_p99_ms", "slotted_erlang_p99_ms",
            "sim_p99_ttft_ms", "sim_p99_tpot_ms", "goodput_per_gpu", "total_goodput",
            "kv_util_max", "meets_slo"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report.cluster_results)
    print(f"Saved numerical results to {csv_path}")

    # Generate 3-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))

    gpus = [r["num_gpus"] for r in report.cluster_results]
    sim_ttft = [r["sim_p99_ttft_ms"] for r in report.cluster_results]
    slotted_erlang = [min(r["slotted_erlang_p99_ms"], 2000.0) for r in report.cluster_results]
    sim_tpot = [r["sim_p99_tpot_ms"] for r in report.cluster_results]
    kv_util = [r["kv_util_max"] * 100 for r in report.cluster_results]

    # Panel 1: TTFT P99 vs Cluster Size (Continuous Batching vs Slotted Erlang-C)
    ax1 = axes[0]
    ax1.plot(gpus, sim_ttft, marker="o", color="#2ca02c", linewidth=2.5, label="Continuous Batching (Simulated)")
    ax1.plot(gpus, slotted_erlang, marker="s", color="#1f77b4", linewidth=2.0, linestyle="--", label="Slotted Erlang-C (32 slots/GPU)")
    ax1.axhline(slo.ttft_s * 1000, color="#d62728", linestyle=":", linewidth=2.0, label=f"TTFT SLO ({slo.ttft_s*1000:.0f} ms)")
    ax1.axvline(report.simulated_min_gpus, color="#2ca02c", linestyle="--", alpha=0.5, label=f"Sim Min ({report.simulated_min_gpus} GPUs)")
    ax1.set_title("P99 TTFT vs. Cluster Size (100 QPS Total)")
    ax1.set_xlabel("Number of H100 GPUs in Cluster")
    ax1.set_ylabel("P99 TTFT (ms)")
    ax1.set_ylim(0, 1000)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Memory Bandwidth Inflation (TPOT & KV Utilization)
    ax2 = axes[1]
    color_tpot = "#d62728"
    ax2.set_xlabel("Number of H100 GPUs in Cluster")
    ax2.set_ylabel("P99 TPOT (ms/token)", color=color_tpot)
    line1 = ax2.plot(gpus, sim_tpot, marker="^", color=color_tpot, linewidth=2.5, label="P99 TPOT (ms)")
    ax2.axhline(slo.tpot_s * 1000, color=color_tpot, linestyle=":", alpha=0.7, label=f"TPOT SLO ({slo.tpot_s*1000:.0f} ms)")
    ax2.tick_params(axis="y", labelcolor=color_tpot)
    ax2.set_ylim(0, 60)

    ax2_right = ax2.twinx()
    color_kv = "#ff7f0e"
    ax2_right.set_ylabel("Peak KV Cache Utilization (%)", color=color_kv)
    line2 = ax2_right.plot(gpus, kv_util, marker="D", color=color_kv, linewidth=2.0, linestyle="--", label="Peak KV Util (%)")
    ax2_right.tick_params(axis="y", labelcolor=color_kv)
    ax2_right.set_ylim(0, 100)
    ax2.set_title("Decode TPOT Inflation vs. Batch Load")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Sizing Model Comparison (Hardware Footprint)
    ax3 = axes[2]
    # Sizing comparison bar chart
    slotted_erlang_min = 5  # At 5 GPUs (160 slots > 149.3 Erlangs)
    models = ["Naive Erlang-C\n(Unbatched)", "Slotted Erlang-C\n(32 slots/GPU)", "Continuous Batching\n(PagedAttention)"]
    counts = [report.erlang_min_gpus, slotted_erlang_min, report.simulated_min_gpus]
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c"]

    bars = ax3.bar(models, counts, color=colors, width=0.55, edgecolor="black", linewidth=1)
    ax3.set_title(f"Cluster Sizing Recommendation (100 QPS)")
    ax3.set_ylabel("Minimum GPUs Required")
    ax3.set_ylim(0, max(counts) * 1.15)
    ax3.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax3.annotate(
            f"{count} GPUs",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontweight="bold", fontsize=10
        )

    ax3.text(
        0.5, 0.65,
        f"Consolidation Ratio:\n{report.batching_efficiency:.1f}x hardware reduction\n({report.erlang_min_gpus} -> {report.simulated_min_gpus} GPUs)",
        transform=ax3.transAxes,
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f5e9", edgecolor="#2ca02c", alpha=0.9),
        fontsize=9.5, fontweight="bold", color="#1b5e20"
    )

    fig.suptitle("LLM Serving Capacity Planning: Erlang-C ($M/M/c$) vs. Continuous Batching (Llama-3-8B on H100)", fontsize=13, y=1.03)
    fig.tight_layout()
    chart_path = "results/capacity_planner.png"
    fig.savefig(chart_path, dpi=140, bbox_inches="tight")
    print(f"Generated capacity planner chart: {chart_path}")


if __name__ == "__main__":
    run_demo()
