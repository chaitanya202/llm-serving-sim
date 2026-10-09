"""Static vs continuous batching: latency / goodput vs offered load.
Run:  python experiments/latency_vs_load.py
"""
import csv, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from llmsim import (Engine, EngineConfig, RooflineTiming, LLAMA3_8B, H100_SXM,
                    WorkloadSpec, generate, SLO, kv_capacity_blocks)

timing = RooflineTiming(LLAMA3_8B, H100_SXM)
NB = kv_capacity_blocks(LLAMA3_8B, H100_SXM)
SLO_ = SLO(ttft_s=0.5, tpot_s=0.05)
QPS = [2, 5, 10, 15, 20, 30, 40, 60, 80, 100]
MODES = {"static (batch=32)": EngineConfig(mode="static", num_blocks=NB, max_batch=32),
         "continuous + chunked prefill": EngineConfig(mode="continuous", num_blocks=NB)}

rows = []
for name, cfg in MODES.items():
    for q in QPS:
        t0 = time.time()
        reqs = generate(WorkloadSpec(qps=q, duration_s=30, seed=1))
        m = Engine(timing, cfg).run(reqs, SLO_)
        rows.append({"policy": name, "qps": q, **m})
        print(f"{name:30s} qps={q:4d} ttft_p99={m['ttft_p99']*1e3:9.1f}ms "
              f"tpot_p99={m['tpot_p99']*1e3:6.1f}ms goodput={m['goodput_req_s']:6.2f}/s "
              f"kv_max={m['kv_util_max']:.2f}  ({time.time()-t0:.1f}s wall)", flush=True)

with open("results/latency_vs_load.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for name in MODES:
    rs = [r for r in rows if r["policy"] == name]
    x = [r["qps"] for r in rs]
    ax[0].plot(x, [r["ttft_p99"] * 1e3 for r in rs], marker="o", label=name)
    ax[1].plot(x, [r["goodput_req_s"] for r in rs], marker="o", label=name)
    ax[2].plot(x, [r["kv_util_mean"] * 100 for r in rs], marker="o", label=name)
ax[0].axhline(SLO_.ttft_s * 1e3, ls="--", c="gray"); ax[0].set_yscale("log")
ax[0].set(title="p99 TTFT vs load (dashed = 500 ms SLO)", xlabel="offered QPS", ylabel="ms")
ax[1].plot(QPS, QPS, ls=":", c="gray", label="ideal"); ax[1].legend()
ax[1].set(title="Goodput (req/s meeting TTFT<500ms & TPOT<50ms)", xlabel="offered QPS", ylabel="req/s")
ax[2].set(title="Mean KV-cache utilization", xlabel="offered QPS", ylabel="%")
for a in ax: a.grid(alpha=.3)
fig.suptitle("Simulated Llama-3-8B on 1x H100 (roofline timing, uncalibrated) | prompt~512, output~200", y=1.02)
fig.tight_layout(); fig.savefig("results/latency_vs_load.png", dpi=140, bbox_inches="tight")
