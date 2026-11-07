"""Reproduce the README results (3 experiments, 5 replications each).

    python run.py           # ~30 s
    python run.py --quick   # CI smoke run
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from invsim import Config, Policy, chen_bullwhip_lower_bound, newsvendor_z, replicate  # noqa: E402


def bullwhip_experiment(base: Config, windows, reps):
    rows = []
    for p in windows:
        for pos in (False, True):
            pol = Policy(forecast="moving_average", window=p)
            cfg = replace(base, store=pol, dc=replace(pol, z=2.0), dc_uses_pos=pos)
            r = replicate(cfg, reps)
            rows.append({"window": p, "dc_uses_pos": pos,
                         **{k: round(v[0], 3) for k, v in r.items()}})
    return rows


def safety_experiment(base: Config, zs, reps):
    rows = []
    for z in zs:
        r = replicate(replace(base, store=Policy(z=z)), reps)
        rows.append({"z": z, **{k: round(v[0], 3) for k, v in r.items()}})
    return rows


def policy_experiment(base: Config, K: float, reps):
    pols = {
        "base-stock, daily review": Policy(order_cost=K),
        "base-stock, weekly review": Policy(review_days=7, order_cost=K),
        "(s,S) with EOQ, daily review": Policy(kind="sS", order_cost=K),
    }
    rows = []
    for name, pol in pols.items():
        r = replicate(replace(base, store=pol), reps)
        rows.append({"policy": name, **{k: round(v[0], 2) for k, v in r.items()}})
    return rows


def plot_bullwhip(rows, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = sorted({r["window"] for r in rows})
    get = lambda pos, k: [next(r[k] for r in rows if r["window"] == x and r["dc_uses_pos"] == pos) for x in w]
    ax.semilogy(w, get(False, "bullwhip_store_orders"), "o-", c="#1f4e9c", label="store orders to DC")
    ax.semilogy(w, get(False, "bullwhip_dc_orders"), "s-", c="#d1495b", label="DC orders to plant")
    ax.semilogy(w, get(True, "bullwhip_dc_orders"), "s--", c="#2a9d55",
                label="DC orders to plant, with POS data sharing")
    ax.semilogy(w, [chen_bullwhip_lower_bound(2, x) for x in w], ":", c="#555",
                label="Chen et al. (2000), textbook single stage (L=2, backorders)")
    ax.axhline(1, c="#999", lw=0.8)
    ax.set_xlabel("moving-average forecast window p (days)")
    ax.set_ylabel("Var(orders) / Var(customer demand)")
    ax.set_title("Bullwhip effect: order variance amplification up the supply chain")
    ax.set_xticks(w)
    ax.set_xticklabels(w)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_safety(rows, z_nv, path):
    z = [r["z"] for r in rows]
    cost = [r["total_cost_per_day"] for r in rows]
    fill = [r["fill_rate_pct"] for r in rows]
    best = int(np.argmin(cost))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(z, cost, "o-", c="#1f4e9c", label="total cost / day")
    ax.axvline(z[best], c="#1f4e9c", ls=":", lw=1)
    ax.axvline(z_nv, c="#d1495b", ls="--", lw=1, label=f"newsvendor critical-ratio z = {z_nv:.2f}")
    ax.annotate(f"simulated optimum z = {z[best]}\nINR {cost[best]:.0f}/day", (z[best], cost[best]),
                textcoords="offset points", xytext=(15, 35), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("store safety factor z")
    ax.set_ylabel("network cost (INR / day)")
    ax2 = ax.twinx()
    ax2.plot(z, fill, "^--", c="#2a9d55", label="fill rate %")
    ax2.set_ylabel("fill rate (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")
    ax.set_title("Choosing store safety stock: cost vs service")
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_policies(rows, path):
    parts = ["holding_store_per_day", "holding_dc_per_day", "shortage_per_day", "ordering_per_day"]
    labels = ["store holding", "DC holding", "lost sales", "ordering (trucks)"]
    cols = ["#1f4e9c", "#7fa7d9", "#d1495b", "#e9a03b"]
    names = [r["policy"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    left = np.zeros(len(rows))
    for p, lab, c in zip(parts, labels, cols):
        v = np.array([r[p] for r in rows])
        ax.barh(names, v, left=left, color=c, label=lab)
        left += v
    for i, r in enumerate(rows):
        ax.text(left[i] + 20, i, f"INR {r['total_cost_per_day']:.0f}/day | fill {r['fill_rate_pct']:.1f}%",
                va="center", fontsize=8)
    ax.set_xlim(0, left.max() * 1.45)
    ax.invert_yaxis()
    ax.set_xlabel("cost (INR / day)")
    ax.set_title("Store replenishment policy with INR 150 per delivery")
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    base = Config(days=200 if a.quick else 730)
    reps = 1 if a.quick else 5
    windows = [3, 14] if a.quick else [3, 5, 7, 14, 28]
    zs = [0.5, 1.5] if a.quick else [round(x, 2) for x in np.arange(0.0, 3.01, 0.25)]

    bw = bullwhip_experiment(base, windows, reps)
    sf = safety_experiment(base, zs, reps)
    z_nv = newsvendor_z(base.h_store, base.shortage)
    pol = policy_experiment(base, 150.0, reps)

    (out / "metrics.json").write_text(json.dumps(
        {"bullwhip": bw, "safety_stock": sf, "newsvendor_z": round(z_nv, 3), "policies": pol}, indent=2))
    plot_bullwhip(bw, out / "bullwhip.png")
    plot_safety(sf, z_nv, out / "safety_stock.png")
    plot_policies(pol, out / "policies.png")

    print("Bullwhip (window, POS sharing, store ratio, DC ratio, cost/day, fill%)")
    for r in bw:
        print(f"  p={r['window']:<3} pos={r['dc_uses_pos']!s:5}  {r['bullwhip_store_orders']:7.2f} "
              f"{r['bullwhip_dc_orders']:8.2f}  {r['total_cost_per_day']:7.1f}  {r['fill_rate_pct']:.2f}")
    print(f"\nSafety stock (newsvendor z = {z_nv:.2f})")
    for r in sf:
        print(f"  z={r['z']:<5} cost {r['total_cost_per_day']:7.1f}  fill {r['fill_rate_pct']:.2f}")
    print("\nPolicies with K = 150")
    for r in pol:
        print(f"  {r['policy']:32s} cost {r['total_cost_per_day']:7.1f}  fill {r['fill_rate_pct']:.2f}  "
              f"ordering {r['ordering_per_day']:.1f}  store inv {r['avg_store_inventory']}")


if __name__ == "__main__":
    main()
