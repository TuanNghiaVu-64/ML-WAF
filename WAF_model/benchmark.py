"""
benchmark.py
============
Compare ML-Driven E  vs  pure Random (RAN) baseline.

Runs both strategies against a real WAF (DVWA + ModSecurity CRS) for
a configurable time budget, then plots:

    X-axis : elapsed time (seconds)
    Y-axis : cumulative number of bypass payloads ("P")

Usage
-----
    # Default: 30-minute budget, variant E, tree classifier
    python benchmark.py

    # Custom params
    python benchmark.py --mu 100 --lam 500 --generations 20

    # Custom session / host
    python benchmark.py --phpsessid abc123 --host http://localhost:9003
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ── sibling imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from random_sampler import generate_unique_attacks
from waf_connector  import WafConnector, DvwaConfig
from ea_loop        import MLDrivenEA, EAConfig

# matplotlib — allow headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# Random baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_random_baseline(
    waf:        WafConnector,
    total:      int,
    batch_size: int = 50,
    verbose:    bool = True,
) -> list[dict]:
    """
    Generate `total` random attacks in batches, test each against the real
    WAF, and record (elapsed_total, cumulative_bypass) after every batch.

    Returns a list of dicts with:
        elapsed_total : float   seconds since start
        n_bypass      : int     cumulative bypass count
        n_total       : int     cumulative total tested
    """
    timeline: list[dict] = []
    cumulative_bypass = 0
    cumulative_total  = 0
    t0 = time.time()

    remaining = total
    batch_num = 0

    while remaining > 0:
        batch_num += 1
        n = min(batch_size, remaining)

        # Generate random attacks
        raw = generate_unique_attacks(n)
        corpus = [{"attack": a, "derivation": d} for a, d in raw]

        # Test against WAF
        for entry in corpus:
            entry["label"] = waf.check(entry["attack"])

        n_bypass  = sum(1 for e in corpus if e["label"] == "P")
        cumulative_bypass += n_bypass
        cumulative_total  += len(corpus)
        remaining -= len(corpus)

        elapsed = time.time() - t0
        timeline.append({
            "elapsed_total": elapsed,
            "n_bypass":      cumulative_bypass,
            "n_total":       cumulative_total,
        })

        if verbose:
            print(f"  [RAN] batch {batch_num:>3}  "
                  f"+{n_bypass} bypass  "
                  f"total={cumulative_bypass}  "
                  f"time={elapsed:.1f}s")

    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# EA strategy wrapper
# ─────────────────────────────────────────────────────────────────────────────

def run_ea_strategy(config: EAConfig) -> list[dict]:
    """
    Run ML-Driven EA and extract the time-series from history.

    Returns list of dicts with:
        elapsed_total : float
        n_bypass      : int
        n_total       : int   (bypass + blocked)
    """
    ea = MLDrivenEA(config)
    results = ea.run()

    timeline = []
    for h in results["history"]:
        timeline.append({
            "elapsed_total": h["elapsed_total"],
            "n_bypass":      h["n_bypass"],
            "n_total":       h["n_bypass"] + h["n_blocked"],
        })

    return timeline, results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(
    random_data: list[dict],
    ea_data:     list[dict],
    output:      str = "benchmark_result.png",
):
    """Plot both curves on a single graph and save to file."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # ── Random baseline curve ────────────────────────────────────────────
    rx = [d["elapsed_total"] for d in random_data]
    ry = [d["n_bypass"]      for d in random_data]
    ax.plot(rx, ry, color="#3b82f6", linewidth=2, marker=".", markersize=4,
            label="Random (RAN)", alpha=0.85)

    # ── ML-Driven E curve ────────────────────────────────────────────────
    ex = [d["elapsed_total"] for d in ea_data]
    ey = [d["n_bypass"]      for d in ea_data]
    ax.plot(ex, ey, color="#f97316", linewidth=2.5, marker="o", markersize=5,
            label="ML-Driven E", alpha=0.9)

    # ── Labels and styling ───────────────────────────────────────────────
    ax.set_xlabel("Elapsed Time (seconds)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Bypass Payloads", fontsize=13, fontweight="bold")
    ax.set_title("ML-Driven E  vs  Random Baseline — WAF Bypass Performance",
                 fontsize=15, fontweight="bold", pad=12)
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Add final-count annotations
    if ry:
        ax.annotate(f"{ry[-1]} bypasses",
                    xy=(rx[-1], ry[-1]), fontsize=10, color="#3b82f6",
                    textcoords="offset points", xytext=(8, -15),
                    fontweight="bold")
    if ey:
        ax.annotate(f"{ey[-1]} bypasses",
                    xy=(ex[-1], ey[-1]), fontsize=10, color="#f97316",
                    textcoords="offset points", xytext=(8, 10),
                    fontweight="bold")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"\n[GRAPH] Saved to {output}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def estimate_generations(time_budget_min: float, mu: int, lam: int) -> int:
    """
    Rough estimate of how many generations fit in the time budget.

    Assumption: each WAF request ≈ 0.1s (network + WAF processing).
    Per generation the EA sends ~lam requests, init sends ~init_size.
    """
    secs = time_budget_min * 60
    est_per_gen = lam * 0.1         # seconds per generation (WAF calls)
    est_init    = mu  * 0.1         # seconds for init phase
    available   = secs - est_init
    if available <= 0:
        return 2
    return max(2, int(available / est_per_gen))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ML-Driven E vs Random baseline against a real WAF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── EA parameters ─────────────────────────────────────────────────────
    parser.add_argument("--mu",          type=int,   default=50,
                        help="Population size (µ)")
    parser.add_argument("--lam",         type=int,   default=200,
                        help="Offspring per generation (λ)")
    parser.add_argument("--generations", type=int,   default=None,
                        help="Number of generations (auto-estimated from "
                             "--time if not set)")
    parser.add_argument("--init-size",   type=int,   default=100,
                        help="Initial random population size")
    parser.add_argument("--classifier",  type=str,   default="tree",
                        choices=["tree", "forest"],
                        help="Classifier type")
    parser.add_argument("--n-trees",     type=int,   default=10,
                        help="Trees in RandomForest (if classifier=forest)")

    # ── Time budget ───────────────────────────────────────────────────────
    parser.add_argument("--time",        type=float, default=30.0,
                        help="Time budget in MINUTES (used to auto-estimate "
                             "generations and random budget)")

    # ── Random baseline ───────────────────────────────────────────────────
    parser.add_argument("--random-budget",     type=int, default=None,
                        help="Total random attacks to generate (default: "
                             "init_size + generations * lam)")
    parser.add_argument("--random-batch-size", type=int, default=50,
                        help="Random attacks per batch")

    # ── WAF connection ────────────────────────────────────────────────────
    parser.add_argument("--phpsessid",   type=str,
                        default="f59690908df4860b89bddd3eaba6922c",
                        help="DVWA PHPSESSID cookie")
    parser.add_argument("--host",        type=str,
                        default="http://localhost:9003",
                        help="DVWA base URL")

    # ── Output ────────────────────────────────────────────────────────────
    parser.add_argument("--output",      type=str,
                        default="benchmark_result.png",
                        help="Output graph filename")

    args = parser.parse_args()

    # ── Auto-estimate generations from time budget ────────────────────────
    if args.generations is None:
        args.generations = estimate_generations(args.time, args.mu, args.lam)
        print(f"[AUTO] Estimated {args.generations} generations "
              f"for {args.time:.0f}-minute budget")

    # ── Random budget: same total WAF calls as the EA ─────────────────────
    if args.random_budget is None:
        args.random_budget = args.init_size + args.generations * args.lam
        print(f"[AUTO] Random budget = {args.random_budget} attacks "
              f"(matching EA total)")

    # ── WAF connector (shared by both strategies) ─────────────────────────
    waf = WafConnector(DvwaConfig(
        host      = args.host,
        phpsessid = args.phpsessid,
    ))

    # ── Print run configuration ───────────────────────────────────────────
    print()
    print("=" * 65)
    print("  BENCHMARK: ML-Driven E  vs  Random (RAN)")
    print("=" * 65)
    print(f"  WAF target     : {args.host}")
    print(f"  Time budget    : {args.time:.0f} minutes")
    print(f"  EA params      : µ={args.mu}  λ={args.lam}  "
          f"gen={args.generations}  init={args.init_size}")
    print(f"  Classifier     : {args.classifier}"
          f"{'  n_trees=' + str(args.n_trees) if args.classifier == 'forest' else ''}")
    print(f"  Random budget  : {args.random_budget} attacks  "
          f"(batch={args.random_batch_size})")
    print(f"  Output graph   : {args.output}")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════════════════
    # Run 1: Random baseline
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"▸ Phase 1: Random Baseline ({args.random_budget} attacks)")
    print(f"{'─'*65}")

    random_data = run_random_baseline(
        waf        = waf,
        total      = args.random_budget,
        batch_size = args.random_batch_size,
        verbose    = True,
    )

    print(f"\n  Random done: {random_data[-1]['n_bypass']} bypasses "
          f"in {random_data[-1]['elapsed_total']:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Run 2: ML-Driven E
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print(f"▸ Phase 2: ML-Driven E  "
          f"(µ={args.mu}, λ={args.lam}, gen={args.generations})")
    print(f"{'─'*65}")

    ea_config = EAConfig(
        mu          = args.mu,
        lam         = args.lam,
        max_gen     = args.generations,
        init_size   = args.init_size,
        variant     = "E",
        classifier  = args.classifier,
        n_trees     = args.n_trees,
        use_mock    = False,
        phpsessid   = args.phpsessid,
        host        = args.host,
        verbose     = True,
    )

    ea_data, ea_results = run_ea_strategy(ea_config)

    print(f"\n  EA done: {ea_data[-1]['n_bypass']} bypasses "
          f"in {ea_data[-1]['elapsed_total']:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # Plot comparison
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("▸ Plotting comparison graph...")
    print(f"{'─'*65}")

    plot_comparison(random_data, ea_data, output=args.output)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print("  SUMMARY")
    print(f"{'═'*65}")
    print(f"  Random : {random_data[-1]['n_bypass']:>5} bypasses  "
          f"/ {random_data[-1]['n_total']:>5} tested  "
          f"({random_data[-1]['elapsed_total']:.1f}s)")
    print(f"  ML-E   : {ea_data[-1]['n_bypass']:>5} bypasses  "
          f"/ {ea_data[-1]['n_total']:>5} tested  "
          f"({ea_data[-1]['elapsed_total']:.1f}s)")

    r_rate = random_data[-1]["n_bypass"] / max(1, random_data[-1]["n_total"])
    e_rate = ea_data[-1]["n_bypass"]     / max(1, ea_data[-1]["n_total"])
    print(f"\n  Bypass rate: Random={r_rate:.1%}  ML-E={e_rate:.1%}")

    if e_rate > r_rate and r_rate > 0:
        print(f"  ML-E advantage: {e_rate/r_rate:.1f}x higher bypass rate")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
