"""Run the toy experiment: T1 (free-region correctness), T2 (gating
decision), T3 (failure-region necessity + composer recovery).

Usage:
    python run_toy.py                 # full run
    python run_toy.py --quick         # reduced budget (sanity check)

Outputs (in ./results/):
    results.json     -- all numbers (the paper tables are built from this)
    tables.txt       -- human-readable tables
    figs/*.png       -- transfer-vs-coupling curve, composer curves, gating
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sfworld import SFWorld
from sf_learn import (train_sf, train_sf_usfa, train_q, make_l1_policy,
                     make_l2_policy, make_q_policy)
from composers import Composer
from gate import interference_rate, relative_gap, threshold_sweep

GRID, N_AGENTS, K = 5, 3, 4
HORIZON, GAMMA = 40, 0.95
RESOURCE_SCALE = 0.6
COLLISION_PENALTY = 2.0
USFA_EPS, RETRAIN_EPS = 12000, 4000
USFA_EPS_DECAY = 4000
COMP_EPS = 1400
EVAL_EP = 80

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def norm_value(vals, ref):
    return [float(v / ref) if ref > 0 else float("nan") for v in vals]


def main():
    global USFA_EPS, USFA_EPS_DECAY, RETRAIN_EPS, COMP_EPS, EVAL_EP
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        # reduced budget: pipeline sanity check, numbers are noisier
        USFA_EPS, USFA_EPS_DECAY = 4000, 1500
        RETRAIN_EPS, COMP_EPS, EVAL_EP = 800, 500, 30
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    R = {"config": dict(grid=GRID, agents=N_AGENTS, features=K, horizon=HORIZON,
                        gamma=GAMMA, usfa_episodes=USFA_EPS,
                        retrain_episodes=RETRAIN_EPS, quick=args.quick)}

    env = SFWorld(grid=GRID, n_agents=N_AGENTS, n_features=K,
                  resource_scale=RESOURCE_SCALE, horizon=HORIZON, seed=0,
                  collision_penalty=COLLISION_PENALTY)
    trng = np.random.RandomState(1234)
    t0 = time.time()

    # ================================================================= #
    # Stage 1 -- USFA w-free features (ONE training run, task-agnostic)
    # ================================================================= #
    # sampler box matches the task-weight box [0.1, 1.5]: SF transfer is exact
    # only where psi is accurate, so the w-free features are trained over the
    # same range the tasks live in (weights outside the training support make
    # the tabular psi extrapolate -- a measurement artifact, not a real
    # transfer failure).
    def sampler(rng):
        return rng.uniform(0.1, 1.5, size=(N_AGENTS, K))

    print("[stage 1] training w-free USFA features ...")
    psi_usfa = train_sf_usfa(env, sampler, episodes=USFA_EPS,
                             eps_decay=USFA_EPS_DECAY, gamma=GAMMA, seed=0)
    R["stage1"] = {"usfa_train_seconds": round(time.time() - t0, 1)}
    print(f"  done in {R['stage1']['usfa_train_seconds']}s")

    n_free = 3 if args.quick else 8
    n_paid = 2 if args.quick else 5
    free_tasks = [env.task_free(rng=trng) for _ in range(n_free)]
    paid_tasks = [env.task_paid(rng=trng) for _ in range(n_paid)]
    kappas = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.25, 0.5, 0.75, 1.0]
    comp_kappas = [1.0] if args.quick else [0.5, 1.0]

    # ================================================================= #
    # T1 -- FREE-region correctness (kappa = 0)
    #   Both aligned AND heterogeneous tasks: whenever the dynamics
    #   factorize, closed-form transfer is exact regardless of reward
    #   coupling ("goals may couple, dynamics must decouple").
    # ================================================================= #
    print("[T1] free-region correctness ...")
    t1_rows = []
    for wi, w in enumerate(free_tasks + paid_tasks):
        vi, _ = env.vi_policy_value(w, 0.0, n_episodes=EVAL_EP, gamma=GAMMA)
        v_l1 = env.evaluate(make_l1_policy(psi_usfa), w, 0.0,
                            n_episodes=EVAL_EP, gamma=GAMMA)[0]
        v_l2 = env.evaluate(make_l2_policy(psi_usfa), w, 0.0,
                            n_episodes=EVAL_EP, gamma=GAMMA)[0]
        psi_rt = train_sf(env, w, kappa=0.0, episodes=RETRAIN_EPS, gamma=GAMMA,
                          seed=100 + wi)
        v_rt = env.evaluate(make_l2_policy(psi_rt), w, 0.0,
                            n_episodes=EVAL_EP, gamma=GAMMA)[0]
        t1_rows.append(dict(task=wi, family="aligned" if wi < len(free_tasks)
                            else "heterogeneous",
                            joint_opt=vi, l1=v_l1, l2=v_l2, retrain=v_rt))
    t1 = {
        "mean": {k: float(np.mean([r[k] for r in t1_rows]))
                 for k in ("joint_opt", "l1", "l2", "retrain")},
        "l2_vs_jointopt_gap_pct": float(np.mean(
            [(r["joint_opt"] - r["l2"]) / r["joint_opt"] * 100 for r in t1_rows])),
        "l2_vs_retrain_gap_pct": float(np.mean(
            [(r["retrain"] - r["l2"]) / r["retrain"] * 100 for r in t1_rows])),
        "ordering_violations": int(np.sum([1 for r in t1_rows
                                          if r["l1"] > r["l2"] + 1e-6
                                          or r["l2"] > r["joint_opt"] + 1e-6])),
        "rows": t1_rows,
    }
    R["T1"] = t1

    # ================================================================= #
    # T3 -- paid region: transfer degrades in kappa, composer recovers
    # ================================================================= #
    print("[T3] transfer-vs-coupling sweep ...")
    sweep = {}          # (task, kappa) -> per-task retrain + transfer values
    for wi, w in enumerate(paid_tasks):
        for kappa in kappas:
            key = f"t{wi}k{kappa}"
            t0k = time.time()
            Q_rt = train_q(env, w, kappa=kappa, episodes=RETRAIN_EPS,
                           gamma=GAMMA, seed=200 + wi)
            v_rt = env.evaluate(make_q_policy(Q_rt), w, kappa,
                                n_episodes=EVAL_EP, gamma=GAMMA)[0]
            v_tr = env.evaluate(make_l2_policy(psi_usfa), w, kappa,
                                n_episodes=EVAL_EP, gamma=GAMMA)[0]
            rate = interference_rate(env, make_l2_policy(psi_usfa), w, kappa)
            sweep[key] = dict(task=wi, kappa=kappa, retrain=v_rt, transfer=v_tr,
                              rel_gap=relative_gap(v_rt, v_tr), rate=rate)
            print(f"  {key}: rt={v_rt:.1f} tr={v_tr:.1f} gap={v_rt - v_tr:+.1f} "
                  f"rate={rate:.3f} ({time.time()-t0k:.1f}s)")

    # composer recovery (paid region, strongest couplings)
    print("[T3] composer training (qmix / attn / comm) ...")
    comp_rows = []
    for wi, w in enumerate(paid_tasks):
        for kappa in comp_kappas:
            key = f"t{wi}k{kappa}"
            ref = sweep[key]["retrain"]
            for fl in Composer.FLAVORS:
                t0c = time.time()
                c = Composer(env, psi_usfa, w, flavor=fl, gamma=GAMMA, seed=0)
                _, evals, td = c.train_task(w, kappa, episodes=COMP_EPS,
                                            eval_every=200, seed=0)
                v_c = env.evaluate(c.policy(), w, kappa, n_episodes=EVAL_EP,
                                   gamma=GAMMA)[0]
                comp_rows.append(dict(task=wi, kappa=kappa, flavor=fl,
                                      value=v_c, norm=norm_value([v_c], ref)[0],
                                      curve=evals, td_trace=td[:50],
                                      seconds=round(time.time() - t0c, 1)))
                print(f"  {key} {fl}: v={v_c:.1f} ({v_c/ref*100:.0f}% of retrain) "
                      f"{time.time()-t0c:.0f}s")
    R["T3"] = {"sweep": sweep, "composers": comp_rows}

    # ================================================================= #
    # T2 -- gating: interference rate vs. oracle routing
    # ================================================================= #
    print("[T2] gating ...")
    gate_tasks = []
    for r in t1_rows:
        gate_tasks.append(dict(rate=0.0, rel_gap=relative_gap(r["retrain"], r["l2"])))
    for key, s in sweep.items():
        gate_tasks.append(dict(rate=s["rate"], rel_gap=s["rel_gap"]))
    best, curve = threshold_sweep(gate_tasks, tol=0.05)
    R["T2"] = {"best": {"threshold": best[0], **best[1]},
               "n_tasks": len(gate_tasks),
               "curve": [(c[0], c[1]) for c in curve]}
    print(f"  best-F1 threshold={best[0]:.3f} -> {best[1]}")

    # ================================================================= #
    # outputs
    # ================================================================= #
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    write_tables(R)
    make_figs(env, R)
    print("[done] results in", OUT)


# --------------------------------------------------------------------- #
def write_tables(R):
    lines = []
    t1 = R["T1"]
    lines.append("=" * 74)
    lines.append("T1  FREE-region correctness (kappa=0: aligned + heterogeneous tasks)")
    lines.append("=" * 74)
    lines.append(f"{'task':>4} {'family':>12} {'joint-opt':>10} {'USFA-L1':>10} "
                 f"{'USFA-L2':>10} {'retrain':>10}")
    for r in t1["rows"]:
        lines.append(f"{r['task']:>4} {r['family']:>12} {r['joint_opt']:>10.1f} "
                     f"{r['l1']:>10.1f} {r['l2']:>10.1f} {r['retrain']:>10.1f}")
    lines.append(f"mean L2-vs-jointopt gap: {t1['l2_vs_jointopt_gap_pct']:.2f}%   "
                 f"L2-vs-retrain gap: {t1['l2_vs_retrain_gap_pct']:.2f}%   "
                 f"ordering violations: {t1['ordering_violations']}")

    lines.append("")
    lines.append("=" * 74)
    lines.append("T3  Paid region: closed-form transfer vs per-task retraining vs composers")
    lines.append("=" * 74)
    lines.append(f"{'task':>4} {'kap':>4} {'retrain':>9} {'transfer':>9} {'relgap':>7} {'rate':>6}")
    for key, s in sorted(R["T3"]["sweep"].items()):
        lines.append(f"{s['task']:>4} {s['kappa']:>4.2f} {s['retrain']:>9.1f} "
                     f"{s['transfer']:>9.1f} {s['rel_gap']:>7.3f} {s['rate']:>6.3f}")
    lines.append("composers (value as % of per-task retrain):")
    agg = {}
    for c in R["T3"]["composers"]:
        agg.setdefault(c["kappa"], {}).setdefault(c["flavor"], []).append(c["norm"])
    for kappa in sorted(agg):
        for fl in Composer.FLAVORS:
            if fl in agg[kappa]:
                vals = agg[kappa][fl]
                lines.append(f"  kappa={kappa}: {fl:>5} {np.mean(vals)*100:6.1f}% "
                             f"+- {np.std(vals)*100:5.1f}%")

    lines.append("")
    lines.append("=" * 74)
    lines.append("T2  Gating (threshold sweep, oracle = rel_gap > 5%)")
    lines.append("=" * 74)
    b = R["T2"]["best"]
    lines.append(f"best threshold={b['threshold']:.3f} precision={b['precision']:.2f} "
                 f"recall={b['recall']:.2f} F1={b['f1']:.2f} accuracy={b['accuracy']:.2f} "
                 f"savings={b['savings']:.2f}  (tp={b['tp']} fp={b['fp']} "
                 f"tn={b['tn']} fn={b['fn']})")
    with open(os.path.join(OUT, "tables.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# --------------------------------------------------------------------- #
def make_figs(env, R):
    sweep = R["T3"]["sweep"]
    kappas = sorted({s["kappa"] for s in sweep.values()})
    paid = sorted({s["task"] for s in sweep.values()})

    # fig 1: transfer vs retrain as a function of coupling    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    rt_mean = [np.mean([sweep[f"t{t}k{k}"]["retrain"] for t in paid]) for k in kappas]
    tr_mean = [np.mean([sweep[f"t{t}k{k}"]["transfer"] for t in paid]) for k in kappas]
    ax.plot(kappas, rt_mean, "o-", color="#2c7fb8", label="per-task retraining (paid)")
    ax.plot(kappas, tr_mean, "s--", color="#d95f02", label="closed-form USFA transfer")
    for k in sorted({c["kappa"] for c in R["T3"]["composers"]}):
        for fl in Composer.FLAVORS:
            vals = [c["value"] for c in R["T3"]["composers"]
                    if c["kappa"] == k and c["flavor"] == fl]
            ax.plot([k], [np.mean(vals)], "D", markersize=8,
                    label=f"composer-{fl} (kappa={k})")
    ax.set_xlabel("transition coupling  kappa")
    ax.set_ylabel("mean discounted joint return")
    ax.set_title("Closed-form transfer degrades with coupling;\ncomposers recover")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig1_transfer_curve.png"), dpi=150)

    # fig 2: composer training curves on a representative paid task
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    w = 0
    k = 1.0
    ref = sweep[f"t{w}k{k}"]["retrain"]
    for fl in Composer.FLAVORS:
        c = [x for x in R["T3"]["composers"]
             if x["task"] == w and x["kappa"] == k and x["flavor"] == fl][0]
        ep = np.arange(200, 200 * (len(c["curve"]) + 1), 200)
        ax.plot(ep, c["curve"], "-o", markersize=3, label=f"{fl}")
    ax.axhline(ref, color="#2c7fb8", ls="--", label="per-task retraining")
    ax.axhline(sweep[f"t{w}k{k}"]["transfer"], color="#d95f02", ls=":",
               label="closed-form transfer")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("joint return")
    ax.set_title(f"Composer fine-tuning on a paid task (task {w}, kappa={k}); "
                 "transfer-initialized")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig2_composer_curves.png"), dpi=150)

    # fig 3: gating scatter + precision/recall
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tasks = R["T2"]
    ax = axes[0]
    # rebuild task list from results (cheap)
    free_pts = [(0.0, c["rel_gap"]) for c in R["T1"]["rows"]]
    paid_pts = [(s["rate"], s["rel_gap"]) for s in R["T3"]["sweep"].values()]
    for rate, gap in free_pts:
        ax.plot(rate, gap, "o", color="#2c7fb8")
    for rate, gap in paid_pts:
        ax.plot(rate, gap, "o", color="#d95f02")
    b = tasks["best"]
    ax.axvline(b["threshold"], color="k", ls="--", lw=1)
    ax.set_xlabel("measured interference rate (gate signal)")
    ax.set_ylabel("value gap vs per-task retraining (oracle)")
    ax.set_title("gate signal vs oracle routing")
    ax.grid(alpha=0.3)
    curve = tasks["curve"]
    ax = axes[1]
    thr = [c[0] for c in curve]
    ax.plot(thr, [c[1]["precision"] for c in curve], "-", label="precision")
    ax.plot(thr, [c[1]["recall"] for c in curve], "-", label="recall")
    ax.plot(thr, [c[1]["f1"] for c in curve], "-", label="F1")
    ax.set_xlabel("interference threshold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figs", "fig3_gating.png"), dpi=150)
    plt.close("all")


if __name__ == "__main__":
    main()
