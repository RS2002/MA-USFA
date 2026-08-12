"""Gating rule for the toy: decide, per task, whether closed-form USFA
transfer is safe (FREE region -> use the dot product, no training) or a
learned composer must be invoked (PAID region -> train the upper-level MARL).

The gate is *theory-gated*: the paper's criterion is whether value-validity
holds, i.e. whether the transition kernel agent i experiences is (nearly)
independent of the task. The observable proxy we use is the *realized
interference rate* under the composed policy: the fraction of steps where
agent i's executed move differs from its intended move. If that rate is ~0
the dynamics factorize and closed-form composition is safe; if it is large,
the composition is operating in a coupled regime and the composer is invoked.

T2 then measures the gate against an ORACLE (route to paid iff the realized
value gap vs. per-task retraining exceeds a tolerance), reporting precision /
recall / F1 across a threshold sweep plus the compute saved vs. always
retraining.
"""

import numpy as np


def interference_rate(env, policy, w, kappa, n_steps=400):
    return env.measure_conflicts(policy, w, kappa, n_steps=n_steps)


def relative_gap(retrain_value, transfer_value):
    """Value gap of closed-form transfer relative to per-task retraining."""
    if retrain_value <= 0:
        return float("inf") if transfer_value < retrain_value else 0.0
    return float((retrain_value - transfer_value) / retrain_value)


def oracle_paid(rel_gap, tol=0.05):
    return bool(rel_gap > tol)


def gate_paid(rate, threshold):
    return bool(rate > threshold)


def evaluate_gate(tasks, threshold, tol=0.05):
    """tasks: list of dicts with keys rate, rel_gap.
    Returns (tp, fp, tn, fn, savings)."""
    tp = fp = tn = fn = 0
    for t in tasks:
        truth = oracle_paid(t["rel_gap"], tol)
        pred = gate_paid(t["rate"], threshold)
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec and not np.isnan(prec + rec) else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    # compute saved: fraction of tasks correctly routed to the cheap (free) route
    savings = tn / (tn + fp) if (tn + fp) else float("nan")
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec,
                f1=f1, accuracy=acc, savings=savings)


def threshold_sweep(tasks, grid=41, tol=0.05):
    """Sweep the interference threshold; return best-F1 threshold + curve."""
    rates = sorted({round(t["rate"], 4) for t in tasks})
    curve = []
    for thr in np.linspace(0.0, max(rates) * 1.05 + 1e-6, grid):
        m = evaluate_gate(tasks, float(thr), tol)
        curve.append((float(thr), m))
    best = max(curve, key=lambda c: (c[1]["f1"] if not np.isnan(c[1]["f1"]) else -1))
    return best, curve
