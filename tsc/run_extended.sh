#!/usr/bin/env bash
# =============================================================================
#  TSC SF+GPI EXTENDED experiments (Linux / macOS)
#
#  Four robustness studies beyond run_all.sh:
#    hetero    : retraining-from-scratch on the w-heterogeneous target task
#    rollouts  : multi-rollout variance (10 rollouts, every rule, no retraining)
#    budget    : MA-USFA fine-tuning-budget sweep 100/200/500 episodes (homo)
#    ksweep    : Synchronized source-library-size sweep K=1..5 (homo)
#
#  Usage:
#    ./run_extended.sh            # all four
#    ./run_extended.sh hetero     # one study only
#    ./run_extended.sh rollouts
#    ./run_extended.sh budget
#    ./run_extended.sh ksweep
#
#  Prerequisite: run_all.sh must have produced the shared successor-feature
#  checkpoint for seed 0 first (or use the shipped checkpoints/sf_psi.pth).
# =============================================================================
set -euo pipefail

WHICH="${1:-all}"
PY="${PYTHON:-python}"
EVAL=5
EVAL_MULTI=10
LOGDIR=logs
mkdir -p "$LOGDIR"

CK="checkpoints/sf_psi.pth"
[[ -f "$CK" ]] || CK="sf_s0/usfa/sf_psi.pth"
[[ -f "$CK" ]] || { echo "ERROR: no successor-feature checkpoint. Run run_all.sh first."; exit 1; }

do_step () { echo "$1"; }

# ---- hetero retraining baseline ----
if [[ "$WHICH" == "all" || "$WHICH" == "hetero" ]]; then
  echo "== EXP hetero: retraining baseline (seed 0) =="
  mkdir -p hetero_retrain
  "$PY" train.py --net gat --reward composite --seed 0 > "$LOGDIR/retrain_hetero.log" 2>&1
  [[ -f composite_s0/DQN_gat/pretrain_eval.txt ]] && \
    cp -f composite_s0/DQN_gat/pretrain_eval.txt hetero_retrain/retrain_hetero_eval.txt
fi

# ---- multi-rollout variance (episodes 0 => load composer, eval only) ----
if [[ "$WHICH" == "all" || "$WHICH" == "rollouts" ]]; then
  echo "== EXP rollouts: 10-rollout variance (seed 0) =="
  mkdir -p rollout_variance
  for spec in \
      "independent  0 transfer  transfer_homo" \
      "independent  1 transfer  transfer_hetero" \
      "synchronized 0 l1        l1_homo" \
      "synchronized 1 l1        l1_hetero" \
      "ma-usfa      0 composer  composer_homo" \
      "ma-usfa      1 composer  composer_hetero"; do
    set -- $spec; phase="$1"; het="$2"; ph="$3"; tag="$4"
    extra=""; [[ "$phase" == "ma-usfa" ]] && extra="--episodes 0"
    "$PY" train_sf.py --phase "$phase" --sf_ckpt "$CK" --seed 0 --hetero "$het" \
        --task_seed 11 --eval_episodes "$EVAL_MULTI" $extra \
        > "$LOGDIR/rollout_${tag}.log" 2>&1
    cp -f "sf_s0/${ph}/${tag}_result.json" "rollout_variance/${tag}_result.json" 2>/dev/null || true
  done
fi

# ---- MA-USFA fine-tuning budget sweep ----
if [[ "$WHICH" == "all" || "$WHICH" == "budget" ]]; then
  echo "== EXP budget: MA-USFA episode-budget sweep (homo, seed 0) =="
  mkdir -p budget_sweep
  for E in 100 200 500; do
    "$PY" train_sf.py --phase ma-usfa --sf_ckpt "$CK" --seed 0 --task_seed 11 \
        --episodes "$E" --eval_episodes "$EVAL" \
        > "$LOGDIR/composer_homo_ep${E}.log" 2>&1
    cp -f sf_s0/composer/composer_homo_result.json \
        "budget_sweep/composer_homo_ep${E}_result.json" 2>/dev/null || true
  done
fi

# ---- Synchronized library-size K sweep ----
if [[ "$WHICH" == "all" || "$WHICH" == "ksweep" ]]; then
  echo "== EXP ksweep: Synchronized library-size sweep K=1..5 (homo, seed 0) =="
  mkdir -p k_sweep
  # Independent reference (library-content-independent)
  "$PY" train_sf.py --phase independent --sf_ckpt "$CK" --seed 0 --task_seed 11 \
      --eval_episodes "$EVAL" > "$LOGDIR/ksweep_transfer.log" 2>&1
  cp -f sf_s0/transfer/transfer_homo_result.json k_sweep/l2_homo_result.json 2>/dev/null || true
  for K in 1 2 3 4 5; do
    "$PY" train_sf.py --phase synchronized --sf_ckpt "$CK" --seed 0 --task_seed 11 \
        --lib_size "$K" --eval_episodes "$EVAL" > "$LOGDIR/ksweep_l1_k${K}.log" 2>&1
    cp -f sf_s0/l1/l1_homo_result.json "k_sweep/l1_homo_k${K}_result.json" 2>/dev/null || true
  done
fi

echo
echo "== extended experiments complete: hetero_retrain/ rollout_variance/ budget_sweep/ k_sweep/ =="
