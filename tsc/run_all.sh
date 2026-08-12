#!/usr/bin/env bash
# =============================================================================
#  TSC SF+GPI experiment runner (Linux / macOS)
#
#  The full four-method matrix on both target-task families:
#    retraining / Independent / Synchronized / MA-USFA
#  on a w-homogeneous and a w-heterogeneous target task.
#
#  The three composition rules are the paper's names; internally the entry
#  point maps them to --phase transfer / l1 / composer (both spellings work).
#
#  Usage:   ./run_all.sh                 # defaults: SEED=0 TASK_SEED=11
#           ./run_all.sh 0 11            # SEED=0, TASK_SEED=11
#
#  Prerequisite: the SUMO Manhattan 28x7 network files under ./data/ (see
#  data/README.md) and a Python env with the packages in ../requirements.txt.
#
#  A pre-trained shared successor-feature checkpoint ships in
#  ./checkpoints/sf_psi.pth; delete it (or the sf_s0/ dir) to re-train from
#  scratch. Per-method results land in ./sf_s<SEED>/<phase>/ and are collected
#  into ./results_s<SEED>/. Everything is logged to ./logs/.
# =============================================================================
set -euo pipefail

SEED="${1:-0}"
TASK_SEED="${2:-11}"

USFA_EPISODES=500
COMPOSER_EPISODES=200
EVAL_EPISODES=5

PY="${PYTHON:-python}"
LOGDIR=logs
mkdir -p "$LOGDIR"

# Prefer the shipped checkpoint; fall back to the freshly trained one.
CK="checkpoints/sf_psi.pth"
if [[ ! -f "$CK" ]]; then
  CK="sf_s${SEED}/usfa/sf_psi.pth"
fi

echo "============================================================"
echo " SEED=$SEED  TASK_SEED=$TASK_SEED"
echo " usfa_ep=$USFA_EPISODES  composer_ep=$COMPOSER_EPISODES  eval=$EVAL_EPISODES"
echo " checkpoint = $CK"
echo "============================================================"

# ---- 0. shared w-free successor features (prerequisite for all rules) ----
if [[ -f "$CK" ]]; then
  echo "[0/7] successor-feature checkpoint present, skipping training: $CK"
else
  echo "[0/7] training shared w-free successor features ..."
  "$PY" train_sf.py --phase usfa --seed "$SEED" --episodes "$USFA_EPISODES" \
      > "$LOGDIR/usfa.log" 2>&1
  CK="sf_s${SEED}/usfa/sf_psi.pth"
fi
[[ -f "$CK" ]] || { echo "ERROR: checkpoint not found after usfa: $CK"; exit 1; }

run () {  # run <step> <label> <logfile> <extra-args...>
  local step="$1" label="$2" log="$3"; shift 3
  echo "[$step] $label ..."
  "$PY" "$@" > "$LOGDIR/$log" 2>&1
}

# ================= w-HOMOGENEOUS target task =================
run 1/7 "Independent  (homogeneous)"  transfer_homo.log \
    train_sf.py --phase independent  --sf_ckpt "$CK" --seed "$SEED" --task_seed "$TASK_SEED" --eval_episodes "$EVAL_EPISODES"
run 2/7 "Synchronized (homogeneous)"  l1_homo.log \
    train_sf.py --phase synchronized --sf_ckpt "$CK" --seed "$SEED" --task_seed "$TASK_SEED" --eval_episodes "$EVAL_EPISODES"
run 3/7 "MA-USFA      (homogeneous)"  composer_homo.log \
    train_sf.py --phase ma-usfa      --sf_ckpt "$CK" --seed "$SEED" --task_seed "$TASK_SEED" --episodes "$COMPOSER_EPISODES" --eval_episodes "$EVAL_EPISODES"
run 4/7 "retraining   (homogeneous)"  retrain.log \
    train.py --net gat --reward composite --seed "$SEED"

# ================= w-HETEROGENEOUS target task =================
run 5/7 "Independent  (heterogeneous)" transfer_hetero.log \
    train_sf.py --phase independent  --sf_ckpt "$CK" --seed "$SEED" --hetero 1 --task_seed "$TASK_SEED" --eval_episodes "$EVAL_EPISODES"
run 6/7 "Synchronized (heterogeneous)" l1_hetero.log \
    train_sf.py --phase synchronized --sf_ckpt "$CK" --seed "$SEED" --hetero 1 --task_seed "$TASK_SEED" --eval_episodes "$EVAL_EPISODES"
run 7/7 "MA-USFA      (heterogeneous)" composer_hetero.log \
    train_sf.py --phase ma-usfa      --sf_ckpt "$CK" --seed "$SEED" --hetero 1 --task_seed "$TASK_SEED" --episodes "$COMPOSER_EPISODES" --eval_episodes "$EVAL_EPISODES"

# ---- collect every result json into one folder ----
OUT="results_s${SEED}"
mkdir -p "$OUT"
for f in transfer_homo l1_homo composer_homo transfer_hetero l1_hetero composer_hetero; do
  ph="${f%%_*}"
  cp -f "sf_s${SEED}/${ph}/${f}_result.json" "$OUT/" 2>/dev/null || true
done

echo
echo "============================================================"
echo " DONE. Per-method results collected in: $OUT/"
echo " (retraining output is under composite_s${SEED}/)"
echo "============================================================"
