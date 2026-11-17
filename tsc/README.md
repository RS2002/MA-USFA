# Traffic-signal-control experiment — Manhattan 28×7

> **Current implementation (2026-11-17):** Use [CURRENT_RUN_GUIDE.md](CURRENT_RUN_GUIDE.md) and `python run.py --help`. The text below describes the historical release. The current pipeline trains scene-bound frozen sources and a shared attention-DDQN selector; it never loads `checkpoints/sf_psi.pth` or fits on the evaluation targets.

The large-scale instantiation of the paper's claims: 196 traffic-signal agents
on a Manhattan 28×7 grid, built on top of an IDDQN reference. Where the toy uses
a synthetic coupling dial, this network has **intrinsic** coupling — 196
intersections share real road cells — so it tests whether the safety story
survives at realistic scale and under a strict locality constraint (a node may
only talk to its one-hop road neighbours).

## The successor-feature hook

The environment's composite reward is already a linear feature model, which is
exactly what makes successor-feature transfer applicable:

```
r = φ · w,   φ = [queue, wait, pressure, speed, fairness]_norm   (5-dim)
             w = [-1.0, -0.3, -1.5, +3.0, 0.0]   (DEFAULT_W, signed importances)
```

The five normalized congestion terms **are** the successor features `φ`, and
the signed term weights **are** the task vector `w`.

## Prerequisites

```bash
pip install -r ../requirements.txt        # torch, sumo-rl, tqdm, numpy
```

You also need a working SUMO install and the Manhattan 28×7 network files placed
under `data/` — see `data/README.md`. The training-set step (pre-training the
task-free successor features) is expensive; a pre-trained checkpoint ships in
`checkpoints/sf_psi.pth` and the runner scripts pick it up automatically, so you
can go straight to the test-set composition rules.

## Step 0 — training set: pre-train the shared successor features

This is the "training-set training" step. A fresh task weight `w` is drawn every
episode and pushed onto the reward, so `ψ` learns the feature-based value
without committing to any single task. Every composition rule below reuses this
one checkpoint.

```bash
python train_sf.py --phase usfa --seed 0 --episodes 500
# -> writes sf_s0/usfa/sf_psi.pth   (already provided in checkpoints/)
```

## Steps 1–3 — test set: the three composition rules

Each rule is evaluated on a target task without touching the pre-trained `ψ`
backbone (Independent and Synchronized do no learning at all; MA-USFA fine-tunes
only its bounded residual). The `--phase` flag accepts the paper names below;
`transfer` / `l1` / `composer` are kept as legacy aliases.

Point every rule at the shipped checkpoint:

```bash
CK=checkpoints/sf_psi.pth
```

### w-homogeneous target task (one shared task for the whole network)

```bash
# Independent — per-agent greedy composition, fully decentralized
python train_sf.py --phase independent  --sf_ckpt $CK --task_seed 11 --eval_episodes 5

# Synchronized — shared-anchor selection over the source library
python train_sf.py --phase synchronized --sf_ckpt $CK --task_seed 11 --eval_episodes 5

# MA-USFA — learned one-hop neighbour composer (fine-tuned on the task)
python train_sf.py --phase ma-usfa      --sf_ckpt $CK --task_seed 11 --episodes 200 --eval_episodes 5

# retraining baseline — IDDQN trained from scratch on the task
python train.py --net gat --reward composite --seed 0
```

### w-heterogeneous target task (independent per-intersection task)

Add `--hetero 1`; the environment reward becomes per-intersection `φ_i·w_i`,
kept consistent with the per-agent `w` each rule optimizes.

```bash
python train_sf.py --phase independent  --sf_ckpt $CK --hetero 1 --task_seed 11 --eval_episodes 5
python train_sf.py --phase synchronized --sf_ckpt $CK --hetero 1 --task_seed 11 --eval_episodes 5
python train_sf.py --phase ma-usfa      --sf_ckpt $CK --hetero 1 --task_seed 11 --episodes 200 --eval_episodes 5
```

Each run writes `sf_s0/<phase>/<phase>_<homo|hetero>_result.json` with the mean
`φ·w` return and the full physical-metric panel: travel time, throughput,
average queue, average waiting time, average speed, and fairness dispersion.
Compare the rules on these physical metrics, not on the `φ·w` objective alone.

## One-command runners

```bash
./run_all.sh              # all three rules + retraining, both task families
./run_all.sh 0 11         # explicit SEED=0 TASK_SEED=11
./run_extended.sh         # the four robustness sweeps below
./run_extended.sh budget  # just one sweep (hetero | rollouts | budget | ksweep)
```

Windows equivalents: `run_tsc.bat` and `run_tsc_extended.bat`.

## Robustness sweeps (`run_extended.sh`)

- **hetero** — retraining from scratch on the heterogeneous target task.
- **rollouts** — 10 evaluation rollouts per rule for a single-seed variance
  estimate (greedy evaluation is deterministic, so the spread is a determinism
  check).
- **budget** — MA-USFA fine-tuning budget 100 / 200 / 500 episodes.
- **ksweep** — Synchronized source-library size K=1..5 against the
  library-independent Independent reference.

## What was added on top of the IDDQN reference

- `composite_reward.py` — the successor-feature reward `make_sf_reward(w)`,
  per-intersection `set_w_map`, task samplers, and `make_source_library`. The
  original scalar reward path is unchanged.
- `model.py` — `SFQNet` (predicts `ψ`, with a closed-form `.q_values(s,w)`) and
  `NeighborComposer` (the `ψ·w` base plus a zero-initialized one-hop
  graph-attention residual).
- `agent.py` — `SFReplayBuffer` and `SFAgent` (the usfa / synchronized / MA-USFA
  modes). The original IDDQN `Agent` is untouched, so the retraining baseline is
  exactly the reference.
- `train_sf.py` — the entry point for pre-training and the three rules.
- `shapecheck.py` — a pure-numpy mirror that validates the tensor contractions.

## Files

- `train_sf.py` — pre-training + the three composition rules (entry point).
- `train.py` — the IDDQN retraining baseline (unchanged reference).
- `composite_reward.py`, `model.py`, `agent.py`, `graph.py`, `utils.py` — the
  environment reward, networks, agent, road graph, and helpers.
- `checkpoints/sf_psi.pth` — pre-trained shared successor features (seed 0).
- `reference_results/` — the exact result JSON behind the paper's tables
  (main matrix, K sweep, budget sweep, rollout variance, hetero retraining).
- `data/` — SUMO Manhattan 28×7 network files go here (see `data/README.md`).

