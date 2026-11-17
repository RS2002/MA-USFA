# Toy experiment — controlled SF composition under coupling

> **Current implementation (2026-11-17):** Use [../REPRODUCING.md](../REPRODUCING.md) and `python run_toy.py --help`. The text below describes the historical release. Current SFs include the blocking penalty; training and independent testing are separate, and old cached figures are not current results.

A small multi-agent grid world (SFWorld) that isolates, in the smallest
controlled form, when per-agent successor-feature composition is safe and when
it is not. It reproduces the paper's controlled claims and produces the three
main-text figures. Runs on `numpy` + `matplotlib` only — no GPU, no simulator.

## The mechanism in one paragraph

Each agent's reward is `r_i = φ_i(s')·w_i − penalty·blocked_i`. The collision
penalty lives **outside** the successor features (`φ` is harvest-only), so the
closed-form SF value `ψ·w` is structurally blind to it. A transition-coupling
dial `κ` controls how often agents contend for the same cell. At `κ=0` the
dynamics factorize, nobody is ever blocked, and independent composition is
exact (the free region). As `κ` rises, blocking becomes frequent, independent
composition keeps colliding with itself, and only a coupling-aware composer
recovers the lost value (the paid region).

## Two target tasks

The crossover driver evaluates every rule on two task families:

- **A_distinct** — agents want different features; their greedy actions rarely
  contend. A null control: Independent stays at or above Synchronized at every
  `κ`.
- **B_overlap** — agents want the same scarce feature. This is where the safety
  crossover appears: Independent beats Synchronized in the free region
  (`κ=0`) but falls **below** it once coupling makes its independent argmax
  self-collide.

## How to run each experiment

Everything writes into `results/`. All three drivers accept `--quick` for a
fast reduced-budget sanity check.

### 1. The headline safety crossover (Independent vs Synchronized)

```bash
python run_crossover.py            # full run
python run_crossover.py --quick    # fast smoke test
```

For one controlled world and one source library, this sweeps several test tasks
across coupling levels `κ` and evaluates every rule in closed form or by
training:

- **Independent** — per-agent greedy composition over the library.
- **Synchronized** — shared-anchor selection over the same library.
- **joint-GPI** — the prior SF-to-MARL transfer that enumerates the joint
  action space `A^N` (included here as a scalability contrast; it becomes
  infeasible at scale, which is why the TSC experiment omits it).
- **MA-USFA** — the learned composer, trained on each task.
- **retraining** — per-agent Q-learning on the true penalty-inclusive reward.

Writes `results/crossover.json` and `results/crossover_tables.txt`.

### 2. Free-region correctness, gating, and recovery

```bash
python run_toy.py                  # full pipeline
python run_toy.py --quick          # reduced budget
```

Shows three things: at `κ=0` closed-form transfer is exact regardless of reward
coupling; a cheaply measured interference rate predicts which (task, regime)
pairs actually need the composer; and in the paid region MA-USFA recovers
toward the retraining baseline while driving the collision rate to zero. Writes
`results/results.json`, `results/tables.txt`, and figures under `results/figs/`.

### 3. Team-size and library-size sweeps

`run_extend.py` is a resumable driver (it caches the library and the task-free
`ψ` once per configuration, then computes one cell at a time, so it fits short
compute budgets):

```bash
# cache the library + task-free successor features for a configuration
python run_extend.py prep  --N 2 --K 4 --lib corners
# one (task, kappa) base row: Independent / Synchronized / joint-GPI / per-source / retraining
python run_extend.py cell  --N 2 --K 4 --lib corners --task B_overlap --kappa 0.5
# train ONE MA-USFA composer flavor for that cell
python run_extend.py comp  --N 2 --K 4 --lib corners --task B_overlap --kappa 0.5 --flavor comm
# assemble tables once cells are computed
python run_extend.py tables --mode sweep_N      # team-size sweep N=3,4,5
python run_extend.py tables --mode sweep_K      # library-size sweep K=1..8
python run_extend.py status --mode main         # see which cells are done
```

The team-size sweep checks that the crossover survives more agents (and that
Independent's paid-region collapse sharpens with contention); joint-GPI is
computed for `N<=3` and omitted for `N>=4` to make the `A^N` wall visible inside
the toy. The library-size sweep shows Synchronized's coverage effect is
regime-dependent while MA-USFA is immune to library content. Writes
`results/ext.json` and `results/ext_tables.txt`.

### Regenerate the paper figures

```bash
python make_figs.py
```

Reads the cached `results/crossover.json` and `results/ext.json` and writes
`fig1_crossover.png`, `fig2_k_coverage.png`, `fig3_n_sweep.png` (the same three
figures used in the paper) back into `results/`.

## Files

- `sfworld.py` — the environment: features, blocking dynamics, collision
  penalty, evaluation, and diagnostics (conflict rates, decoupled value
  iteration).
- `sf_learn.py` — per-task SF training, task-free USFA features, the
  penalty-inclusive Q baseline, and the closed-form composition policies.
- `library.py` — source-library construction and the Independent / Synchronized
  / joint-GPI closed-form policies.
- `composers.py` — the learned MA-USFA composer variants with per-agent honest
  TD training.
- `gate.py` — the interference-rate gating diagnostic.
- `run_toy.py`, `run_crossover.py`, `run_crossover_ckpt.py`, `run_extend.py` —
  the drivers described above.
- `make_figs.py` — figure regeneration from cached JSON.
- `results/` — cached result JSON, human-readable tables, and the paper PNGs.
