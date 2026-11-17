# Running the repaired MA-USFA implementation

This is the current execution guide. README bodies are retained from the original release, with a short pointer here. Training runs create their own artifacts and never promote old cached tables or the historical traffic SF checkpoint to current evidence.

## Install

Use Python 3.10 or newer in a fresh environment. From the repository root:

```bash
python -m pip install -r requirements.txt
```

Traffic additionally needs `python -m pip install -r requirements-traffic.txt` and a working SUMO executable. The recorded traffic server used Python 3.10.20, PyTorch 2.5.1, NumPy 2.2.6, SUMO-RL 1.4.5 and SUMO/TraCI 1.23.1. The controlled experiment artifacts were trained with the environment recorded in their frozen manifest. Cross-platform bitwise identity is not assumed.

## Controlled experiment

```bash
cd toy
python run_crossover.py --quick --output results/smoke
```

This is a short interface and learning smoke run, not a paper experiment. It leaves the independent test unopened. Use a fresh output directory for each run.

The full attention-DDQN main-method training recipe is:

```bash
python run_toy.py --output results/repaired
python run_toy.py --stage freeze --output results/repaired
python run_toy.py --stage test --allow-test --output results/repaired
```

The training command uses five coupling levels, three training seeds, 81,920 lower and 81,920 upper transitions, and the fixed 64/16 development split. Checkpoints are selected by one mean over all 16 validation objectives. The final explicit command evaluates the frozen controller on 32 test objectives with no optimizer updates. It creates an exclusive access receipt before reading the test objectives; an interrupted test is preserved rather than silently retried. The test set in this public release has already been evaluated for the paper and is no longer secret.

The source policies are fixed components obtained in the uncoupled world. The SF predictor includes all five reward features, including blocking with coefficient -2. The upper attention network learns from team reward and selects one frozen source action per agent. It never changes source-policy parameters or replaces those candidate actions by an unrestricted residual action.

`--optimizer ppo-surrogate` retains the historical synthetic alternative: one Monte Carlo trajectory followed by one PPO-style surrogate update. At the behavior parameters clipping is inactive; the paper labels it **PG surrogate**, not multi-epoch PPO. It uses the same strict source-entry interface and train/validation objectives. The traffic PPO experiment is not part of the current paper.

Full component and shared-policy implementations are under `toy/repaired/evidence_ablations.py` and `evidence_shared_baseline.py`; the exact frozen-matrix evaluator is `evidence_evaluate.py`. The shipped numerical evidence under `evidence/synthetic/` identifies the executed matrix and its original fingerprints. The main/default CLI trains the main method; it does not silently claim to rerun all ablations.

The old team/library-size sweep drivers no longer manufacture figures from old caches. Their historical results are not used in the current manuscript.

## Traffic experiment

```bash
cd tsc
python run.py --stage prepare --scene-dir ../data/raw_data/manhattan_28x7
python run.py --stage all --scene-dir ../data/raw_data/manhattan_28x7 --output runs/traffic_ddqn_seed100
```

Place `manhattan_28x7.net.xml` and `manhattan_28x7.rou.xml` in the chosen scene directory. Paths are relative to the working directory. From `tsc`, Windows CMD can run `run_tsc.bat --scene-dir ../data/raw_data/manhattan_28x7`; do not append a backslash to the BAT filename. `bash run_all.sh` calls the same entry point. See `tsc/CURRENT_RUN_GUIDE.md` for stage-by-stage commands, resume checks and resource limits.

The pipeline creates scene-bound sources, collects training reference trajectories, fits the SF predictor, freezes both lower components, trains the normalized attention-DDQN composer on training objectives, then runs closed-loop SUMO evaluation on separate Homo/Hetero validation objectives. Every source is tied to the network/route hashes, signal identities and simulator settings. Earlier unbound checkpoints cannot be relabeled as scene-verified. A new source-training run produces new evidence and is not guaranteed to reproduce the earlier server table byte for byte.

Traffic baselines `independent_entry` and `sync_entry` use the same approximate SF candidate scores. They differ from the exact primitive-action Independent and exact Sync references in the controlled experiment. Five replays of traffic seed 10000000 are paired repeats of one traffic realization, not five independent traffic seeds.

## Evidence and checks

`evidence/synthetic/` contains the complete published 32-target test rows, the original frozen manifest, grouped CSVs, full negative and positive ablations, and validation-only optimizer/retraining summaries. `python verify_evidence.py` recomputes the frozen-test aggregates from the raw rows and checks their fingerprints without running training. Scientific source-copy fingerprints are recorded with each implementation.

The release also includes the frozen model parameters. From the repository root:

```bash
python verify_evidence.py
python replay_frozen.py --method all --split test --output runs/frozen_test_replay.json
```

The second command evaluates all 105 frozen controllers on the already-published 32 test objectives and compares every learned-controller return with its saved value. It performs no training or checkpoint selection. This is reproduction of published evidence, not a new independent test. Use a fresh output path; existing reports are preserved. The checked release replay is recorded in `validation/frozen_test_replay.json`.

The release validation report records the actual checks run on this checkout. Traffic tensor, reward, source-provenance, device and mocked closed-loop tests can run without SUMO. A real Manhattan rollout additionally requires the server's scenario files and SUMO installation; local contract tests do not certify simulator results.
