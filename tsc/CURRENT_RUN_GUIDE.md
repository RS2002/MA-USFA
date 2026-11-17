# Current traffic implementation

This guide supersedes commands in the historical README files, which are retained for continuity.

From this directory:

```bash
python run.py --stage prepare --scene-dir ../data/raw_data/manhattan_28x7
python run.py --stage all --scene-dir ../data/raw_data/manhattan_28x7
```

`run_tsc.bat` (Windows CMD: `run_tsc.bat`, without a trailing slash) and `bash run_all.sh` call the same `--stage all` pipeline. Set `PYTHON` to choose an interpreter. Run `python run.py --help` for options. All command-line relative paths resolve from the working directory; the shell launchers first enter their own directory.

The scene folder must contain `manhattan_28x7.net.xml` and `manhattan_28x7.rou.xml`. Install PyTorch, numpy and psutil, plus the traffic dependencies and a working SUMO executable. The Python files do not depend on any parent research folder.

The pipeline freezes the original Homo/Hetero weight-generation rule (train task seed12, validation task seed11) and the 600-second SUMO settings. It trains five source policies with the paper's per-node DQN source objective (local reward targets, one transition update, target copy every 100 episodes), collects reference transitions on training tasks, fits task-conditioned vector SF, freezes source/SF parameters, trains one attention-DDQN selector across both training tasks, then evaluates both validation tasks with the same paired traffic seed10000000. Each agent executes exactly one frozen source's action. Training uses node-mean team reward; reported team return sums all node rewards. Five repeated replays of the same traffic seed are not independent traffic seeds.

Defaults follow the measured development recipe: 5 episodes/source, 20 reference episodes, 200 SF updates, 30 upper episodes, 5 paired replays/task; 2 torch threads, 8 GiB RSS limit, at least 20 GiB free disk, and a 12-hour invocation limit. These modest defaults are not a claim of converged traffic policies. Source training in the release keeps the source objective and target-copy schedule and adds explicit initialization seeds, terminal/missing-reward checks, and network/route hashes, node identities and simulation settings. It therefore produces new artifacts; earlier server table values are not shipped as fabricated reproduction outputs.

Expensive stages can be invoked separately with `--stage sources`, `reference`, `sf`, `upper`, or `evaluate`. Re-run with exactly the same settings and output to reuse completed stages only after checking their hashes. Interrupted stages are preserved and rejected; use a fresh `--output` to restart. Changing scene, code, seed, budget or device also requires a fresh output. No checkpoint selection or parameter update occurs during evaluation. Only train and validation objectives are supplied; a new independent-test experiment requires a separately frozen protocol.

The incompatible historical `checkpoints/sf_psi.pth` and cached traffic tables are archived outside this release. This pipeline trains new checkpoints. `train.py`, `train_sf.py`, `agent.py` and `model.py` no longer execute the withdrawn primitive-action residual composer. Traffic PPO is outside the current manuscript.

Validation without SUMO:

```bash
python shapecheck.py
python -m unittest tsc_repaired.test_train_sources_v2 development_followup.test_train_budget paper_aligned_followup.test_legacy_tasks test_release_pipeline
```

These contracts use small mocked environments and validate implementation properties; the full Manhattan simulation still requires the declared server scene and SUMO installation.
