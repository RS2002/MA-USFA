# Repaired controlled experiments

The current implementation uses PyTorch, two agents, four frozen source
policies and a five-dimensional feature vector per agent (four harvest
features plus the positive blocking indicator). The blocking coefficient is
-2 in every scalar reward readout. Every upper action selects one source
entry per agent and executes that source's primitive action.

From this directory:

```bash
python run_crossover.py --quick --output results/smoke
python run_crossover.py --output results/repaired
```

The first command is a short interface/training check. The second trains all
five coupling levels and three seeds with 81,920 lower and 81,920 upper
transitions per run. Lower training loads only the 64 training objectives.
Upper training uses those objectives and selects one shared checkpoint by the
mean return over all 16 validation objectives. It never trains on validation
objectives or reads test weights. Output directories must be fresh.

Once all development decisions are final:

```bash
python run_toy.py --stage freeze --output results/repaired
python run_toy.py --stage test --allow-test --output results/repaired
python make_figs.py --evaluation results/repaired/evaluation_test.json --output results/repaired/test_comparison.png
```

Each completed unit records a receipt binding its upper checkpoint, lower checkpoint, coupling, seed, budgets and objective protocol. Freezing verifies those receipts and binds checkpoint, development records, code and protocol fingerprints.
The test command checks them, writes an exclusive access receipt, then evaluates
32 test objectives without updates. An existing access receipt prevents silent
retries, including after interruption. `--quick` runs cannot be frozen as final
evidence. The included test split is the paper's already evaluated split;
new method development requires a new independent test protocol.

The default upper optimizer is DDQN. `--optimizer ppo-surrogate` reproduces the
historical synthetic single-trajectory, single-update clipped-surrogate
experiment; it is reported as a PG surrogate, not multi-epoch PPO.
`--optimizer pg` is the separate Monte Carlo policy-gradient diagnostic.

For the separate validation-target-specific retraining diagnostic:

```bash
python run_retrain.py --output results/validation_scratch
python run_retrain.py --kappas 1 --task-id validation_homo_000 --budget 320 --output results/scratch_smoke
```

The full diagnostic trains 80 independent models (5 coupling levels × 16
validation targets) at the fixed 81,920-transition budget. It reports endpoint
and best-at-or-before-budget returns. These target-fitted validation models
are not used as the independent-test scratch comparator.

`repaired/evidence_ablations.py` contains the entry-greedy, no-cross-token
attention, no-SF and no-teammate-lower controls.
`repaired/evidence_shared_baseline.py` contains the full-state, task-conditioned
25-primitive-joint-action DDQN at either budget.
`repaired/evidence_evaluate.py` supplies the stricter full-matrix frozen
validation/test evaluator used by the supplementary experiment matrix.
These share the same kernel, source definitions and task protocol as the
public main runner. Their API metadata and tests distinguish incomplete runs.

Runtime protection uses one CPU worker, one or two numerical threads, a 3 GiB
worker memory high-water limit, at least 20 GiB free disk, and a wall-clock
limit. Change limits only deliberately with the named command-line options.
On Windows, memory monitoring additionally requires `psutil`; POSIX systems use
Python's standard `resource` module. Resource limits stop the run and preserve
partial files; no automatic restart or resume occurs.

Run short numerical and evidence-integrity tests from the repository root:

```bash
PYTHONPATH=toy python -m unittest repaired.test_evidence_ablations repaired.test_evidence_shared_baseline repaired.test_evidence_evaluate repaired.test_public_release -q
```

The cached `results/crossover.json`, `results/ext.json` and their figures from
the earlier erroneous release have been moved out of the public release into
the complete provenance backup. Current runners neither read nor regenerate
those historical outputs. Historical team-size/library-size sweeps and old
gating claims are not current results. The earlier Python files were also
backed up before replacement. `repaired/SOURCE_PROVENANCE.json` records the
audited scientific source mapping.
