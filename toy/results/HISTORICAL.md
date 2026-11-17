# Historical cached outputs

The old cached JSON, tables and figures have been moved out of the public
release to the complete provenance backup. Their previous numerical results
came from the erroneous implementation and are excluded from current evidence.

Current entry points write only into a new `results/repaired` directory (or an
explicit fresh `--output` path). `make_figs.py` accepts only complete repaired
evaluation reports and never consumes the old cached results.
