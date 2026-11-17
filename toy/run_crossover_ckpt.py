"""Compatibility entry for repaired training, freezing and checkpoint evaluation.

Training: --stage train; after development: --stage freeze; evaluation-only
independent test: --stage test --allow-test. Old checkpoints are incompatible.
"""
from repaired.public_run import main
if __name__ == '__main__':
    raise SystemExit(main())
