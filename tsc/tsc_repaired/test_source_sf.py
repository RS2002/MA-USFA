"""Dependency-light contract smoke for source_library and conditional_sf.

Run from the traffic folder with
`python -m tsc_repaired.test_source_sf`.
"""
from __future__ import annotations
import numpy as np
from .source_library import FrozenSourceLibrary, k1_entry_consistency
from .conditional_sf import numpy_terminal_target

def main() -> int:
    actions = np.arange(1 * 3 * 4, dtype=np.int64).reshape(1, 3, 4) % 5
    library = FrozenSourceLibrary.from_array(actions, action_count=5)
    assert k1_entry_consistency(library, 1)
    assert np.array_equal(library.actions_for_entries([0, 0, 0, 0], 1), actions[0, 1])
    try:
        library.actions[0, 0, 0] = 9
    except ValueError:
        pass
    else:
        raise AssertionError("frozen action table is mutable")
    features = np.ones((2, 4, 5), dtype=np.float64)
    next_prediction = np.full_like(features, 3.0)
    target = numpy_terminal_target(features, next_prediction, np.array([0, 1]), .99)
    assert np.allclose(target[0], 3.97)
    assert np.allclose(target[1], 1.0)
    print("source/SF contract smoke PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
