"""Current traffic feature/reward implementation.

Feature order is queue, waiting-time, pressure, relative-speed score, fairness.
The reward is the dot product with the declared task weights.
"""
from tsc_repaired.features import DEFAULT_W, FEATURE_NAMES, TrafficFeatureExtractor, validate_weight_matrix
from tsc_repaired.task_protocol_v2 import default_anchors

def make_composite_reward(weights=None):
    extractor = TrafficFeatureExtractor()
    chosen = DEFAULT_W if weights is None else weights
    return lambda signal: extractor.reward(signal, weights=chosen)

def make_source_library():
    return default_anchors().copy()
