"""Teammate-task-conditioned vector TD predictor with frozen source continuation.

Blocking is an explicit feature in the TD target and scalar readout. No
lower predictor is adapted to a validation or test target.
"""
from repaired.paper_aligned_lower import (ContextActionSF, candidate_values,
    source_anchor_probabilities, sf_batch_loss, train_context_sf, load_context_sf)
__all__=['ContextActionSF','candidate_values','source_anchor_probabilities',
         'sf_batch_loss','train_context_sf','load_context_sf']
