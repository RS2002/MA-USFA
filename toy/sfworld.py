"""Repaired SFWorld kernel: two agents, five features including blocking.

The scalar team reward is sum_i theta_i dot phi_i with
 theta_i=[w_i0,...,w_i3,-2]. Joint state, reward and transition semantics
match the exact finite-horizon evaluation used in the current paper.
"""
from repaired.legacy_world import ACTIONS, SFWorldExact
SFWorld = SFWorldExact
__all__=['ACTIONS','SFWorld','SFWorldExact']
