"""Same attention utilities with one explicit change to team-Q mixing.

Normalizing a given model's positive mixing weights does not change its
source argmax. This comparison must retrain the upper network to test the
normalization's effect on learning. Raw node-local utility inputs are intact.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from tsc_repaired.task_selector import (
    SELECTOR_VERSION, TaskAttentionSelector, TaskSelectorConfig,
)

NORMALIZED_VERSION = "ma-usfa-traffic-attention-ddqn-normalized-mixer-v1"
VARIANTS = ("control", "normalized")


def parameter_sha256(model):
    """Stable tensor fingerprint, independent of the checkpoint container."""
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class NormalizedTaskAttentionSelector(TaskAttentionSelector):
    """Preserve original source utilities; normalize positive weights to mean 1.

    The normalization spans nodes during centralized team-Q training. It adds
    no information to local source utilities or to decentralized greedy action
    selection. Unlike the control's numerical weights, the normalized mixing
    coefficient itself need not be a one-hop-local quantity.
    """

    def components(self, obs, weights, candidate_actions, sf_values, time_fraction=None):
        parts = super().components(obs, weights, candidate_actions, sf_values, time_fraction)
        raw = parts["mixing"]
        # Scaling first avoids overflow when summing many finite raw weights.
        scaled = raw / raw.max(dim=-1, keepdim=True).values
        mixing = (scaled / scaled.sum(dim=-1, keepdim=True)) * raw.shape[-1]
        # Avoid silent zeros caused by extreme float underflow. In the normal
        # regime this clamp is inactive; re-normalization retains mean one.
        mixing = mixing.clamp_min(torch.finfo(mixing.dtype).tiny)
        mixing = mixing / mixing.mean(dim=-1, keepdim=True)
        if not bool(torch.isfinite(mixing).all()) or not bool((mixing > 0).all()):
            raise FloatingPointError("normalized mixer is non-finite or non-positive")
        return {**parts, "mixing": mixing}

    def spec(self):
        return {**super().spec(), "version": NORMALIZED_VERSION,
                "mixing_normalization": "mean_one",
                "mixing_normalization_scope": "centralized across nodes; source utilities remain one-hop",
                "team_score_reduction": "mean of mean-one positive-weighted selected node utilities"}


def selector_for_variant(config, variant):
    if variant not in VARIANTS:
        raise ValueError("variant must be control or normalized")
    cls = TaskAttentionSelector if variant == "control" else NormalizedTaskAttentionSelector
    return cls(config)


def checkpoint_spec(model, variant):
    spec = model.spec()
    if variant == "control":
        return {**spec, "mixing_normalization": "none"}
    if variant != "normalized" or not isinstance(model, NormalizedTaskAttentionSelector):
        raise ValueError("selector class and ablation variant disagree")
    return spec


def version_for_variant(variant):
    if variant not in VARIANTS:
        raise ValueError("variant must be control or normalized")
    return SELECTOR_VERSION if variant == "control" else NORMALIZED_VERSION


def load_ablation_selector(path: str | Path, device="cpu"):
    """Load only explicitly tagged ablation checkpoints with the right class."""
    saved = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(saved, dict):
        raise ValueError("not an ablation selector checkpoint")
    metadata = saved.get("metadata")
    spec = saved.get("spec")
    if not isinstance(metadata, dict) or not isinstance(spec, dict):
        raise ValueError("ablation selector is missing metadata/spec")
    variant = metadata.get("ablation_variant")
    if variant not in VARIANTS:
        raise ValueError("checkpoint is missing a supported ablation_variant")
    version = version_for_variant(variant)
    normalization = "none" if variant == "control" else "mean_one"
    if saved.get("version") != version or spec.get("version") != version:
        raise ValueError("checkpoint version and ablation variant disagree")
    if spec.get("mixing_normalization") != normalization:
        raise ValueError("checkpoint mixing normalization and variant disagree")
    if spec.get("neighbor_sf_rebroadcast") is not False:
        raise ValueError("ablation checkpoint violates the one-hop no-SF-rebroadcast contract")
    config = TaskSelectorConfig(**saved["config"])
    if spec.get("config") != asdict(config):
        raise ValueError("checkpoint spec and selector config disagree")
    if "adjacency" not in saved:
        raise ValueError("ablation selector checkpoint is missing its topology mask")
    model = selector_for_variant(config, variant).to(device)
    model.load_state_dict(saved["online"], strict=True)
    model.set_adjacency(saved["adjacency"])
    model.eval().requires_grad_(False)
    return model, saved
