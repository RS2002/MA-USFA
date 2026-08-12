"""Coupling-graph utilities for traffic signal control.

This module mirrors the ICLR traffic probing idea, but uses a safer training
default: prefer the real SUMO road topology when available, and only fall back
to ID-based grid parsing when topology probing fails. Deriving neighbours from
the actual road network is more reliable than assuming agent names alone define
the coupling graph.

It builds:
  * a 1-hop coupling adjacency C with self-loops, and
  * k-hop support matrices S (S_ij = 1 iff graph distance(i, j) <= k).
"""

import re
import numpy as np

_INTERSECTION_PATTERN = re.compile(r"^intersection_(\d+)_(\d+)$")
_ALNUM_PATTERN = re.compile(r"^([A-Za-z]+)[_\-]?(\d+)$")
_NUM_PATTERN = re.compile(r"^(\d+)[_\-](\d+)$")


def _letter_code_to_int(text):
    """Map A->1, B->2, ..., AA->27 for grid-like IDs such as A0 / B12."""
    return sum((ord(ch.lower()) - 96) for ch in text)


def try_parse_grid_coords(agent_ids):
    """Best-effort grid-coordinate parsing for several common ID formats.

    Supported patterns:
      * intersection_<col>_<row>
      * A0 / B12 / AA3
      * 3_5 / 12-4

    Returns a list of (x, y) tuples, or None if any ID cannot be parsed.
    """
    coords = []
    for aid in agent_ids:
        text = str(aid)
        m = _INTERSECTION_PATTERN.fullmatch(text)
        if m is not None:
            coords.append((int(m.group(1)), int(m.group(2))))
            continue

        m = _ALNUM_PATTERN.fullmatch(text)
        if m is not None:
            coords.append((_letter_code_to_int(m.group(1)), int(m.group(2))))
            continue

        m = _NUM_PATTERN.fullmatch(text)
        if m is not None:
            coords.append((int(m.group(1)), int(m.group(2))))
            continue

        return None
    return coords


def parse_grid_coords(agent_ids):
    """Strict wrapper around try_parse_grid_coords (raises on failure)."""
    coords = try_parse_grid_coords(agent_ids)
    if coords is None:
        raise ValueError(
            "Agent ids could not be parsed as grid coordinates. "
            "Supported patterns include 'intersection_<col>_<row>', 'A0', and '3_5'."
        )
    return coords


def build_grid_adjacency(agent_ids):
    """Build 4-neighbor (Manhattan) grid adjacency with self-loops."""
    coords = parse_grid_coords(agent_ids)
    n = len(coords)
    C = np.eye(n, dtype=np.float32)
    for i, (xi, yi) in enumerate(coords):
        for j in range(i + 1, n):
            xj, yj = coords[j]
            if abs(xi - xj) + abs(yi - yj) == 1:
                C[i, j] = 1.0
                C[j, i] = 1.0
    return C


def infer_topology_adjacency(env, agent_ids):
    """Infer adjacency from the real SUMO road topology.

    Primary rule: connect two controlled intersections if there exists a road
    edge whose from-junction and to-junction are both controllable TLS ids.

    Fallback rule: if the direct edge-endpoint API is unavailable, connect two
    TLS when they share any controlled edge prefix, similar to the looser ICLR
    probe fallback.

    Parameters
    ----------
    env : object or None
        Active sumo_rl / SUMO environment exposing a traci-like ``sumo`` conn.
    agent_ids : list[str]
        Sorted traffic-signal ids.

    Returns
    -------
    tuple[np.ndarray | None, str]
        (adjacency_with_self_loops, source_description). adjacency is None on
        failure so the caller can decide whether to fall back.
    """
    if env is None:
        return None, "topology unavailable: env is None"

    try:
        conn = getattr(getattr(env, "unwrapped", env), "sumo", None)
        if conn is None:
            import traci
            conn = traci
    except Exception as exc:
        return None, f"topology unavailable: {exc}"

    n = len(agent_ids)
    idx = {aid: i for i, aid in enumerate(agent_ids)}
    A = np.eye(n, dtype=np.float32)

    # Preferred: use the real road graph via edge from/to junctions.
    try:
        edge_ids = conn.edge.getIDList()
        direct_edges = 0
        for edge_id in edge_ids:
            edge_name = str(edge_id)
            if edge_name.startswith(":"):
                continue  # SUMO internal connector edge inside a junction
            try:
                src = conn.edge.getFromJunction(edge_id)
                dst = conn.edge.getToJunction(edge_id)
            except Exception:
                continue
            if src in idx and dst in idx and src != dst:
                A[idx[src], idx[dst]] = 1.0
                A[idx[dst], idx[src]] = 1.0
                direct_edges += 1
        if direct_edges > 0:
            return A, "traci-edge-endpoints"
    except Exception:
        pass

    # Fallback: connect TLS sharing any controlled edge prefix.
    try:
        tls_edges = {}
        for tls_id in agent_ids:
            lanes = conn.trafficlight.getControlledLanes(tls_id)
            edges = {
                lane.rsplit("_", 1)[0]
                for lane in lanes
                if isinstance(lane, str) and not lane.startswith(":")
            }
            tls_edges[tls_id] = edges

        shared = 0
        for i, aid in enumerate(agent_ids):
            for j in range(i + 1, n):
                bid = agent_ids[j]
                if tls_edges[aid] & tls_edges[bid]:
                    A[i, j] = 1.0
                    A[j, i] = 1.0
                    shared += 1
        if shared > 0:
            return A, "traci-shared-controlled-edges"
    except Exception as exc:
        return None, f"topology fallback failed: {exc}"

    return None, "topology inference produced no edges"


def build_coupling_adjacency(agent_ids, env=None, method="auto"):
    """Build the 1-hop coupling adjacency used for hop supports.

    Parameters
    ----------
    agent_ids : list[str]
        Sorted traffic-signal ids.
    env : optional
        Active SUMO/sumo_rl environment. Required for topology-based probing.
    method : str
        "auto"     -> prefer topology, then fall back to grid parsing
        "topology" -> topology only (raises if it fails)
        "grid"     -> grid parsing only

    Returns
    -------
    tuple[np.ndarray, str]
        (adjacency_with_self_loops, source_description)
    """
    method = str(method).lower()
    if method not in {"auto", "topology", "grid"}:
        raise ValueError(f"Unknown coupling-graph method: {method}")

    if method in {"auto", "topology"}:
        A, source = infer_topology_adjacency(env, agent_ids)
        if A is not None:
            np.fill_diagonal(A, 1.0)
            return A.astype(np.float32), source
        if method == "topology":
            raise ValueError(f"Failed to infer topology adjacency: {source}")

    if method in {"auto", "grid"}:
        A = build_grid_adjacency(agent_ids)
        np.fill_diagonal(A, 1.0)
        return A.astype(np.float32), "grid-from-ids"

    raise ValueError("Could not build coupling adjacency")


def k_hop_support(C, k):
    """Return the k-hop support matrix S (S_ij = 1 iff dist(i, j) <= k).

    Parameters
    ----------
    C : np.ndarray [N, N]
        1-hop adjacency with self-loops.
    k : int or None
        Hop radius. k = 0 -> identity (per-agent); k >= diameter -> all-ones
        (fully joint); k = None -> all-ones. Negative k is treated as 0.
    """
    n = C.shape[0]
    if k is None:
        return np.ones((n, n), dtype=np.float32)
    if k <= 0:
        return np.eye(n, dtype=np.float32)

    reach = (C > 0).astype(np.float32)
    S = reach.copy()
    for _ in range(k - 1):
        S = ((S @ reach) > 0).astype(np.float32)
    S = (S > 0).astype(np.float32)
    np.fill_diagonal(S, 1.0)
    return S


def describe_support(S):
    """Small helper for logging: neighborhood-size summary and extremes."""
    sizes = S.sum(axis=1)
    return {
        "avg_neighbors": float(sizes.mean()),
        "min_neighbors": int(sizes.min()),
        "max_neighbors": int(sizes.max()),
        "is_identity": bool(np.allclose(S, np.eye(S.shape[0]))),
        "is_full": bool(np.allclose(S, np.ones_like(S))),
    }


if __name__ == "__main__":
    ids = sorted(f"intersection_{c}_{r}" for c in range(1, 4) for r in range(1, 4))
    C = build_grid_adjacency(ids)
    for k in [0, 1, 2, 10]:
        S = k_hop_support(C, k)
        print(f"k={k}: {describe_support(S)}")