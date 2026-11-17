"""One-hop signal adjacency from SUMO XML connectivity, without ID heuristics."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def signal_adjacency(network_path: str | Path, node_ids) -> np.ndarray:
    ids = tuple(map(str, node_ids))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("controlled signal IDs must be nonempty and unique")
    root = ET.parse(network_path).getroot()
    declared = {item.get("id") for item in root.findall("tlLogic")}
    if not set(ids).issubset(declared):
        raise ValueError("controlled signal IDs are missing from network tlLogic")
    edges = {item.get("id"): item for item in root.findall("edge")}
    junction_signals = {}
    for connection in root.findall("connection"):
        signal = connection.get("tl")
        edge = edges.get(connection.get("from"))
        if signal not in ids or edge is None or not edge.get("to"):
            continue
        junction = edge.get("to")
        previous = junction_signals.get(junction)
        if previous is not None and previous != signal:
            raise ValueError("ambiguous junction-to-signal assignment in SUMO XML")
        junction_signals[junction] = signal
    index = {node: i for i, node in enumerate(ids)}
    mask = np.eye(len(ids), dtype=bool)
    for edge in edges.values():
        start = junction_signals.get(edge.get("from"))
        end = junction_signals.get(edge.get("to"))
        if start in index and end in index:
            i, j = index[start], index[end]
            mask[i, j] = mask[j, i] = True
    return mask
