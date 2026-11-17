"""Manifest-bound Manhattan/SUMO scenario adapter.

SUMO is imported lazily so the laptop can run XML and hash checks without the
server environment.  The adapter never searches for fallback network files or
silently reconstructs a grid topology.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

from .config import Manifest, resolve_manifest_path, sha256_file


def _file_for_role(manifest: Manifest, role: str) -> Path:
    entries = manifest.raw.get("files", [])
    matches = [item for item in entries if isinstance(item, dict) and item.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"manifest must declare exactly one file with role={role!r}")
    item = matches[0]
    path = resolve_manifest_path(manifest, str(item.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"declared {role} file is missing: {path}")
    expected = str(item.get("sha256", ""))
    if expected and expected != sha256_file(path):
        raise ValueError(f"SHA-256 mismatch for declared {role}: {path}")
    required_name = str(item.get("required_name", ""))
    if required_name and path.name != required_name:
        raise ValueError(f"{role} must be named {required_name!r}, got {path.name!r}")
    return path


@dataclass(frozen=True)
class ScenarioFiles:
    network: Path
    routes: Path

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "ScenarioFiles":
        return cls(_file_for_role(manifest, "network"), _file_for_role(manifest, "routes"))

    def hashes(self) -> dict[str, str]:
        return {"network": sha256_file(self.network), "routes": sha256_file(self.routes)}


@dataclass(frozen=True)
class NetworkSummary:
    network: str
    junction_count: int
    signal_logic_count: int
    controlled_signal_ids: tuple[str, ...]


def inspect_network(path: str | Path) -> NetworkSummary:
    network = Path(path).expanduser().resolve()
    if not network.is_file():
        raise FileNotFoundError(network)
    try:
        root = ET.parse(network).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid SUMO network XML: {network}") from exc
    junctions = [node for node in root.findall("junction") if node.get("type") != "internal"]
    logics = root.findall("tlLogic")
    ids = tuple(sorted({str(node.get("id")) for node in logics if node.get("id") is not None}))
    return NetworkSummary(str(network), len(junctions), len(logics), ids)


def inspect_routes(path: str | Path) -> dict[str, Any]:
    routes = Path(path).expanduser().resolve()
    if not routes.is_file():
        raise FileNotFoundError(routes)
    try:
        root = ET.parse(routes).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid SUMO route XML: {routes}") from exc
    return {
        "path": str(routes),
        "vehicle_count": len(root.findall("vehicle")),
        "flow_count": len(root.findall("flow")),
        "route_count": len(root.findall("route")),
    }


@dataclass(frozen=True)
class ScenarioSpec:
    files: ScenarioFiles
    delta_time: int = 10
    horizon_steps: int = 60
    yellow_time: int = 2
    min_green: int = 5
    max_green: int = 50
    scale: float = 2.0

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> "ScenarioSpec":
        scenario = manifest.raw.get("scenario", {})
        if not isinstance(scenario, dict):
            scenario = {}

        def positive(name: str, default: int | float) -> int | float:
            value = scenario.get(name, default)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"scenario.{name} must be positive")
            return value

        return cls(
            files=ScenarioFiles.from_manifest(manifest),
            delta_time=int(positive("delta_time", 10)),
            horizon_steps=int(positive("horizon_steps", 60)),
            yellow_time=int(positive("yellow_time", 2)),
            min_green=int(positive("min_green", 5)),
            max_green=int(positive("max_green", 50)),
            scale=float(positive("scale", 2.0)),
        )

    def summary(self) -> dict[str, Any]:
        network = inspect_network(self.files.network)
        return {
            "files": self.files.hashes(),
            "network": {
                "path": network.network,
                "junction_count": network.junction_count,
                "signal_logic_count": network.signal_logic_count,
                "controlled_signal_ids": list(network.controlled_signal_ids),
            },
            "routes": inspect_routes(self.files.routes),
            "delta_time": self.delta_time,
            "horizon_steps": self.horizon_steps,
            "yellow_time": self.yellow_time,
            "min_green": self.min_green,
            "max_green": self.max_green,
            "scale": self.scale,
        }

    def build_parallel_env(self, reward_fn: Any, tripinfo_path: str | Path | None = None,
                           use_gui: bool = False, sumo_seed: int | None = None) -> Any:
        """Construct sumo_rl's parallel_env using only declared files."""
        try:
            import sumo_rl  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "SUMO runner dependency is missing; install the server-locked "
                "sumo-rl/libsumo environment before starting the pilot"
            ) from exc
        extra = ["--scale", str(self.scale), "--no-warnings"]
        if tripinfo_path is not None:
            # sumo-rl 1.4.5 tokenizes additional_sumo_cmd with str.split(),
            # rather than a shell. Quoting Windows absolute paths would leave
            # literal quote characters in the SUMO argument. A relative path
            # also works when the containing project folder has spaces.
            trip = Path(tripinfo_path).resolve()
            trip.parent.mkdir(parents=True, exist_ok=True)
            trip_arg = os.path.relpath(trip, Path.cwd()).replace(os.sep, "/")
            if any(character.isspace() for character in trip_arg):
                raise ValueError("tripinfo output name must not contain spaces")
            extra.extend(["--tripinfo-output", trip_arg])
        seed_args = {} if sumo_seed is None else {"sumo_seed": int(sumo_seed)}
        return sumo_rl.parallel_env(
            net_file=str(self.files.network),
            route_file=str(self.files.routes),
            use_gui=bool(use_gui),
            num_seconds=self.horizon_steps * self.delta_time,
            delta_time=self.delta_time,
            yellow_time=self.yellow_time,
            min_green=self.min_green,
            max_green=self.max_green,
            reward_fn=reward_fn,
            additional_sumo_cmd=" ".join(extra),
            **seed_args,
        )


def write_scenario_probe(manifest: Manifest, output: str | Path) -> dict[str, Any]:
    """Write an XML/hash probe without importing or launching SUMO."""
    spec = ScenarioSpec.from_manifest(manifest)
    result = {"status": "PASS", "scenario": spec.summary(), "launches_sumo": False}
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result

