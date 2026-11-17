"""Immutable manifest loading and stage gates for the traffic runner."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_STATUS = {"NOT_READY", "READY_FOR_PILOT", "READY_FOR_FORMAL"}
REQUIRED_CONTRACT = (
    "strict_entry",
    "conditional_sf",
    "frozen_source_library",
    "reference_td_uses_executed_context",
    "zero_test_updates",
    "test_split_frozen",
    "k1_entry_consistency",
)
V2_CONTRACT = (
    "strict_entry", "task_weight_conditioned_sf", "frozen_source_library",
    "declared_reference_distribution", "zero_evaluation_updates",
    "held_out_validation", "k1_entry_consistency",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    """A validated, read-only view of a server manifest.

    The loader intentionally validates declarations only.  It does not search
    for alternative data, parse test objectives, import SUMO, or launch a
    training process.
    """

    path: Path
    raw: dict[str, Any]

    @property
    def status(self) -> str:
        return str(self.raw.get("status", ""))

    @property
    def scenario_id(self) -> str:
        return str(self.raw.get("scenario_id", ""))

    @property
    def protocol(self) -> dict[str, Any]:
        value = self.raw.get("protocol", {})
        return value if isinstance(value, dict) else {}

    @property
    def method_contract(self) -> dict[str, Any]:
        value = self.raw.get("method_contract", {})
        return value if isinstance(value, dict) else {}

    @property
    def resource_limits(self) -> dict[str, Any]:
        value = self.raw.get("resource_limits", {})
        return value if isinstance(value, dict) else {}

    def validate_declaration(self) -> list[str]:
        """Return declaration errors without touching scientific inputs."""
        errors: list[str] = []
        version = self.raw.get("version")
        if version not in {"ma-usfa-tsc-server-manifest-v1", "ma-usfa-tsc-server-manifest-v2"}:
            errors.append("unsupported traffic manifest.version")
        if self.status not in ALLOWED_STATUS:
            errors.append(f"manifest.status must be one of {sorted(ALLOWED_STATUS)}")
        if not self.scenario_id:
            errors.append("manifest.scenario_id is empty")
        for key in (V2_CONTRACT if version == "ma-usfa-tsc-server-manifest-v2" else REQUIRED_CONTRACT):
            if self.method_contract.get(key) is not True:
                errors.append(f"method_contract.{key} must be true")
        protocol_required = (
            "protocol_id",
            "code_sha256",
            "data_manifest_sha256",
            "train_split_sha256",
            "validation_split_sha256",
            "training_seeds",
            "traffic_seeds",
        )
        if version == "ma-usfa-tsc-server-manifest-v1":
            protocol_required += ("test_split_sha256",)
        elif version == "ma-usfa-tsc-server-manifest-v2":
            task_protocol = self.raw.get("traffic_task_protocol", {})
            if not isinstance(task_protocol, dict) or any(
                    not task_protocol.get(key) for key in ("version", "path", "sha256")):
                errors.append("v2 requires a declared traffic_task_protocol file and hash")
        for key in protocol_required:
            value = self.protocol.get(key)
            if value in (None, "", [], {}):
                errors.append(f"protocol.{key} is not filled")
        for key in ("threads", "max_wall_seconds", "max_rss_gib", "min_free_disk_gib"):
            value = self.resource_limits.get(key)
            if not isinstance(value, (int, float)) or value <= 0:
                errors.append(f"resource_limits.{key} must be positive")
        if self.raw.get("test_accessed_before_formal_run") is not False:
            errors.append("test_accessed_before_formal_run must be false")
        return errors

    def require_stage(self, stage: str) -> None:
        """Raise if a command is not allowed by the manifest status."""
        allowed = {
            "preflight": ALLOWED_STATUS,
            "pilot": {"READY_FOR_PILOT", "READY_FOR_FORMAL"},
            "formal": {"READY_FOR_FORMAL"},
        }
        if stage not in allowed:
            raise ValueError(f"Unknown stage: {stage}")
        if self.status not in allowed[stage]:
            raise RuntimeError(
                f"Manifest status {self.status!r} cannot run stage {stage!r}; "
                f"allowed statuses are {sorted(allowed[stage])}"
            )


def load_manifest(path: str | Path) -> Manifest:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest is missing: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Manifest root must be an object")
    manifest = Manifest(manifest_path, raw)
    errors = manifest.validate_declaration()
    if errors:
        raise ValueError("Invalid manifest declaration:\n- " + "\n- ".join(errors))
    return manifest


def declared_file_paths(manifest: Manifest) -> tuple[Path, ...]:
    """Return declared file paths without searching for fallbacks."""
    files = manifest.raw.get("files", [])
    if not isinstance(files, list):
        raise ValueError("manifest.files must be a list")
    paths: list[Path] = []
    for item in files:
        if not isinstance(item, dict) or not str(item.get("path", "")).strip():
            raise ValueError("Every manifest file declaration needs a declared path")
        path = resolve_manifest_path(manifest, str(item["path"]))
        paths.append(path)
    return tuple(paths)


def resolve_manifest_path(manifest: Manifest, value: str | Path) -> Path:
    """Resolve a declared path relative to the manifest, without fallback search."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest.path.parent / path
    return path.resolve()

