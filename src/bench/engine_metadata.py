from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

import yaml


_MANTICORE_ENGINE_NAMES = {
    "columnar",
    "manticore",
    "manticore_all_default",
    "manticore_avgdl",
    "manticore_default_ranker",
    "manticore_full_default",
}


def compose_service_for_engine(engine: str) -> str:
    normalized = str(engine or "").strip().lower()
    return "manticore" if normalized in _MANTICORE_ENGINE_NAMES else normalized


def version_from_image(image: str) -> str:
    raw_reference = str(image or "").strip()
    reference, separator, _digest = raw_reference.partition("@")
    leaf = reference.rsplit("/", 1)[-1]
    if ":" not in leaf:
        return "unknown" if separator else "latest" if reference else "unknown"
    tag = leaf.rsplit(":", 1)[-1]
    if re.match(r"^v\d", tag):
        tag = tag[1:]
    return tag or "unknown"


def _find_project_root(start: str | Path | None = None) -> Path | None:
    origin = Path(start or Path.cwd()).resolve()
    if origin.is_file():
        origin = origin.parent
    module_root = Path(__file__).resolve().parents[2]
    candidates = [origin, *origin.parents, module_root, *module_root.parents]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "compose.yaml").is_file():
            return candidate
    return None


def _compose_image(project_root: Path, service: str) -> str:
    try:
        payload = yaml.safe_load((project_root / "compose.yaml").read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return ""
    services = payload.get("services", {}) if isinstance(payload, dict) else {}
    service_config = services.get(service, {}) if isinstance(services, dict) else {}
    if not isinstance(service_config, dict):
        return ""
    return str(service_config.get("image") or "").strip()


def _run_json_command(
    args: list[str],
    *,
    cwd: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> Any:
    try:
        result = runner(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        return json.loads((result.stdout or "").strip())
    except (json.JSONDecodeError, TypeError):
        return None


def _inspect_image_metadata(
    project_root: Path,
    image: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, str]:
    if not image:
        return {}
    inspected = _run_json_command(
        ["docker", "image", "inspect", image], cwd=project_root, runner=runner
    )
    if not isinstance(inspected, list) or not inspected or not isinstance(inspected[0], dict):
        return {}
    image_info = inspected[0]
    repo_digests = image_info.get("RepoDigests", [])
    digest = ""
    if isinstance(repo_digests, list):
        digest = next((str(value).strip() for value in repo_digests if str(value).strip()), "")
    return {
        "image_id": str(image_info.get("Id") or "").strip(),
        "image_digest": digest,
        "image_created_at": str(image_info.get("Created") or "").strip(),
    }


def _runtime_image_metadata(
    project_root: Path,
    service: str,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, str]:
    try:
        result = runner(
            ["docker", "compose", "ps", "-q", service],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if result.returncode != 0:
        return {}
    container_id = (result.stdout or "").strip().splitlines()
    if not container_id:
        return {}

    inspected = _run_json_command(
        ["docker", "inspect", container_id[0]], cwd=project_root, runner=runner
    )
    if not isinstance(inspected, list) or not inspected or not isinstance(inspected[0], dict):
        return {}
    container = inspected[0]
    config = container.get("Config", {}) if isinstance(container.get("Config"), dict) else {}
    image = str(config.get("Image") or "").strip()
    image_id = str(container.get("Image") or "").strip()
    inspected_image = _inspect_image_metadata(project_root, image_id, runner=runner)
    return {
        "image": image,
        "image_id": inspected_image.get("image_id") or image_id,
        "image_digest": inspected_image.get("image_digest", ""),
        "image_created_at": inspected_image.get("image_created_at", ""),
    }


def resolve_engine_artifact_metadata(
    engine: str,
    *,
    start: str | Path | None = None,
    inspect_runtime: bool = True,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, str]:
    service = compose_service_for_engine(engine)
    project_root = _find_project_root(start)
    compose_image = _compose_image(project_root, service) if project_root else ""
    runtime: dict[str, str] = {}
    if inspect_runtime and project_root and service:
        runtime = _runtime_image_metadata(project_root, service, runner=runner)

    image = runtime.get("image") or compose_image
    local_image: dict[str, str] = {}
    if inspect_runtime and project_root and image and not runtime.get("image_created_at"):
        local_image = _inspect_image_metadata(project_root, image, runner=runner)
    image_id = runtime.get("image_id") or local_image.get("image_id", "")
    image_digest = runtime.get("image_digest") or local_image.get("image_digest", "")
    image_created_at = runtime.get("image_created_at") or local_image.get("image_created_at", "")
    if not image_digest and "@" in image:
        image_digest = image
    source = "container" if runtime.get("image") else "compose" if compose_image else "unknown"
    return {
        "engine_version": version_from_image(image),
        "engine_image": image,
        "engine_image_id": image_id,
        "engine_image_digest": image_digest,
        "engine_image_created_at": image_created_at,
        "engine_version_source": source,
    }
