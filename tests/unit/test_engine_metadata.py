from __future__ import annotations

import json
from pathlib import Path
import subprocess

from bench.engine_metadata import resolve_engine_artifact_metadata, version_from_image


def test_version_from_image_handles_tags_and_registry_ports() -> None:
    assert version_from_image("qdrant/qdrant:v1.19.1") == "1.19.1"
    assert version_from_image("registry.example:5000/team/search:2.4.0") == "2.4.0"
    assert version_from_image("search/image") == "latest"
    assert version_from_image("search/image@sha256:digest") == "unknown"
    assert version_from_image("") == "unknown"


def test_resolve_metadata_uses_compose_and_maps_columnar_to_manticore(tmp_path: Path) -> None:
    (tmp_path / "compose.yaml").write_text(
        """
services:
  manticore:
    image: manticoresearch/manticore:29.9.0
  qdrant:
    image: qdrant/qdrant:v1.19.1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    columnar = resolve_engine_artifact_metadata(
        "columnar", start=tmp_path, inspect_runtime=False
    )
    qdrant = resolve_engine_artifact_metadata("qdrant", start=tmp_path, inspect_runtime=False)

    assert columnar == {
        "engine_version": "29.9.0",
        "engine_image": "manticoresearch/manticore:29.9.0",
        "engine_image_id": "",
        "engine_image_digest": "",
        "engine_image_created_at": "",
        "engine_version_source": "compose",
    }
    assert qdrant["engine_version"] == "1.19.1"
    assert qdrant["engine_image"] == "qdrant/qdrant:v1.19.1"


def test_resolve_metadata_prefers_running_container_and_records_immutable_image(
    tmp_path: Path,
) -> None:
    (tmp_path / "compose.yaml").write_text(
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.1\n",
        encoding="utf-8",
    )

    def fake_runner(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if args == ["docker", "compose", "ps", "-q", "qdrant"]:
            stdout = "container-123\n"
        elif args == ["docker", "inspect", "container-123"]:
            stdout = json.dumps(
                [{"Config": {"Image": "qdrant/qdrant:v1.20.0"}, "Image": "sha256:image-id"}]
            )
        elif args == ["docker", "image", "inspect", "sha256:image-id"]:
            stdout = json.dumps(
                [{
                    "Id": "sha256:image-id",
                    "RepoDigests": ["qdrant/qdrant@sha256:registry-digest"],
                    "Created": "2026-09-03T13:10:26Z",
                }]
            )
        else:
            raise AssertionError(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    metadata = resolve_engine_artifact_metadata("qdrant", start=tmp_path, runner=fake_runner)

    assert metadata == {
        "engine_version": "1.20.0",
        "engine_image": "qdrant/qdrant:v1.20.0",
        "engine_image_id": "sha256:image-id",
        "engine_image_digest": "qdrant/qdrant@sha256:registry-digest",
        "engine_image_created_at": "2026-09-03T13:10:26Z",
        "engine_version_source": "container",
    }


def test_resolve_metadata_inspects_local_compose_image_without_running_container(
    tmp_path: Path,
) -> None:
    (tmp_path / "compose.yaml").write_text(
        "services:\n  qdrant:\n    image: qdrant/qdrant:v1.19.1\n",
        encoding="utf-8",
    )

    def fake_runner(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if args == ["docker", "compose", "ps", "-q", "qdrant"]:
            stdout = ""
        elif args == ["docker", "image", "inspect", "qdrant/qdrant:v1.19.1"]:
            stdout = json.dumps(
                [{
                    "Id": "sha256:image-id",
                    "RepoDigests": ["qdrant/qdrant@sha256:registry-digest"],
                    "Created": "2026-09-03T13:10:26Z",
                }]
            )
        else:
            raise AssertionError(f"unexpected command: {args}")
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    metadata = resolve_engine_artifact_metadata("qdrant", start=tmp_path, runner=fake_runner)

    assert metadata["engine_version_source"] == "compose"
    assert metadata["engine_image_id"] == "sha256:image-id"
    assert metadata["engine_image_created_at"] == "2026-09-03T13:10:26Z"
