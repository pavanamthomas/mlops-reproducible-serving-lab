"""Previous artifact in a linear local registry.

No canary, no traffic split. Schema mismatch still has to be checked.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactRecord:
    artifact_id: str
    model_version: str
    schema_version: str
    created_at: str
    path: str


def select_previous_artifact(
    records: list[ArtifactRecord],
    current_id: str,
) -> ArtifactRecord:
    """Return the record immediately before ``current_id`` in time order.

    Raises ValueError if the current id is unknown or is the oldest
    record (nothing to roll back to).
    """
    if not records:
        raise ValueError("registry is empty")
    ordered = sorted(records, key=lambda rec: (rec.created_at, rec.artifact_id))
    ids = [rec.artifact_id for rec in ordered]
    if current_id not in ids:
        raise ValueError(f"unknown artifact_id {current_id!r}")
    index = ids.index(current_id)
    if index == 0:
        raise ValueError(f"artifact {current_id!r} is the oldest; no previous artifact")
    return ordered[index - 1]
