"""Select the previous artifact in a linear local registry.

Problem: replacing a bad model requires naming the artifact to restore.
Assumptions: records are totally ordered by ``created_at`` then
``artifact_id``; there is no canary, no traffic split, and no
distributed lock.
Why this method: a previous-id lookup is the smallest rollback that can
be unit tested. A markdown note in ``docs/rollback.md`` states what this
does not do.
Alternative: MLflow model stages (Staging/Production). Not used; they
imply a process this laboratory does not run.
What can go wrong: rolling back to an artifact whose schema_version no
longer matches the API; treating this function as an incident manager.
Independent check: a three-record registry returns the middle id when
the current id is the latest.
Can conclude: which recorded id precedes the current one in this list.
Cannot conclude: that restoring that file will restore production
behaviour.
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
