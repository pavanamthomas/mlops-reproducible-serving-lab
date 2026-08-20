"""Reject serving when model_version or schema_version do not match.

Problem: an API process can load yesterday's Pipeline while today's
request contract, or the reverse, and still return HTTP 200 if versions
are not checked.
Assumptions: versions are opaque strings recorded on the artifact;
mismatch is a hard error, not a warning.
Why this method: equality of two recorded strings is the smallest check
that makes rollback and replacement explicit.
Alternative: embedding a content hash of the Pydantic model. Not
implemented.
What can go wrong: a new schema that happens to use the same version
string; a model_version bump with no schema change that is still
incompatible because preprocessing changed.
Independent check: ``tests/test_failures.py`` expects ValueError on a
swapped pair.
Can conclude: these two version strings are not equal.
Cannot conclude: semantic compatibility of two artifacts that share a
schema_version.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VersionPair:
    model_version: str
    schema_version: str


class VersionMismatchError(ValueError):
    """Artifact versions do not match the expected serving contract."""


def check_versions(
    artifact: VersionPair,
    expected: VersionPair,
) -> None:
    """Raise VersionMismatchError unless both strings match."""
    if artifact.schema_version != expected.schema_version:
        raise VersionMismatchError(
            f"schema_version artifact={artifact.schema_version!r} "
            f"expected={expected.schema_version!r}"
        )
    if artifact.model_version != expected.model_version:
        raise VersionMismatchError(
            f"model_version artifact={artifact.model_version!r} "
            f"expected={expected.model_version!r}"
        )
