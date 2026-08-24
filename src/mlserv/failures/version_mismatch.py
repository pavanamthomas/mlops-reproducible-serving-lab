"""Hard-fail when model_version or schema_version disagree.

An API can still return HTTP 200 if versions are not checked.
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
