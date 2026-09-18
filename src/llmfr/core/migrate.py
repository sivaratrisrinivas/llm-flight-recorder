"""Reject unknown schema versions. v1 has no migrations."""

from __future__ import annotations

from typing import Any

from llmfr.core.version import SCHEMA_VERSION


class UnsupportedSchemaVersionError(ValueError):
    """Raised when a payload is not SCHEMA_VERSION 1.0.0."""


def migrate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Identity for v1 payloads. No other versions are accepted.

    This is a migration stub, not a migrator. When a future schema ships, replace
    this function with real upgrade steps and a new ADR.
    """
    version = payload.get("schema_version")
    if version == SCHEMA_VERSION:
        return payload
    raise UnsupportedSchemaVersionError(
        f"unsupported schema_version {version!r}; llmfr v1 only reads {SCHEMA_VERSION}"
    )
