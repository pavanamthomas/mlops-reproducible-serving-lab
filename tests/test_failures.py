"""Failure experiments: drift, schema, versions, rollback, retention."""

from __future__ import annotations

import numpy as np
import pytest

from mlserv.failures.concept_drift import score_on_shifted_concept
from mlserv.failures.data_drift import ks_feature, population_stability_index, shift_income
from mlserv.failures.reproducibility import REQUIRED_RETENTION, missing_retention
from mlserv.failures.rollback import ArtifactRecord, select_previous_artifact
from mlserv.failures.schema_drift import add_column, drop_column, schema_drift_report
from mlserv.failures.version_mismatch import VersionMismatchError, VersionPair, check_versions


def test_psi_near_zero_on_identical_samples():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=400)
    psi = population_stability_index(x, x.copy())
    assert psi < 0.02


def test_psi_and_ks_rise_under_mean_shift():
    rng = np.random.default_rng(2026)
    ref = rng.normal(size=400)
    cur = ref + 1.5
    assert population_stability_index(ref, cur) > 0.2
    stat, _p = ks_feature(ref, cur)
    assert stat > 0.2


def test_income_shift_helper_changes_only_income(train_result):
    shifted = shift_income(train_result.X_val, log_shift=0.5)
    np.testing.assert_allclose(
        shifted["age"], train_result.X_val["age"].to_numpy(), atol=0.0
    )
    assert not np.allclose(shifted["income"], train_result.X_val["income"].to_numpy())


def test_concept_shift_lowers_accuracy_on_this_dgp(train_result):
    shifted = score_on_shifted_concept(
        train_result.pipeline, n=500, seed=2027, coef_credit=-2.4
    )
    assert shifted.accuracy < train_result.metrics.accuracy - 0.03


def test_schema_drift_missing_and_new_fields(train_result):
    missing = schema_drift_report(drop_column(train_result.X_val, "segment"))
    extra = schema_drift_report(add_column(train_result.X_val, "ip", "0.0.0.0"))
    assert missing.ok is False
    assert "segment" in missing.missing
    assert extra.ok is False
    assert "ip" in extra.unexpected


def test_version_mismatch_raises():
    artifact = VersionPair(model_version="1.0.0", schema_version="1.0")
    check_versions(artifact, VersionPair(model_version="1.0.0", schema_version="1.0"))
    with pytest.raises(VersionMismatchError, match="schema_version"):
        check_versions(artifact, VersionPair(model_version="1.0.0", schema_version="2.0"))
    with pytest.raises(VersionMismatchError, match="model_version"):
        check_versions(artifact, VersionPair(model_version="9.0.0", schema_version="1.0"))


def test_select_previous_artifact():
    records = [
        ArtifactRecord("a2", "1.1.0", "1.0", "2026-03-01T00:00:00Z", "p2"),
        ArtifactRecord("a0", "0.9.0", "1.0", "2026-01-01T00:00:00Z", "p0"),
        ArtifactRecord("a1", "1.0.0", "1.0", "2026-02-01T00:00:00Z", "p1"),
    ]
    previous = select_previous_artifact(records, "a2")
    assert previous.artifact_id == "a1"
    with pytest.raises(ValueError, match="oldest"):
        select_previous_artifact(records, "a0")
    with pytest.raises(ValueError, match="unknown"):
        select_previous_artifact(records, "nope")
    with pytest.raises(ValueError, match="empty"):
        select_previous_artifact([], "a1")


def test_missing_retention_lists_absent_keys():
    provided = {"seed", "python_version"}
    missing = missing_retention(provided)
    assert "fitted_pipeline" in missing
    assert "schema_version" in missing
    assert "seed" not in missing
    assert set(REQUIRED_RETENTION) - set(missing) == provided
