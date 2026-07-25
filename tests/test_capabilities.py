from __future__ import annotations

import pytest

from geo_dl_runtime.capabilities import (
    MODEL_STAGE_CAPABILITIES,
    STAGES,
    TRAINING_STAGE_CAPABILITIES,
    RuntimeCapabilities,
)


def test_linear_stage_is_supported_by_linear_only():
    available = RuntimeCapabilities.from_iterable(["linear"])
    assert available.supports(STAGES["linear"])
    assert not available.supports(STAGES["core"])


def test_missing_stage_operations_are_reported():
    available = RuntimeCapabilities.from_iterable(["linear", "add"])
    with pytest.raises(RuntimeError, match="mul, rms_norm, scale"):
        available.require(STAGES["core"], stage="core")


def test_model_stage_requires_embedding():
    available = RuntimeCapabilities.from_iterable(
        MODEL_STAGE_CAPABILITIES - {"embedding"}
    )
    with pytest.raises(RuntimeError, match="embedding"):
        available.require(STAGES["model"], stage="model")


def test_training_stage_requires_optimizer():
    available = RuntimeCapabilities.from_iterable(
        TRAINING_STAGE_CAPABILITIES - {"adamw"}
    )
    with pytest.raises(RuntimeError, match="adamw"):
        available.require(STAGES["training"], stage="training")


def test_complete_training_surface_is_supported():
    available = RuntimeCapabilities.from_iterable(
        TRAINING_STAGE_CAPABILITIES
    )
    assert available.supports(STAGES["model"])
    assert available.supports(STAGES["transformer"])
    assert available.supports(STAGES["training"])
