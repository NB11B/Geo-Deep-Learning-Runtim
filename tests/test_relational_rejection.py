import pytest
import torch

import geo_dl_runtime as geo_rt


def test_non_square_logits_rejection():
    geo_rt.require_stage("relational")
    # Non-square logits (M=2, P=4, Q=5) must raise RuntimeError / ValueError
    logits = torch.randn(2, 4, 5)
    with pytest.raises(RuntimeError):
        geo_rt.birkhoff_project(logits)


def test_non_finite_logits_rejection():
    geo_rt.require_stage("relational")
    logits = torch.randn(2, 4, 4)
    logits[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        geo_rt.birkhoff_project(logits)


def test_dimension_mismatch_mix_rejection():
    geo_rt.require_stage("relational")
    state = torch.randn(4, 4, 16) # G=4, P=4, D=16
    rel = torch.randn(3, 4, 4)     # M=3 (invalid, must be 1 or G=4)
    with pytest.raises(RuntimeError):
        geo_rt.relational_mix(state, rel)


def test_dimension_mismatch_read_rejection():
    geo_rt.require_stage("relational")
    state = torch.randn(4, 4, 16)
    read_weights = torch.randn(2, 5) # P=5 instead of P=4
    with pytest.raises(RuntimeError):
        geo_rt.relational_read(state, read_weights)


def test_invalid_stage_rejection():
    with pytest.raises(ValueError):
        geo_rt.require_stage("non_existent_stage")
