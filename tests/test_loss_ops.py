from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

runtime = pytest.importorskip("geo_dl_runtime")
if not runtime.native_available():
    pytest.skip("native GEO runtime core extension is not built", allow_module_level=True)
if not runtime.LOSS_CAPABILITIES.issubset(runtime.GEO_CAPABILITIES):
    pytest.skip("native GEO loss extension is not built", allow_module_level=True)


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available() and runtime.GEO_CUDA_AVAILABLE:
        result.append(torch.device("cuda"))
    return result


@pytest.mark.parametrize("device", devices())
def test_cross_entropy_forward_and_vjp(device: torch.device):
    torch.manual_seed(41)
    rows, classes = 13, 17
    logits = torch.randn(
        rows, classes, device=device, dtype=torch.float32, requires_grad=True
    )
    logits_ref = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, classes, (rows,), device=device, dtype=torch.int64)
    targets[2] = -1
    targets[9] = -1
    upstream = torch.tensor(0.7, device=device)

    loss = runtime.cross_entropy(logits, targets, ignore_index=-1)
    reference = F.cross_entropy(logits_ref, targets, ignore_index=-1)

    forward_tolerance = 8e-5 if device.type == "cpu" else 5e-4
    gradient_tolerance = 1e-4 if device.type == "cpu" else 5e-4
    torch.testing.assert_close(
        loss, reference,
        rtol=forward_tolerance,
        atol=forward_tolerance,
    )

    loss.backward(upstream)
    reference.backward(upstream)
    torch.testing.assert_close(
        logits.grad, logits_ref.grad,
        rtol=gradient_tolerance,
        atol=gradient_tolerance,
    )
    torch.testing.assert_close(logits.grad[2], torch.zeros_like(logits.grad[2]))
    torch.testing.assert_close(logits.grad[9], torch.zeros_like(logits.grad[9]))


@pytest.mark.parametrize("device", devices())
def test_all_ignored_matches_reference(device: torch.device):
    torch.manual_seed(42)
    logits = torch.randn(4, 7, device=device, requires_grad=True)
    logits_ref = logits.detach().clone().requires_grad_(True)
    targets = torch.full((4,), -1, device=device, dtype=torch.int64)

    loss = runtime.cross_entropy(logits, targets, ignore_index=-1)
    reference = F.cross_entropy(logits_ref, targets, ignore_index=-1)
    assert torch.isnan(loss)
    assert torch.isnan(reference)

    loss.backward()
    reference.backward()
    torch.testing.assert_close(logits.grad, logits_ref.grad)
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits.grad))


def test_transformer_stage_is_declared():
    runtime.require_stage("loss")
    runtime.require_stage("transformer")
    assert runtime.TRANSFORMER_CAPABILITIES.issubset(
        runtime.native_capabilities().operations
    )


def test_cross_entropy_rejects_non_int64_targets():
    logits = torch.ones(3, 5)
    targets = torch.ones(3, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="int64"):
        runtime.cross_entropy(logits, targets)


def test_cross_entropy_rejects_row_mismatch():
    logits = torch.ones(3, 5)
    targets = torch.ones(4, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="counts must match"):
        runtime.cross_entropy(logits, targets)
