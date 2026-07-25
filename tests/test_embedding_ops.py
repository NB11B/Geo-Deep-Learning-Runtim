from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

runtime = pytest.importorskip("geo_dl_runtime")
if not runtime.native_available():
    pytest.skip("native GEO runtime core extension is not built", allow_module_level=True)
if not runtime.EMBEDDING_CAPABILITIES.issubset(runtime.GEO_CAPABILITIES):
    pytest.skip("native GEO embedding extension is not built", allow_module_level=True)


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available() and runtime.GEO_CUDA_AVAILABLE:
        result.append(torch.device("cuda"))
    return result


@pytest.mark.parametrize("device", devices())
def test_embedding_forward_and_repeated_index_vjp(device: torch.device):
    torch.manual_seed(51)
    vocabulary, dimension = 11, 7
    weight = torch.randn(
        vocabulary, dimension, device=device, dtype=torch.float32, requires_grad=True
    )
    weight_ref = weight.detach().clone().requires_grad_(True)
    indices = torch.tensor(
        [[2, 5, 2, 0], [7, 5, 2, 7]],
        device=device,
        dtype=torch.int64,
    )
    grad = torch.randn(2, 4, dimension, device=device)

    output = runtime.embedding(indices, weight)
    reference = F.embedding(indices, weight_ref)
    torch.testing.assert_close(output, reference)

    output.backward(grad)
    reference.backward(grad)
    tolerance = 1e-6 if device.type == "cpu" else 2e-5
    torch.testing.assert_close(
        weight.grad, weight_ref.grad, rtol=tolerance, atol=tolerance
    )
    assert torch.count_nonzero(weight.grad[1]) == 0
    assert torch.count_nonzero(weight.grad[2]) > 0


def test_model_stage_is_declared():
    runtime.require_stage("model")
    runtime.require_stage("transformer")
    assert runtime.MODEL_STAGE_CAPABILITIES.issubset(
        runtime.native_capabilities().operations
    )


def test_embedding_rejects_non_int64_indices():
    indices = torch.ones(2, 3, dtype=torch.int32)
    weight = torch.ones(5, 4)
    with pytest.raises(RuntimeError, match="int64"):
        runtime.embedding(indices, weight)


def test_embedding_rejects_device_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for a device mismatch")
    indices = torch.ones(2, 3, dtype=torch.int64, device="cuda")
    weight = torch.ones(5, 4, device="cpu")
    with pytest.raises(RuntimeError, match="share a device"):
        runtime.embedding(indices, weight)
