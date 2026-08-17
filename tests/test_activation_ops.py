from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

runtime = pytest.importorskip("geo_dl_runtime")
if not runtime.native_available():
    pytest.skip("native GEO runtime extension is not built", allow_module_level=True)


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available() and runtime.GEO_CUDA_AVAILABLE:
        result.append(torch.device("cuda"))
    return result


@pytest.mark.parametrize("device", devices())
def test_exact_gelu_forward_and_vjp(device: torch.device):
    torch.manual_seed(11)
    x = torch.randn(4, 9, device=device, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x)

    y = runtime.gelu(x)
    y_ref = F.gelu(x_ref, approximate="none")

    torch.testing.assert_close(y, y_ref, rtol=3e-5, atol=3e-5)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("device", devices())
def test_fused_silu_mul_forward_and_vjp(device: torch.device):
    torch.manual_seed(12)
    gate = torch.randn(2, 3, 17, device=device, dtype=torch.float32, requires_grad=True)
    up = torch.randn(2, 3, 17, device=device, dtype=torch.float32, requires_grad=True)
    gate_ref = gate.detach().clone().requires_grad_(True)
    up_ref = up.detach().clone().requires_grad_(True)
    grad = torch.randn_like(gate)

    y = runtime.silu_mul(gate, up)
    y_ref = F.silu(gate_ref) * up_ref

    torch.testing.assert_close(y, y_ref, rtol=3e-5, atol=3e-5)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(gate.grad, gate_ref.grad, rtol=4e-5, atol=4e-5)
    torch.testing.assert_close(up.grad, up_ref.grad, rtol=4e-5, atol=4e-5)


def test_activation_stage_is_declared():
    runtime.require_stage("activation")
    assert runtime.ACTIVATION_STAGE_CAPABILITIES.issubset(
        runtime.native_capabilities().operations
    )


def test_silu_mul_rejects_shape_mismatch():
    with pytest.raises(RuntimeError, match="shape mismatch"):
        runtime.silu_mul(torch.ones(2, 3), torch.ones(2, 4))
