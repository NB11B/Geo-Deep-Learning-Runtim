from __future__ import annotations

import pytest
import torch

runtime = pytest.importorskip("geo_dl_runtime")


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available() and runtime.native_available():
        result.append(torch.device("cuda"))
    return result


@pytest.mark.parametrize("device", devices())
def test_add_forward_and_vjp(device: torch.device):
    torch.manual_seed(1)
    a = torch.randn(2, 3, 5, device=device, requires_grad=True)
    b = torch.randn(2, 3, 5, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    grad = torch.randn_like(a)

    y = runtime.add(a, b)
    y_ref = a_ref + b_ref
    torch.testing.assert_close(y, y_ref)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(a.grad, a_ref.grad)
    torch.testing.assert_close(b.grad, b_ref.grad)


@pytest.mark.parametrize("device", devices())
def test_mul_forward_and_vjp(device: torch.device):
    torch.manual_seed(2)
    a = torch.randn(4, 7, device=device, requires_grad=True)
    b = torch.randn(4, 7, device=device, requires_grad=True)
    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    grad = torch.randn_like(a)

    y = runtime.mul(a, b)
    y_ref = a_ref * b_ref
    torch.testing.assert_close(y, y_ref)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(a.grad, a_ref.grad)
    torch.testing.assert_close(b.grad, b_ref.grad)


@pytest.mark.parametrize("device", devices())
def test_scale_forward_and_vjp(device: torch.device):
    torch.manual_seed(3)
    scalar = 0.375
    x = torch.randn(3, 11, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x)

    y = runtime.scale(x, scalar)
    y_ref = x_ref * scalar
    torch.testing.assert_close(y, y_ref)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad)


@pytest.mark.parametrize("device", devices())
def test_rms_norm_forward_and_vjp(device: torch.device):
    torch.manual_seed(4)
    epsilon = 1e-6
    x = torch.randn(2, 3, 16, device=device, requires_grad=True)
    weight = torch.randn(16, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x)

    y = runtime.rms_norm(x, weight, epsilon)
    inv = torch.rsqrt(x_ref.square().mean(dim=-1, keepdim=True) + epsilon)
    y_ref = x_ref * inv * weight_ref
    torch.testing.assert_close(y, y_ref, rtol=2e-5, atol=2e-5)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-5, atol=3e-5)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=3e-5, atol=3e-5)


def test_core_stage_is_declared_when_native_extension_is_built():
    if not runtime.native_available():
        pytest.skip("native extension is not built")
    runtime.require_stage("core")
    assert runtime.CORE_CAPABILITIES.issubset(runtime.native_capabilities().available)


def test_binary_ops_reject_shape_mismatch():
    if not runtime.native_available():
        pytest.skip("native extension is not built")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        runtime.add(torch.ones(2, 3), torch.ones(2, 4))
