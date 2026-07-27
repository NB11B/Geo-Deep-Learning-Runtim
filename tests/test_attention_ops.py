from __future__ import annotations

import math

import pytest
import torch

runtime = pytest.importorskip("geo_dl_runtime")
if not runtime.native_available():
    pytest.skip("native GEO runtime core extension is not built", allow_module_level=True)
if not runtime.ATTENTION_CAPABILITIES.issubset(runtime.GEO_CAPABILITIES):
    pytest.skip("native GEO attention extension is not built", allow_module_level=True)


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available() and runtime.GEO_CUDA_AVAILABLE:
        result.append(torch.device("cuda"))
    return result


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    tokens = q.shape[-2]
    mask = torch.triu(
        torch.ones((tokens, tokens), device=q.device, dtype=torch.bool),
        diagonal=1,
    )
    scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    return torch.matmul(torch.softmax(scores, dim=-1), v)


@pytest.mark.parametrize("device", devices())
def test_causal_attention_forward_and_vjp(device: torch.device):
    torch.manual_seed(31)
    batch, tokens, heads, head_dim = 2, 5, 3, 4

    q_base = torch.randn(
        batch, tokens, heads, head_dim,
        device=device, dtype=torch.float32, requires_grad=True,
    )
    k_base = torch.randn(
        batch, tokens, heads, head_dim,
        device=device, dtype=torch.float32, requires_grad=True,
    )
    v_base = torch.randn(
        batch, tokens, heads, head_dim,
        device=device, dtype=torch.float32, requires_grad=True,
    )
    q_ref_base = q_base.detach().clone().requires_grad_(True)
    k_ref_base = k_base.detach().clone().requires_grad_(True)
    v_ref_base = v_base.detach().clone().requires_grad_(True)

    q = q_base.transpose(1, 2)
    k = k_base.transpose(1, 2)
    v = v_base.transpose(1, 2)
    q_ref = q_ref_base.transpose(1, 2)
    k_ref = k_ref_base.transpose(1, 2)
    v_ref = v_ref_base.transpose(1, 2)
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert not v.is_contiguous()

    grad = torch.randn(batch, heads, tokens, head_dim, device=device)
    output = runtime.causal_attention(q, k, v)
    reference = reference_attention(q_ref, k_ref, v_ref)

    forward_tolerance = 8e-5 if device.type == "cpu" else 3e-4
    gradient_tolerance = 1.2e-4 if device.type == "cpu" else 8e-4
    torch.testing.assert_close(
        output, reference,
        rtol=forward_tolerance,
        atol=forward_tolerance,
    )

    output.backward(grad)
    reference.backward(grad)
    torch.testing.assert_close(
        q_base.grad, q_ref_base.grad,
        rtol=gradient_tolerance,
        atol=gradient_tolerance,
    )
    torch.testing.assert_close(
        k_base.grad, k_ref_base.grad,
        rtol=gradient_tolerance,
        atol=gradient_tolerance,
    )
    torch.testing.assert_close(
        v_base.grad, v_ref_base.grad,
        rtol=gradient_tolerance,
        atol=gradient_tolerance,
    )


@pytest.mark.parametrize("device", devices())
def test_first_query_is_exactly_first_value(device: torch.device):
    torch.manual_seed(32)
    q = torch.randn(1, 2, 4, 6, device=device)
    k = torch.randn(1, 2, 4, 6, device=device)
    v = torch.randn(1, 2, 4, 6, device=device)
    output = runtime.causal_attention(q, k, v)
    torch.testing.assert_close(output[..., 0, :], v[..., 0, :])


def test_attention_stage_is_declared():
    runtime.require_stage("attention")
    assert runtime.ATTENTION_STAGE_CAPABILITIES.issubset(
        runtime.native_capabilities().operations
    )


def test_attention_rejects_shape_mismatch():
    q = torch.ones(1, 2, 4, 8)
    k = torch.ones(1, 2, 5, 8)
    v = torch.ones(1, 2, 4, 8)
    with pytest.raises(RuntimeError, match="shapes must match"):
        runtime.causal_attention(q, k, v)


def test_attention_rejects_non_float32():
    q = torch.ones(1, 2, 4, 8, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="float32"):
        runtime.causal_attention(q, q, q)
