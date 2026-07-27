from __future__ import annotations

import pytest
import torch

runtime = pytest.importorskip("geo_dl_runtime")
if not runtime.native_available():
    pytest.skip("native GEO runtime core extension is not built", allow_module_level=True)
if not runtime.POSITION_CAPABILITIES.issubset(runtime.GEO_CAPABILITIES):
    pytest.skip("native GEO RoPE extension is not built", allow_module_level=True)


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available() and runtime.GEO_CUDA_AVAILABLE:
        result.append(torch.device("cuda"))
    return result


def reference_tables(seq_len: int, head_dim: int, theta: float, device: torch.device):
    indices = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inverse = torch.pow(torch.tensor(theta, device=device), -indices / head_dim)
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    frequencies = torch.outer(positions, inverse)
    return torch.cos(frequencies), torch.sin(frequencies)


def reference_apply(x: torch.Tensor, cos_table: torch.Tensor, sin_table: torch.Tensor):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cosine = cos_table[: x.shape[-2]][None, None, :, :]
    sine = sin_table[: x.shape[-2]][None, None, :, :]
    return torch.cat((x1 * cosine - x2 * sine, x2 * cosine + x1 * sine), dim=-1)


@pytest.mark.parametrize("device", devices())
def test_rope_table_build_matches_reference(device: torch.device):
    seq_len = 19
    head_dim = 16
    theta = 10000.0
    cosine, sine = runtime.build_rope(seq_len, head_dim, theta, device)
    cosine_ref, sine_ref = reference_tables(seq_len, head_dim, theta, device)

    assert cosine.shape == (seq_len, head_dim // 2)
    assert sine.shape == (seq_len, head_dim // 2)
    assert cosine.device.type == device.type
    assert sine.device.type == device.type
    torch.testing.assert_close(cosine, cosine_ref, rtol=4e-5, atol=4e-5)
    torch.testing.assert_close(sine, sine_ref, rtol=4e-5, atol=4e-5)


@pytest.mark.parametrize("device", devices())
def test_rope_apply_forward_and_vjp_on_attention_layout(device: torch.device):
    torch.manual_seed(21)
    batch, tokens, heads, head_dim = 2, 7, 3, 8
    theta = 10000.0
    cosine, sine = runtime.build_rope(16, head_dim, theta, device)

    base = torch.randn(
        batch,
        tokens,
        heads,
        head_dim,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    base_ref = base.detach().clone().requires_grad_(True)
    x = base.transpose(1, 2)
    x_ref = base_ref.transpose(1, 2)
    assert not x.is_contiguous()
    grad = torch.randn(batch, heads, tokens, head_dim, device=device)

    y = runtime.apply_rope(x, cosine, sine)
    y_ref = reference_apply(x_ref, cosine, sine)

    torch.testing.assert_close(y, y_ref, rtol=4e-5, atol=4e-5)
    y.backward(grad)
    y_ref.backward(grad)
    torch.testing.assert_close(base.grad, base_ref.grad, rtol=5e-5, atol=5e-5)


def test_position_stage_is_declared():
    runtime.require_stage("position")
    assert runtime.POSITION_STAGE_CAPABILITIES.issubset(
        runtime.native_capabilities().operations
    )


def test_rope_rejects_odd_head_dimension():
    with pytest.raises(RuntimeError, match="even"):
        runtime.build_rope(8, 7, 10000.0, torch.device("cpu"))


def test_rope_rejects_short_table():
    x = torch.ones(1, 1, 4, 8)
    cosine, sine = runtime.build_rope(3, 8, 10000.0, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="shorter"):
        runtime.apply_rope(x, cosine, sine)
