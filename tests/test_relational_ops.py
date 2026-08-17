import math
import pytest
import torch

import geo_dl_runtime as geo_rt


def is_cuda_available():
    return torch.cuda.is_available() and getattr(geo_rt, "GEO_CUDA_AVAILABLE", False)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if is_cuda_available() else []))
def test_birkhoff_project_forward_backward(device):
    geo_rt.require_stage("relational")
    M, P = 4, 4
    kappa = math.log((1.0 - (P - 1) * 1e-3) / 1e-3)

    logits = torch.zeros(M, P, P, device=device, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for m in range(M):
            logits[m].diagonal().fill_(kappa)

    rel = geo_rt.birkhoff_project(logits, iterations=20, epsilon=1e-5)
    assert rel.shape == (M, P, P)
    assert torch.allclose(rel.sum(dim=-1), torch.ones(M, P, device=device), atol=1e-3)
    assert torch.allclose(rel.sum(dim=-2), torch.ones(M, P, device=device), atol=1e-3)

    loss = (rel ** 2).sum()
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == (M, P, P)
    assert torch.isfinite(logits.grad).all()

    telemetry = geo_rt.get_last_relational_telemetry()
    assert telemetry["fallback_status"] == "NONE"


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if is_cuda_available() else []))
def test_relational_identity_gate(device):
    geo_rt.require_stage("relational")
    M, P = 2, 4
    projected = torch.eye(P, device=device).unsqueeze(0).repeat(M, 1, 1).requires_grad_(True)
    gate = torch.tensor([0.0, 1.0], device=device, requires_grad=True)

    effective = geo_rt.relational_identity_gate(projected, gate)
    assert effective.shape == (M, P, P)

    # Gate 0 = 0 -> output is Identity
    assert torch.allclose(effective[0], torch.eye(P, device=device), atol=1e-5)
    # Gate 1 = 1 -> output is projected
    assert torch.allclose(effective[1], projected[1], atol=1e-5)

    loss = (effective ** 2).sum()
    loss.backward()

    assert projected.grad is not None
    assert gate.grad is not None


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if is_cuda_available() else []))
def test_relational_mix(device):
    geo_rt.require_stage("relational")
    G, P, D = 4, 4, 16
    M = G
    state = torch.randn(G, P, D, device=device, requires_grad=True)
    relationship = torch.eye(P, device=device).unsqueeze(0).repeat(M, 1, 1).requires_grad_(True)

    output = geo_rt.relational_mix(state, relationship)
    assert output.shape == (G, P, D)
    # With identity relationship, output should equal state
    assert torch.allclose(output, state, atol=1e-5)

    loss = (output ** 2).sum()
    loss.backward()

    assert state.grad is not None
    assert relationship.grad is not None


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if is_cuda_available() else []))
def test_relational_read(device):
    geo_rt.require_stage("relational")
    G, P, D = 4, 4, 16
    weight_count = 1
    state = torch.randn(G, P, D, device=device, requires_grad=True)
    read_weights = torch.softmax(torch.randn(weight_count, P, device=device), dim=-1).requires_grad_(True)

    read_state = geo_rt.relational_read(state, read_weights)
    assert read_state.shape == (G, D)

    loss = (read_state ** 2).sum()
    loss.backward()

    assert state.grad is not None
    assert read_weights.grad is not None


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if is_cuda_available() else []))
def test_relational_write_add(device):
    geo_rt.require_stage("relational")
    G, P, D = 4, 4, 16
    weight_count = 1
    transported = torch.randn(G, P, D, device=device, requires_grad=True)
    source = torch.randn(G, D, device=device, requires_grad=True)
    write_weights = torch.softmax(torch.randn(weight_count, P, device=device), dim=-1).requires_grad_(True)
    source_scale = torch.tensor([0.5], device=device, requires_grad=True)

    output = geo_rt.relational_write_add(transported, source, write_weights, source_scale)
    assert output.shape == (G, P, D)

    loss = (output ** 2).sum()
    loss.backward()

    assert transported.grad is not None
    assert source.grad is not None
    assert write_weights.grad is not None
    assert source_scale.grad is not None
