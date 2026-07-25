from __future__ import annotations

import math

import pytest
import torch

runtime = pytest.importorskip("geo_dl_runtime")
if not runtime.native_available():
    pytest.skip("native GEO runtime core extension is not built", allow_module_level=True)
if not runtime.OPTIMIZER_CAPABILITIES.issubset(runtime.GEO_CAPABILITIES):
    pytest.skip("native GEO optimizer extension is not built", allow_module_level=True)


def devices():
    result = [torch.device("cpu")]
    if torch.cuda.is_available() and runtime.GEO_CUDA_AVAILABLE:
        result.append(torch.device("cuda"))
    return result


def global_clip_scale(gradients: list[torch.Tensor], max_norm: float) -> float:
    sum_square = sum(float(torch.sum(gradient.square()).item()) for gradient in gradients)
    return min(1.0, max_norm / (math.sqrt(sum_square) + 1e-6))


@pytest.mark.parametrize("device", devices())
def test_geo_adamw_matches_reference_for_multiple_steps(device: torch.device):
    torch.manual_seed(61)
    learning_rate = 0.0125
    betas = (0.8, 0.93)
    epsilon = 1e-6
    weight_decay = 0.07
    max_grad_norm = 0.65

    parameters = [
        torch.nn.Parameter(torch.randn(3, 5, device=device)),
        torch.nn.Parameter(torch.randn(7, device=device)),
    ]
    reference_parameters = [
        torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters
    ]

    optimizer = runtime.GeoAdamW(
        parameters,
        learning_rate=learning_rate,
        betas=betas,
        epsilon=epsilon,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
    )
    reference_optimizer = torch.optim.AdamW(
        reference_parameters,
        lr=learning_rate,
        betas=betas,
        eps=epsilon,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )

    expected_scales: list[float] = []
    for step in range(2):
        torch.manual_seed(100 + step)
        gradients = [torch.randn_like(parameter) for parameter in parameters]
        expected_scales.append(global_clip_scale(gradients, max_grad_norm))

        for parameter, reference_parameter, gradient in zip(
            parameters, reference_parameters, gradients, strict=True
        ):
            parameter.grad = gradient.clone()
            reference_parameter.grad = gradient.clone()

        clip_scale = optimizer.step()
        reference_norm = torch.nn.utils.clip_grad_norm_(
            reference_parameters, max_grad_norm
        )
        reference_optimizer.step()

        assert clip_scale is not None
        assert float(reference_norm.item()) > max_grad_norm
        torch.testing.assert_close(
            clip_scale,
            torch.tensor(expected_scales[-1], device=device),
            rtol=2e-5,
            atol=2e-6,
        )

        parameter_tolerance = 3e-6 if device.type == "cpu" else 8e-5
        for parameter, reference_parameter in zip(
            parameters, reference_parameters, strict=True
        ):
            torch.testing.assert_close(
                parameter,
                reference_parameter,
                rtol=parameter_tolerance,
                atol=parameter_tolerance,
            )

        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        assert all(parameter.grad is None for parameter in parameters)

    state = optimizer.state_dict()
    assert state["step"] == 2
    assert len(state["states"]) == len(parameters)
    state_tolerance = 3e-6 if device.type == "cpu" else 8e-5
    for parameter, geo_state in zip(parameters, state["states"], strict=True):
        reference_state = reference_optimizer.state[
            reference_parameters[parameters.index(parameter)]
        ]
        torch.testing.assert_close(
            geo_state["first_moment"],
            reference_state["exp_avg"],
            rtol=state_tolerance,
            atol=state_tolerance,
        )
        torch.testing.assert_close(
            geo_state["second_moment"],
            reference_state["exp_avg_sq"],
            rtol=state_tolerance,
            atol=state_tolerance,
        )


@pytest.mark.parametrize("device", devices())
def test_geo_adamw_without_gradients_is_a_noop(device: torch.device):
    parameter = torch.nn.Parameter(torch.randn(4, device=device))
    original = parameter.detach().clone()
    optimizer = runtime.GeoAdamW([parameter])
    assert optimizer.step() is None
    assert optimizer.step_count == 0
    torch.testing.assert_close(parameter, original)


def test_training_stage_is_declared():
    runtime.require_stage("training")
    assert runtime.TRAINING_STAGE_CAPABILITIES.issubset(
        runtime.native_capabilities().operations
    )


def test_geo_adamw_rejects_non_float32_parameters():
    parameter = torch.nn.Parameter(torch.ones(3, dtype=torch.float64))
    optimizer = runtime.GeoAdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="float32"):
        optimizer.step()
