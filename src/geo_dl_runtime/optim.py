from __future__ import annotations

from collections.abc import Iterable

import torch

from . import _optimizer


class GeoAdamW:
    """AdamW whose numerical update and global norm clipping are GEO-owned."""

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
        if not self.parameters:
            raise ValueError("GeoAdamW requires at least one trainable parameter")
        if learning_rate < 0:
            raise ValueError("learning_rate must be nonnegative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if weight_decay < 0 or max_grad_norm < 0:
            raise ValueError("weight_decay and max_grad_norm must be nonnegative")

        self.learning_rate = float(learning_rate)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = float(max_grad_norm)
        self.step_count = 0
        self._state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _state_for(self, parameter: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = id(parameter)
        state = self._state.get(key)
        if state is None:
            state = (torch.zeros_like(parameter), torch.zeros_like(parameter))
            self._state[key] = state
        return state

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        if not set_to_none:
            raise ValueError("GeoAdamW supports only set_to_none=True")
        for parameter in self.parameters:
            parameter.grad = None

    @torch.no_grad()
    def step(self) -> torch.Tensor | None:
        active_parameters: list[torch.Tensor] = []
        gradients: list[torch.Tensor] = []
        first_moments: list[torch.Tensor] = []
        second_moments: list[torch.Tensor] = []

        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise RuntimeError("GeoAdamW does not support sparse gradients")
            if parameter.dtype != torch.float32 or gradient.dtype != torch.float32:
                raise RuntimeError("GeoAdamW currently requires float32 parameters and gradients")
            first, second = self._state_for(parameter)
            active_parameters.append(parameter)
            gradients.append(gradient.contiguous())
            first_moments.append(first)
            second_moments.append(second)

        if not active_parameters:
            return None

        self.step_count += 1
        clip_scale = _optimizer.adamw_step(
            active_parameters,
            gradients,
            first_moments,
            second_moments,
            self.step_count,
            self.learning_rate,
            self.beta1,
            self.beta2,
            self.epsilon,
            self.weight_decay,
            self.max_grad_norm,
        )
        for parameter in active_parameters:
            torch.autograd.graph.increment_version(parameter)
        return clip_scale

    def state_dict(self) -> dict[str, object]:
        states = []
        for parameter in self.parameters:
            first, second = self._state_for(parameter)
            states.append({"first_moment": first, "second_moment": second})
        return {
            "step": self.step_count,
            "states": states,
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "betas": (self.beta1, self.beta2),
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
                "max_grad_norm": self.max_grad_norm,
            },
        }
