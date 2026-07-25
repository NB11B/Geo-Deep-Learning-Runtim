from __future__ import annotations

from .capabilities import (
    GEO_DL_RUNTIME_ABI_VERSION,
    CORE_CAPABILITIES,
    LINEAR_CAPABILITIES,
    STAGES,
    TRANSFORMER_CAPABILITIES,
    RuntimeCapabilities,
)

try:
    from . import _C
except ImportError:
    _C = None


def native_available() -> bool:
    return _C is not None


def native_capabilities() -> RuntimeCapabilities:
    if _C is None:
        return RuntimeCapabilities(frozenset())
    return RuntimeCapabilities.from_iterable(getattr(_C, "GEO_CAPABILITIES", ()))


def require_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown GEO runtime stage: {stage}")
    if _C is None:
        raise RuntimeError("the native GEO deep-learning runtime extension is not built")
    native_capabilities().require(STAGES[stage], stage=stage)


if _C is not None:
    import torch

    class _GeoLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_contiguous = x.contiguous()
            weight_contiguous = weight.contiguous()
            ctx.save_for_backward(x_contiguous, weight_contiguous)
            return _C.linear_forward(x_contiguous, weight_contiguous)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            x, weight = ctx.saved_tensors
            grad_x, grad_weight = _C.linear_backward(x, weight, grad_output.contiguous())
            return grad_x, grad_weight

    def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _GeoLinearFunction.apply(x, weight)
else:
    def linear(*args, **kwargs):
        raise RuntimeError("the native GEO deep-learning runtime extension is not built")


__all__ = [
    "GEO_DL_RUNTIME_ABI_VERSION",
    "CORE_CAPABILITIES",
    "LINEAR_CAPABILITIES",
    "TRANSFORMER_CAPABILITIES",
    "RuntimeCapabilities",
    "linear",
    "native_available",
    "native_capabilities",
    "require_stage",
]
