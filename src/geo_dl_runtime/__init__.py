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


def _unavailable(*args, **kwargs):
    raise RuntimeError("the native GEO deep-learning runtime extension is not built")


if _C is not None:
    import torch

    class _GeoLinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            x_c = x.contiguous()
            weight_c = weight.contiguous()
            ctx.save_for_backward(x_c, weight_c)
            return _C.linear_forward(x_c, weight_c)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            x, weight = ctx.saved_tensors
            return tuple(_C.linear_backward(x, weight, grad_output.contiguous()))

    class _GeoAddFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return _C.add_forward(a.contiguous(), b.contiguous())

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            return tuple(_C.add_backward(grad_output.contiguous()))

    class _GeoMulFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a_c = a.contiguous()
            b_c = b.contiguous()
            ctx.save_for_backward(a_c, b_c)
            return _C.mul_forward(a_c, b_c)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            a, b = ctx.saved_tensors
            return tuple(_C.mul_backward(a, b, grad_output.contiguous()))

    class _GeoScaleFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, scalar: float) -> torch.Tensor:
            ctx.scalar = float(scalar)
            return _C.scale_forward(x.contiguous(), ctx.scalar)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            return _C.scale_backward(grad_output.contiguous(), ctx.scalar), None

    class _GeoRMSNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
            x_c = x.contiguous()
            weight_c = weight.contiguous()
            output, inv_rms = _C.rms_norm_forward(x_c, weight_c, float(epsilon))
            ctx.save_for_backward(x_c, weight_c, inv_rms)
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            x, weight, inv_rms = ctx.saved_tensors
            grad_x, grad_weight = _C.rms_norm_backward(
                x, weight, grad_output.contiguous(), inv_rms
            )
            return grad_x, grad_weight, None

    def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _GeoLinearFunction.apply(x, weight)

    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _GeoAddFunction.apply(a, b)

    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _GeoMulFunction.apply(a, b)

    def scale(x: torch.Tensor, scalar: float) -> torch.Tensor:
        return _GeoScaleFunction.apply(x, float(scalar))

    def rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        return _GeoRMSNormFunction.apply(x, weight, float(epsilon))
else:
    linear = add = mul = scale = rms_norm = _unavailable


__all__ = [
    "GEO_DL_RUNTIME_ABI_VERSION",
    "CORE_CAPABILITIES",
    "LINEAR_CAPABILITIES",
    "TRANSFORMER_CAPABILITIES",
    "RuntimeCapabilities",
    "add",
    "linear",
    "mul",
    "native_available",
    "native_capabilities",
    "require_stage",
    "rms_norm",
    "scale",
]
