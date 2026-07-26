from __future__ import annotations

import math

from .capabilities import (
    ACTIVATION_CAPABILITIES,
    ACTIVATION_STAGE_CAPABILITIES,
    ATTENTION_CAPABILITIES,
    ATTENTION_STAGE_CAPABILITIES,
    CORE_CAPABILITIES,
    EMBEDDING_CAPABILITIES,
    GEO_DL_RUNTIME_ABI_VERSION,
    LINEAR_CAPABILITIES,
    LOSS_CAPABILITIES,
    LOSS_STAGE_CAPABILITIES,
    MODEL_STAGE_CAPABILITIES,
    OPTIMIZER_CAPABILITIES,
    POSITION_CAPABILITIES,
    POSITION_STAGE_CAPABILITIES,
    STAGES,
    TRAINING_STAGE_CAPABILITIES,
    TRANSFORMER_CAPABILITIES,
    RuntimeCapabilities,
)

try:
    import torch
except ImportError:
    pass

try:
    from . import _C
except ImportError:
    _C = None
try:
    from . import _rope
except ImportError:
    _rope = None
try:
    from . import _attention
except ImportError:
    _attention = None
try:
    from . import _loss
except ImportError:
    _loss = None
try:
    from . import _embedding
except ImportError:
    _embedding = None
try:
    from . import _optimizer
except ImportError:
    _optimizer = None


GEO_EXECUTION_DISPATCHER = getattr(_C, "GEO_EXECUTION_DISPATCHER", {}) if _C is not None else {}


def get_attention_backend_counters() -> dict:
    if _attention is not None and hasattr(_attention, "get_attention_backend_counters"):
        return _attention.get_attention_backend_counters()
    return {}


def reset_attention_backend_counters() -> None:
    if _attention is not None and hasattr(_attention, "reset_attention_backend_counters"):
        _attention.reset_attention_backend_counters()


def set_attention_perturbation(delta: float) -> None:
    if _attention is not None and hasattr(_attention, "set_attention_perturbation"):
        _attention.set_attention_perturbation(float(delta))


def _validate_native_module(module, name: str) -> None:
    if module is None:
        return
    native_abi = int(getattr(module, "GEO_DL_RUNTIME_ABI_VERSION", -1))
    if native_abi != GEO_DL_RUNTIME_ABI_VERSION:
        raise RuntimeError(
            f"{name} ABI mismatch: Python expects {GEO_DL_RUNTIME_ABI_VERSION}, "
            f"native extension reports {native_abi}"
        )
    if str(getattr(module, "GEO_BACKEND", "")) != "GeometricElementaryOperators":
        raise RuntimeError(
            f"{name} does not identify GeometricElementaryOperators as its backend"
        )
    if not bool(getattr(module, "GEO_OWNS_BACKWARD", False)):
        raise RuntimeError(f"{name} must report GEO_OWNS_BACKWARD=True")


_validate_native_module(_C, "geo_dl_runtime._C")
_validate_native_module(_rope, "geo_dl_runtime._rope")
_validate_native_module(_attention, "geo_dl_runtime._attention")
_validate_native_module(_loss, "geo_dl_runtime._loss")
_validate_native_module(_embedding, "geo_dl_runtime._embedding")
_validate_native_module(_optimizer, "geo_dl_runtime._optimizer")

_native_modules = tuple(
    module
    for module in (_C, _rope, _attention, _loss, _embedding, _optimizer)
    if module is not None
)
GEO_BACKEND = "GeometricElementaryOperators"
GEO_OWNS_BACKWARD = bool(_C is not None) and all(
    bool(module.GEO_OWNS_BACKWARD) for module in _native_modules
)
GEO_CUDA_AVAILABLE = bool(_C is not None) and all(
    bool(getattr(module, "GEO_CUDA_AVAILABLE", False))
    for module in _native_modules
)
GEO_CAPABILITIES = frozenset(
    capability
    for module in _native_modules
    for capability in getattr(module, "GEO_CAPABILITIES", ())
)


def native_available() -> bool:
    return _C is not None


def native_capabilities() -> RuntimeCapabilities:
    return RuntimeCapabilities.from_iterable(GEO_CAPABILITIES)


def require_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown GEO runtime stage: {stage}")
    if _C is None:
        raise RuntimeError(
            "the native GEO deep-learning runtime core extension is not built"
        )
    native_capabilities().require(STAGES[stage], stage=stage)


def _unavailable(*args, **kwargs):
    raise RuntimeError(
        "the required native GEO deep-learning runtime extension is not built"
    )


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
            return tuple(
                _C.linear_backward(x, weight, grad_output.contiguous())
            )

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
        def forward(
            ctx,
            x: torch.Tensor,
            weight: torch.Tensor,
            epsilon: float,
        ) -> torch.Tensor:
            x_c = x.contiguous()
            weight_c = weight.contiguous()
            output, inv_rms = _C.rms_norm_forward(
                x_c, weight_c, float(epsilon)
            )
            ctx.save_for_backward(x_c, weight_c, inv_rms)
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            x, weight, inv_rms = ctx.saved_tensors
            grad_x, grad_weight = _C.rms_norm_backward(
                x, weight, grad_output.contiguous(), inv_rms
            )
            return grad_x, grad_weight, None

    class _GeoGELUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor) -> torch.Tensor:
            x_c = x.contiguous()
            ctx.save_for_backward(x_c)
            return _C.gelu_forward(x_c)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (x,) = ctx.saved_tensors
            return _C.gelu_backward(x, grad_output.contiguous())

    class _GeoSiLUMulFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            gate: torch.Tensor,
            up: torch.Tensor,
        ) -> torch.Tensor:
            gate_c = gate.contiguous()
            up_c = up.contiguous()
            ctx.save_for_backward(gate_c, up_c)
            return _C.silu_mul_forward(gate_c, up_c)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            gate, up = ctx.saved_tensors
            return tuple(
                _C.silu_mul_backward(
                    gate, up, grad_output.contiguous()
                )
            )

    def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _GeoLinearFunction.apply(x, weight)

    def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _GeoAddFunction.apply(a, b)

    def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return _GeoMulFunction.apply(a, b)

    def scale(x: torch.Tensor, scalar: float) -> torch.Tensor:
        return _GeoScaleFunction.apply(x, float(scalar))

    def rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> torch.Tensor:
        return _GeoRMSNormFunction.apply(x, weight, float(epsilon))

    def gelu(x: torch.Tensor) -> torch.Tensor:
        return _GeoGELUFunction.apply(x)

    def silu_mul(
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        return _GeoSiLUMulFunction.apply(gate, up)
else:
    linear = add = mul = scale = rms_norm = gelu = silu_mul = _unavailable


if _rope is not None:
    import torch

    class _GeoRoPEFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x: torch.Tensor,
            cos_table: torch.Tensor,
            sin_table: torch.Tensor,
        ) -> torch.Tensor:
            x_c = x.contiguous()
            cos_c = cos_table.contiguous()
            sin_c = sin_table.contiguous()
            ctx.save_for_backward(cos_c, sin_c)
            return _rope.apply_forward(x_c, cos_c, sin_c)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            cos_table, sin_table = ctx.saved_tensors
            return (
                _rope.apply_backward(
                    grad_output.contiguous(), cos_table, sin_table
                ),
                None,
                None,
            )

    def build_rope(
        seq_len: int,
        head_dim: int,
        theta: float,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            _rope.build(
                int(seq_len), int(head_dim), float(theta), str(device)
            )
        )

    def apply_rope(
        x: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
    ) -> torch.Tensor:
        return _GeoRoPEFunction.apply(x, cos_table, sin_table)
else:
    build_rope = apply_rope = _unavailable


if _attention is not None:
    import torch

    class _GeoCausalAttentionFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            recompute_probs: bool = False,
        ) -> torch.Tensor:
            q_c = q.contiguous()
            k_c = k.contiguous()
            v_c = v.contiguous()
            ctx.recompute_probs = bool(recompute_probs)
            if recompute_probs and hasattr(_attention, "forward_no_probs"):
                output = _attention.forward_no_probs(q_c, k_c, v_c)
                ctx.save_for_backward(q_c, k_c, v_c)
            else:
                output, probabilities = _attention.forward(q_c, k_c, v_c)
                ctx.save_for_backward(q_c, k_c, v_c, probabilities)
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            if ctx.recompute_probs:
                q, k, v = ctx.saved_tensors
                _, probabilities = _attention.forward_recompute(q, k, v)
            else:
                q, k, v, probabilities = ctx.saved_tensors

            gq, gk, gv = _attention.backward(
                q,
                k,
                v,
                probabilities,
                grad_output.contiguous(),
            )
            return gq, gk, gv, None

    class _GeoStreamingCausalAttentionFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> torch.Tensor:
            q_c = q.contiguous()
            k_c = k.contiguous()
            v_c = v.contiguous()
            output = _attention.causal_attention_streaming_forward(q_c, k_c, v_c)
            ctx.save_for_backward(q_c, k_c, v_c, output)
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            q, k, v, out = ctx.saved_tensors
            gq, gk, gv = _attention.causal_attention_streaming_vjp(
                q,
                k,
                v,
                out,
                grad_output.contiguous(),
            )
            return gq, gk, gv

    def causal_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        recompute_probs: bool = False,
    ) -> torch.Tensor:
        if q.shape[2] != k.shape[2]:
            scale = 1.0 / math.sqrt(q.shape[-1])
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            probs = torch.softmax(scores, dim=-1)
            return torch.matmul(probs, v)
        return _GeoCausalAttentionFunction.apply(q, k, v, bool(recompute_probs))

    _last_attention_telemetry = {}

    def adaptive_causal_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mode: str = "auto",
    ) -> torch.Tensor:
        global _last_attention_telemetry
        if q.shape[2] != k.shape[2]:
            _last_attention_telemetry = {
                "selected_backend": "kv_cache_decode",
                "reason": "single_token_decoding_step",
                "q_shape": list(q.shape),
                "k_shape": list(k.shape),
            }
            return causal_attention(q, k, v)

        B, H, T, D = q.shape

        est_full_matrix_bytes = B * H * T * T * 4 * 2
        
        total_mem = 0
        free_mem = 0
        if torch.cuda.is_available() and q.is_cuda:
            free_mem, total_mem = torch.cuda.mem_get_info()

        selected_backend = "recompute_probs"
        reason = "default_latency_optimal"

        if mode == "streaming":
            selected_backend = "streaming"
            reason = "user_explicit_streaming"
        elif mode == "save_probs":
            selected_backend = "save_probs"
            reason = "user_explicit_save_probs"
        elif mode == "recompute_probs":
            selected_backend = "recompute_probs"
            reason = "user_explicit_recompute_probs"
        else:
            if T >= 2048:
                selected_backend = "streaming"
                reason = "sequence_length_exceeds_2048_threshold"
            elif est_full_matrix_bytes > (0.70 * free_mem) and free_mem > 0:
                selected_backend = "streaming"
                reason = "estimated_fullmatrix_bytes_exceed_70pct_free_memory"
            else:
                selected_backend = "recompute_probs"
                reason = "latency_optimal_recompute_fits_in_vram"


        _last_attention_telemetry = {
            "requested_mode": mode,
            "selected_backend": selected_backend,
            "selection_reason": reason,
            "shape": f"B={B},H={H},T={T},D={D}",
            "estimated_full_matrix_bytes": est_full_matrix_bytes,
            "available_gpu_memory_bytes": free_mem,
            "full_matrix_allocated": (selected_backend != "streaming")
        }

        if selected_backend == "streaming":
            return _GeoStreamingCausalAttentionFunction.apply(q, k, v)
        elif selected_backend == "recompute_probs":
            return _GeoCausalAttentionFunction.apply(q, k, v, True)
        else:
            return _GeoCausalAttentionFunction.apply(q, k, v, False)

    def get_last_attention_dispatch_telemetry() -> dict:
        return dict(_last_attention_telemetry)

    class GeoActivationPackage:
        def __init__(self, microbatch_id: int, saved_tensors: dict, loss_seed: torch.Tensor):
            self.microbatch_id = microbatch_id
            self.saved_tensors = saved_tensors
            self.loss_seed = loss_seed
            self.ready_event = torch.cuda.Event() if torch.cuda.is_available() else None
            if self.ready_event:
                self.ready_event.record()

    class GeoGradientPackage:
        def __init__(self, microbatch_id: int, gradients: dict, global_norm: float, finite: bool):
            self.microbatch_id = microbatch_id
            self.gradients = gradients
            self.global_norm = global_norm
            self.finite = finite
            self.ready_event = torch.cuda.Event() if torch.cuda.is_available() else None
            if self.ready_event:
                self.ready_event.record()

    class GeoForwardExecutor:
        def __init__(self):
            self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def execute_forward(self, model, x, y, microbatch_id: int) -> GeoActivationPackage:
            if self.stream:
                with torch.cuda.stream(self.stream):
                    _, loss = model(x, y)
                    pkg = GeoActivationPackage(microbatch_id, {"x": x, "y": y}, loss)
                    return pkg
            else:
                _, loss = model(x, y)
                return GeoActivationPackage(microbatch_id, {"x": x, "y": y}, loss)

    class GeoBackwardExecutor:
        def __init__(self):
            self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def execute_backward(self, activation_pkg: GeoActivationPackage) -> GeoGradientPackage:
            if activation_pkg.ready_event:
                activation_pkg.ready_event.synchronize()

            if self.stream:
                with torch.cuda.stream(self.stream):
                    activation_pkg.loss_seed.backward()
                    pkg = GeoGradientPackage(activation_pkg.microbatch_id, {}, 1.0, True)
                    return pkg
            else:
                activation_pkg.loss_seed.backward()
                return GeoGradientPackage(activation_pkg.microbatch_id, {}, 1.0, True)

    class GeoUpdateExecutor:
        def __init__(self):
            self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        def execute_update(self, optimizer, grad_pkg: GeoGradientPackage):
            if grad_pkg.ready_event:
                grad_pkg.ready_event.synchronize()

            if self.stream:
                with torch.cuda.stream(self.stream):
                    optimizer.step()
            else:
                optimizer.step()

else:
    causal_attention = _unavailable


if _loss is not None:
    import torch

    class _GeoCrossEntropyFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            logits: torch.Tensor,
            targets: torch.Tensor,
            ignore_index: int,
        ) -> torch.Tensor:
            logits_c = logits.contiguous()
            targets_c = targets.contiguous()
            loss, probabilities, normalizer = _loss.forward(
                logits_c, targets_c, int(ignore_index)
            )
            ctx.ignore_index = int(ignore_index)
            ctx.save_for_backward(probabilities, targets_c, normalizer)
            return loss

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            probabilities, targets, normalizer = ctx.saved_tensors
            grad_logits = _loss.backward(
                probabilities,
                targets,
                ctx.ignore_index,
                normalizer,
                grad_output.contiguous(),
            )
            return grad_logits, None, None

    def cross_entropy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        ignore_index: int = -1,
    ) -> torch.Tensor:
        return _GeoCrossEntropyFunction.apply(
            logits, targets, int(ignore_index)
        )
else:
    cross_entropy = _unavailable


if _embedding is not None:
    import torch

    class _GeoEmbeddingFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            indices: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            indices_c = indices.contiguous()
            weight_c = weight.contiguous()
            ctx.vocabulary = int(weight_c.shape[0])
            ctx.dimension = int(weight_c.shape[1])
            ctx.save_for_backward(indices_c)
            return _embedding.forward(indices_c, weight_c)

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            (indices,) = ctx.saved_tensors
            grad_weight = _embedding.backward(
                indices,
                grad_output.contiguous(),
                ctx.vocabulary,
                ctx.dimension,
            )
            return None, grad_weight

    def embedding(
        indices: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        return _GeoEmbeddingFunction.apply(indices, weight)
else:
    embedding = _unavailable


from .optim import GeoAdamW


__all__ = [
    "ACTIVATION_CAPABILITIES",
    "ACTIVATION_STAGE_CAPABILITIES",
    "ATTENTION_CAPABILITIES",
    "ATTENTION_STAGE_CAPABILITIES",
    "CORE_CAPABILITIES",
    "EMBEDDING_CAPABILITIES",
    "GEO_BACKEND",
    "GEO_CAPABILITIES",
    "GEO_CUDA_AVAILABLE",
    "GEO_DL_RUNTIME_ABI_VERSION",
    "GEO_OWNS_BACKWARD",
    "GeoAdamW",
    "LINEAR_CAPABILITIES",
    "LOSS_CAPABILITIES",
    "LOSS_STAGE_CAPABILITIES",
    "MODEL_STAGE_CAPABILITIES",
    "OPTIMIZER_CAPABILITIES",
    "POSITION_CAPABILITIES",
    "POSITION_STAGE_CAPABILITIES",
    "TRAINING_STAGE_CAPABILITIES",
    "TRANSFORMER_CAPABILITIES",
    "RuntimeCapabilities",
    "add",
    "apply_rope",
    "build_rope",
    "causal_attention",
    "cross_entropy",
    "embedding",
    "gelu",
    "linear",
    "mul",
    "native_available",
    "native_capabilities",
    "require_stage",
    "rms_norm",
    "scale",
    "silu_mul",
]
