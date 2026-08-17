from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable

GEO_DL_RUNTIME_ABI_VERSION = 1

LINEAR_CAPABILITIES = frozenset({"linear"})
CORE_CAPABILITIES = frozenset({"linear", "add", "mul", "scale", "rms_norm"})
ACTIVATION_CAPABILITIES = frozenset({"gelu", "silu_mul"})
POSITION_CAPABILITIES = frozenset({"build_rope", "apply_rope"})
ATTENTION_CAPABILITIES = frozenset({"causal_attention"})
LOSS_CAPABILITIES = frozenset({"cross_entropy"})
EMBEDDING_CAPABILITIES = frozenset({"embedding"})
OPTIMIZER_CAPABILITIES = frozenset({"adamw"})

RELATIONAL_CAPABILITIES = frozenset({
    "birkhoff_project",
    "relational_identity_gate",
    "relational_mix",
    "relational_read",
    "relational_write_add",
})

ACTIVATION_STAGE_CAPABILITIES = frozenset(set(CORE_CAPABILITIES) | set(ACTIVATION_CAPABILITIES))
POSITION_STAGE_CAPABILITIES = frozenset(set(ACTIVATION_STAGE_CAPABILITIES) | set(POSITION_CAPABILITIES))
ATTENTION_STAGE_CAPABILITIES = frozenset(set(POSITION_STAGE_CAPABILITIES) | set(ATTENTION_CAPABILITIES))
LOSS_STAGE_CAPABILITIES = frozenset(set(ATTENTION_STAGE_CAPABILITIES) | set(LOSS_CAPABILITIES))
MODEL_STAGE_CAPABILITIES = frozenset(set(LOSS_STAGE_CAPABILITIES) | set(EMBEDDING_CAPABILITIES))
TRAINING_STAGE_CAPABILITIES = frozenset(set(MODEL_STAGE_CAPABILITIES) | set(OPTIMIZER_CAPABILITIES) | set(RELATIONAL_CAPABILITIES))
RELATIONAL_STAGE_CAPABILITIES = frozenset(set(TRAINING_STAGE_CAPABILITIES))
TRANSFORMER_CAPABILITIES = MODEL_STAGE_CAPABILITIES


@dataclass(frozen=True)
class RuntimeCapabilities:
    operations: FrozenSet[str]

    @classmethod
    def from_iterable(cls, operations: Iterable[str]) -> "RuntimeCapabilities":
        return cls(frozenset(operations))

    def supports(self, required: Iterable[str]) -> bool:
        return frozenset(required).issubset(self.operations)

    def require(self, required: Iterable[str], *, stage: str) -> None:
        missing = sorted(frozenset(required) - self.operations)
        if missing:
            raise RuntimeError(
                f"GEO runtime stage {stage!r} is unavailable; missing capabilities: "
                + ", ".join(missing)
            )


STAGES = {
    "linear": LINEAR_CAPABILITIES,
    "core": CORE_CAPABILITIES,
    "activation": ACTIVATION_STAGE_CAPABILITIES,
    "position": POSITION_STAGE_CAPABILITIES,
    "attention": ATTENTION_STAGE_CAPABILITIES,
    "loss": LOSS_STAGE_CAPABILITIES,
    "model": MODEL_STAGE_CAPABILITIES,
    "transformer": TRANSFORMER_CAPABILITIES,
    "training": TRAINING_STAGE_CAPABILITIES,
    "relational": RELATIONAL_STAGE_CAPABILITIES,
}
