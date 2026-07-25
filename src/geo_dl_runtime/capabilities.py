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
TRANSFORMER_CAPABILITIES = frozenset(
    set(CORE_CAPABILITIES)
    | set(ACTIVATION_CAPABILITIES)
    | set(POSITION_CAPABILITIES)
    | set(ATTENTION_CAPABILITIES)
    | set(LOSS_CAPABILITIES)
)


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
    "transformer": TRANSFORMER_CAPABILITIES,
}
