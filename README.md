# GEO Deep Learning Runtime

GEO Deep Learning Runtime is the tensor, autograd, CUDA-binding, and optimizer layer built on top of [NB11B/GeometricElementaryOperators](https://github.com/NB11B/GeometricElementaryOperators).

It intentionally sits between the mathematical kernel and trained models:

```text
GeometricElementaryOperators
    core geometric operators, plans, forward kernels, JVP/VJP

Geo-Deep-Learning-Runtim
    tensor runtime, PyTorch bridge, autograd bindings, optimizers, model primitives

GEOSDP
    tokenizer, corpus, model definition, training, evaluation, export
```

## Design rules

1. GEO owns every numerical forward and backward operation.
2. PyTorch may own tensor storage and graph scheduling, but not hidden numerical recomputation.
3. No ATen, cuBLAS, or PyTorch math may be presented as GEO execution.
4. Runtime capabilities are versioned and explicit.
5. Incomplete capability sets fail closed.
6. CPU and CUDA paths must be independently validated against mathematical references.

## Initial capability stages

| Stage | Operations |
|---|---|
| `linear` | linear forward and VJP |
| `core` | linear, add, multiply, scale, RMSNorm |
| `activation` | GELU, SiLU, fused SiLU-multiply |
| `position` | RoPE construction and application |
| `attention` | causal attention forward and VJP |
| `loss` | cross-entropy forward and VJP |
| `transformer` | complete primitive surface required by GEOSDP |

## Repository layout

```text
include/geo_dl/       public runtime ABI
native/               C++/CUDA extension bindings
src/geo_dl_runtime/   Python package and autograd functions
tests/                host and CUDA parity tests
cmake/                native build integration
```

## Current milestone

The first milestone is to migrate the existing `geo_torch` linear bridge out of GEOSDP and extend it with elementwise arithmetic and fused RMSNorm. GeometricElementaryOperators remains the source of the underlying kernels and VJPs.

## Target environment

- NVIDIA RTX 5070
- CUDA 13.x
- Python 3.11+
- a CUDA-compatible PyTorch build used as tensor host and graph scheduler
