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

## Capability stages

| Stage | Operations | Status |
|---|---|---|
| `linear` | linear forward and VJP | implemented |
| `core` | linear, add, multiply, scale, RMSNorm | implemented; physical CUDA acceptance pending |
| `activation` | GELU, SiLU, fused SiLU-multiply | next |
| `position` | RoPE construction and application | planned |
| `attention` | causal attention forward and VJP | planned |
| `loss` | cross-entropy forward and VJP | planned |
| `transformer` | complete primitive surface required by GEOSDP | planned |

## Repository layout

```text
native/               C++ bridge into GEO CPU/CUDA kernels
src/geo_dl_runtime/   Python package, capabilities, custom autograd functions
tests/                CPU and CUDA forward/VJP parity tests
setup.py              native extension build against a GEO checkout
```

## Current native surface

```python
geo_dl_runtime.linear(x, weight)
geo_dl_runtime.add(a, b)
geo_dl_runtime.mul(a, b)
geo_dl_runtime.scale(x, scalar)
geo_dl_runtime.rms_norm(x, weight, epsilon)
```

Every operation has a GEO-owned backward path. The native module declares the `core` capability only when all five operations are compiled into the extension.

## Local build

The corresponding GEO tensor sources currently live on the GEO integration branch until physical acceptance and merge:

```powershell
git clone https://github.com/NB11B/GeometricElementaryOperators.git
git clone https://github.com/NB11B/Geo-Deep-Learning-Runtim.git
git clone https://github.com/NB11B/GEOSDP.git

cd GeometricElementaryOperators
git checkout agent/geo-dl-runtime-v1

cd ..\Geo-Deep-Learning-Runtim
$env:GEO_ROOT = (Resolve-Path ..\GeometricElementaryOperators)
$env:TORCH_CUDA_ARCH_LIST = "12.0"
python -m pip install -U pip setuptools wheel ninja
python -m pip install -e . --no-build-isolation
pytest -q

cd ..\GEOSDP
git checkout agent/geo-dl-runtime-v1
python -m pip install -e ".[dev]"
pytest -q
```

## Acceptance boundary

A source-level implementation is not a physical CUDA acceptance result. Before the `core` stage is considered released, the following must pass on the RTX 5070/CUDA 13 target:

- extension compilation for compute capability 12.0;
- CPU forward and VJP parity;
- CUDA forward and VJP parity;
- finite-gradient checks;
- repeated training-step smoke tests;
- no hidden PyTorch or ATen numerical fallback.

## Target environment

- NVIDIA RTX 5070
- CUDA 13.x
- Python 3.11+
- a CUDA-compatible PyTorch build used only as tensor host and graph scheduler
