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
| `core` | linear, add, multiply, scale, RMSNorm | implemented; CUDA acceptance pending |
| `activation` | GELU, fused SiLU-multiply | GEO kernels implemented; binding in progress |
| `position` | RoPE construction and application | pending |
| `attention` | causal attention forward and VJP | pending |
| `loss` | cross-entropy forward and VJP | pending |
| `transformer` | complete GEOSDP primitive surface | pending |

## Repository layout

```text
native/               C++/CUDA extension bindings
src/geo_dl_runtime/   Python package and autograd functions
tests/                host and CUDA parity tests
setup.py              extension build against GEO_ROOT
```

## Local build

Expected checkout layout:

```text
workspace/
├── GeometricElementaryOperators/
├── Geo-Deep-Learning-Runtim/
└── GEOSDP/
```

### PowerShell

```powershell
cd GeometricElementaryOperators
git checkout agent/geo-dl-runtime-v1

cd ..\Geo-Deep-Learning-Runtim
$env:GEO_ROOT = (Resolve-Path ..\GeometricElementaryOperators)
$env:TORCH_CUDA_ARCH_LIST = "12.0"
python -m pip install -U pip setuptools wheel ninja
python -m pip install -e ".[dev]" --no-build-isolation
pytest -q
```

### Bash

```bash
cd GeometricElementaryOperators
git checkout agent/geo-dl-runtime-v1

cd ../Geo-Deep-Learning-Runtim
export GEO_ROOT="$(cd ../GeometricElementaryOperators && pwd)"
export TORCH_CUDA_ARCH_LIST="12.0"
python -m pip install -U pip setuptools wheel ninja
python -m pip install -e '.[dev]' --no-build-isolation
pytest -q
```

## Target environment

- NVIDIA RTX 5070
- CUDA 13.x
- Python 3.11+
- CUDA-compatible PyTorch used as tensor host, graph scheduler, and independent correctness oracle
