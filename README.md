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

Stages are cumulative. The `position` stage includes the complete `activation` and `core` stages.

| Stage | Operations | Status |
|---|---|---|
| `linear` | linear forward and VJP | implemented |
| `core` | linear, add, multiply, scale, RMSNorm | implemented; physical CUDA acceptance pending |
| `activation` | core plus exact GELU and fused SiLU-multiply | implemented; physical CUDA acceptance pending |
| `position` | activation plus split-half RoPE construction, application, and VJP | implemented; physical CUDA acceptance pending |
| `attention` | causal attention forward and VJP | pending |
| `loss` | cross-entropy forward and VJP | pending |
| `transformer` | complete GEOSDP primitive surface | pending |

The position implementation matches GEOSDP exactly:

- cosine and sine tables have shape `[sequence, head_dim / 2]`;
- the first and second halves of each head are rotated against one another;
- tables are constant buffers;
- the VJP returns a gradient only for the rotated tensor;
- non-contiguous attention layouts are made contiguous at the runtime boundary before GEO execution.

PyTorch is used only to allocate tensors, schedule autograd, provide the active CUDA stream, move registered buffers with the model, and act as an independent parity oracle in tests.

## Repository layout

```text
native/               modular C++/CUDA extension bindings
src/geo_dl_runtime/   Python package, capability union, and autograd functions
tests/                host and CUDA parity tests
setup.py              extension build against GEO_ROOT
```

The runtime currently builds two native modules:

```text
_C       linear, core, activation
_rope    position tables, application, VJP
```

The public `geo_dl_runtime` package validates both modules independently and unions their capability sets.

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
git checkout agent/activation-stage-v1
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
git checkout agent/activation-stage-v1
export GEO_ROOT="$(cd ../GeometricElementaryOperators && pwd)"
export TORCH_CUDA_ARCH_LIST="12.0"
python -m pip install -U pip setuptools wheel ninja
python -m pip install -e '.[dev]' --no-build-isolation
pytest -q
```

## Acceptance boundary

The checked-in tests compare CPU and CUDA forward values and gradients against exact PyTorch mathematical references, including GEOSDP's non-contiguous attention tensor layout. A physical CUDA 13.x build and run on the target RTX 5070 remains required before the position stage is accepted for training.

## Target environment

- NVIDIA RTX 5070
- CUDA 13.x
- Python 3.11+
- CUDA-compatible PyTorch used as tensor host, graph scheduler, and independent correctness oracle
