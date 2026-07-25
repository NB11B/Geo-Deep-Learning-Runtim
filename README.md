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

Stages are cumulative. The `training` stage includes all lower stages (`loss`, `attention`, `position`, `activation`, `core`, `linear`).

| Stage | Operations | Hardware Acceptance Status |
|---|---|---|
| `linear` | linear forward and VJP | Accepted (RTX 5070 / CUDA 13.1) |
| `core` | linear, add, multiply, scale, RMSNorm | Accepted (RTX 5070 / CUDA 13.1) |
| `activation` | core plus exact GELU and fused SiLU-multiply | Accepted (RTX 5070 / CUDA 13.1) |
| `position` | activation plus split-half RoPE construction, application, and VJP | Accepted (RTX 5070 / CUDA 13.1) |
| `attention` | causal attention forward and VJP | Accepted (RTX 5070 / CUDA 13.1) |
| `loss` | cross-entropy forward and VJP | Accepted (RTX 5070 / CUDA 13.1) |
| `training` | complete GEOSDP training primitive surface (embedding, clipping, AdamW) | Accepted (RTX 5070 / CUDA 13.1) |

PyTorch is used only to allocate tensors, schedule autograd, provide the active CUDA stream, move registered buffers with the model, and act as an independent parity oracle in tests.

## Repository layout

```text
native/               modular C++/CUDA extension bindings
src/geo_dl_runtime/   Python package, capability union, and autograd functions
tests/                host and CUDA parity tests
scripts/              cuda_acceptance.ps1 validation harness
setup.py              extension build against GEO_ROOT
```

The runtime builds six native modules:

```text
_C            linear, core, activation (GELU, SiLU-multiply, RMSNorm)
_rope         split-half RoPE table construction, application, and VJP
_attention    causal attention forward and VJP
_loss         fused cross-entropy forward and VJP
_embedding    lookup and repeated-index scatter-add VJP
_optimizer    native AdamW fused update and gradient clipping
```

The public `geo_dl_runtime` package validates all native modules independently and unions their capability sets.

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
$env:FORCE_CUDA = "1"
$env:DISTUTILS_USE_SDK = "1"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"
$env:NVCC_APPEND_FLAGS = "-allow-unsupported-compiler"

python -m pip install -U pip setuptools wheel ninja pytest
python -m pip install -e ".[dev]" --no-build-isolation --no-deps
python -m pytest -q
```

## Hardware Acceptance Status

The complete runtime surface has passed hardware acceptance testing on NVIDIA Blackwell (`sm_120`) under CUDA 13.1 and MSVC 19.50:

- **GEO Host C++ Kernel Suite**: 36/36 tests passed (100%)
- **Runtime Capability Test Suite**: 53/53 tests passed (100%)
- **CUDA Parity Test Suite**: 31/31 configurations passed (100%)
- **GEOSDP Test Suite**: 32/32 tests passed (100%)
- **Native Two-Step Training Smoke**: PASSED (`step=1 loss=5.542590`, `step=2 loss=5.570806`, `peak_gpu_memory=1.65MB`)

## Target environment

- NVIDIA RTX 5070 (`sm_120`, Compute Capability 12.0)
- CUDA 13.1
- Python 3.11+
- Visual Studio 2026 / 2022 C++ Build Tools (`cl.exe` v19.50)
- PyTorch `2.13.0+cu130` used as tensor host, graph scheduler, and independent correctness oracle
