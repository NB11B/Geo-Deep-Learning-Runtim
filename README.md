# GEO Deep Learning Runtime

GEO Deep Learning Runtime is the tensor, autograd, CUDA-binding, execution-dispatch, and optimizer layer built on top of [NB11B/GeometricElementaryOperators](https://github.com/NB11B/GeometricElementaryOperators).

It sits between the mathematical kernel and trained models:

```text
GeometricElementaryOperators
    core operators, execution plans, forward kernels, JVP/VJP

Geo-Deep-Learning-Runtim
    tensor runtime, PyTorch bridge, autograd bindings,
    geometry-aware backend dispatch, optimizers, model primitives

GEOSDP
    tokenizer, corpus, model definition, training, evaluation, export
```

## Execution contract

1. Numerical execution is explicit and auditable.
2. PyTorch may own tensor storage, autograd graph scheduling, and the active CUDA stream.
3. Each operation records the backend that actually executed it; no fallback is presented as native GEO execution.
4. GEO-native kernels are used where their execution geometry is advantageous.
5. Vendor libraries may be selected deliberately for shapes they are optimized to execute, and are reported as such in dispatcher telemetry.
6. Runtime capabilities are versioned and incomplete capability sets fail closed.
7. CPU and CUDA paths are independently validated against mathematical references.

The current runtime is therefore not a claim that every operation should use a custom kernel. It is a geometry-aware execution layer that preserves operator intent while selecting an appropriate implementation for the observed shape, layout, precision, and hardware.

## Adaptive computational dispatch

The runtime currently supports audited shape-aware dispatch across several operation classes:

| Operation | Current execution policy |
|---|---|
| Linear projection | GEO vectorized CUDA for smaller output dimensions; cuBLAS for wide projections |
| Causal attention | `save_probs`, `recompute_probs`, and end-to-end tile-streamed forward/VJP modes |
| Cross entropy | serial row kernel for small vocabularies; parallel block reduction for larger vocabularies |
| Optimizer | fused multi-tensor GeoAdamW with fused global-gradient norm clipping |

Per-invocation telemetry is exposed through the native bridge and records fields such as:

```text
operator
shape
requested_mode
selected_backend
selection_reason
algorithm_id
workspace_bytes
full_matrix_allocated
fallback
fallback_reason
```

Backend counters are also available for attention-path verification, including distinct counters for probability-producing forward execution, no-probability forward execution, streaming forward, standard VJP, and streaming VJP.

## Capability stages

Stages are cumulative. The `training` stage includes all lower stages (`loss`, `attention`, `position`, `activation`, `core`, `linear`).

| Stage | Operations | Hardware acceptance status |
|---|---|---|
| `linear` | linear forward and VJP, including wide-projection dispatch | Accepted (RTX 5070 / CUDA 13.1) |
| `core` | linear, add, multiply, scale, RMSNorm | Accepted |
| `activation` | exact GELU and fused SiLU-multiply | Accepted |
| `position` | split-half RoPE construction, application, and VJP | Accepted |
| `attention` | full-matrix, recompute, and tile-streamed causal attention forward/VJP | Accepted |
| `loss` | serial and parallel cross-entropy forward/VJP | Accepted |
| `training` | embedding, fused clipping, fused AdamW, and complete GEOSDP training surface | Accepted |

PyTorch is used as the tensor host, autograd scheduler, CUDA-stream provider, registered-buffer manager, and independent parity oracle. The selected numerical backend remains explicit.

## Streaming causal attention

The runtime includes a complete tile-streamed causal-attention engine for both forward and backward execution.

The streaming VJP reconstructs score and probability tiles from saved row statistics and accumulates `dQ`, `dK`, and `dV` without materializing a full `B × H × T × T` score or probability matrix.

For fixed tile sizes, retained attention state scales as:

```text
O(B × H × T × D) + bounded tile workspace
```

Validation through `T=2048` demonstrated:

- no full global score/probability matrix in streaming forward or backward;
- output parity with the accepted full-matrix path;
- FP32 gradient differences in the expected reduction-order range;
- measured peak memory increasing linearly with sequence length for the tested fixed batch, head count, and head dimension.

Example measured streaming peaks for the accepted benchmark configuration:

| Sequence length | Peak allocated memory |
|---:|---:|
| 256 | 9 MB |
| 512 | 18 MB |
| 1,024 | 36 MB |
| 2,048 | 72 MB |

These values are configuration-specific and do not imply universal performance or memory ratios.

## Native modules

```text
_C            linear, core, activation, RMSNorm, dispatcher telemetry
_rope         split-half RoPE construction, application, and VJP
_attention    full-matrix, recompute, and streaming causal attention
_loss         serial/parallel cross-entropy forward and VJP
_embedding    lookup and repeated-index scatter-add VJP
_optimizer    fused AdamW update and fused gradient clipping
```

The public `geo_dl_runtime` package validates all native modules independently and unions their capability sets.

## Repository layout

```text
native/               modular C++/CUDA extension bindings
src/geo_dl_runtime/   Python package, capability union, autograd functions
tests/                host and CUDA parity tests
scripts/              CUDA acceptance harness
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

## Accepted system results

On the recorded 32K-vocabulary benchmark (`V=32,768`, `B=2`, `T=64`, `D=64`), targeted optimization reduced the measured training step from `12.337 ms` to `4.918 ms`:

| Phase | Initial | Accepted optimized result |
|---|---:|---:|
| Forward | 4.547 ms | 1.642 ms |
| Backward | 3.456 ms | 2.923 ms |
| Optimizer | 4.335 ms | 0.354 ms |
| Total step | 12.337 ms | 4.918 ms |

Key contributors included fused gradient clipping, fused multi-tensor AdamW, parallel cross entropy, and cuBLAS dispatch for the wide LM-head projection. Results are workload-, implementation-, and environment-specific.

## Hardware acceptance status

The accepted branch has passed hardware validation on NVIDIA Blackwell (`sm_120`) under CUDA 13.1 and MSVC 19.50:

- **GEO Host C++ Kernel Suite:** 36/36 passed
- **Runtime Capability Test Suite:** 59/59 passed
- **CUDA Parity Test Suite:** 37/37 passed
- **GEOSDP Subsystem Test Suite:** 98/98 passed
- **End-to-End Native Training:** 3/3 passed

The two-step smoke test remains a finite-execution and parameter-update gate rather than a convergence benchmark.

## Target environment

- NVIDIA RTX 5070 (`sm_120`, Compute Capability 12.0)
- CUDA 13.1
- Python 3.11+
- Visual Studio C++ Build Tools (`cl.exe` v19.50 in the accepted environment)
- PyTorch `2.13.0+cu130` as tensor host, graph scheduler, stream provider, and independent correctness oracle

## Claim boundary

The runtime has demonstrated correct and repeatable execution under the tested shapes, precision, software stack, and hardware. It does not establish that one backend is universally optimal. The dispatcher exists precisely because the preferred implementation depends on operation geometry and execution conditions.