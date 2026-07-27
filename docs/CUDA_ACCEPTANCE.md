# CUDA 13 / RTX 5070 Acceptance Procedure

This procedure is the release gate for the GEO Deep Learning Runtime and GEOSDP native-training path.

It validates this chain:

```text
GEOSDP model and training loop
    -> geo_dl_runtime autograd and optimizer bindings
    -> GeometricElementaryOperators tensor kernels and VJPs
    -> CUDA 13.x
    -> NVIDIA RTX 5070
```

A successful build alone is not acceptance. The complete forward, backward, embedding, loss, and optimizer surfaces must pass numerical parity and an end-to-end training step.

## Expected checkout layout

```text
workspace/
├── GeometricElementaryOperators/
├── Geo-Deep-Learning-Runtim/
└── GEOSDP/
```

Use these development branches until their PRs are merged:

```powershell
git -C .\GeometricElementaryOperators checkout agent/geo-dl-runtime-v1
git -C .\Geo-Deep-Learning-Runtim checkout agent/activation-stage-v1
git -C .\GEOSDP checkout agent/geo-dl-runtime-v1
```

## Prerequisites

- Windows 11 x64.
- NVIDIA driver that supports the installed CUDA 13 toolkit.
- CUDA 13.x toolkit with `nvcc` on `PATH`.
- Visual Studio 2022 Build Tools with the Desktop development with C++ workload.
- CMake and Ninja.
- Python 3.11 or newer.
- A CUDA-enabled PyTorch build capable of compiling and executing for compute capability 12.0.
- Sufficient free disk space for six native extension modules and CMake artifacts.

Open a new **x64 Native Tools Command Prompt for VS 2022**, then start PowerShell from that prompt. This ensures `cl.exe`, the Windows SDK, and `nvcc` can locate one another.

## Preflight checks

Run:

```powershell
nvidia-smi
nvcc --version
where.exe cl
where.exe nvcc
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

Expected critical results:

```text
torch.cuda.is_available() == True
device capability == (12, 0)
```

Do not continue when PyTorch reports no CUDA device, the compiler cannot be found, or the reported device capability is not `(12, 0)`.

## One-command acceptance run

From the runtime repository:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\cuda_acceptance.ps1 -Workspace C:\path\to\workspace
```

The script performs:

1. Driver, toolkit, Python, PyTorch, GPU, and compute-capability checks.
2. Repository revision capture.
3. Deterministic GEO CPU compilation and CTest execution.
4. Forced CUDA extension compilation for `sm_120`.
5. Runtime ABI and complete `training` capability validation.
6. CPU and CUDA forward/gradient parity tests.
7. Native RoPE, attention, cross-entropy, embedding, and AdamW tests.
8. GEOSDP model tests.
9. One complete native forward/backward/update test.
10. A two-step GEOSDP CUDA training smoke.
11. Log and JSON summary generation under `artifacts/cuda-acceptance-<timestamp>/`.

The environment variables used by the script are:

```powershell
$env:GEO_ROOT = "C:\path\to\workspace\GeometricElementaryOperators"
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:FORCE_CUDA = "1"
```

## Manual procedure

Use this when isolating a failure.

### 1. Build GEO host verification

```powershell
$workspace = "C:\path\to\workspace"
$geo = Join-Path $workspace "GeometricElementaryOperators"
$build = Join-Path $workspace "build-geo-host"

cmake -S $geo -B $build `
  -DGEO_BUILD_TESTS=ON `
  -DGEO_BUILD_BENCHMARKS=OFF `
  -DGEO_BUILD_TOOLS=OFF `
  -DGEO_USE_DOUBLE=OFF
cmake --build $build --config Release --parallel
ctest --test-dir $build -C Release --output-on-failure
```

All tensor tests must pass, including linear, core, activation, RoPE, attention, loss, embedding, and optimizer.

### 2. Build the runtime for the RTX 5070

```powershell
$runtime = Join-Path $workspace "Geo-Deep-Learning-Runtim"
$env:GEO_ROOT = $geo
$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:FORCE_CUDA = "1"
$env:MAX_JOBS = [Math]::Max(1, [Environment]::ProcessorCount - 1)

Set-Location $runtime
python -m pip install -U pip setuptools wheel ninja pytest
python -m pip install -e ".[dev]" --no-build-isolation --force-reinstall
```

Verify the runtime contract:

```powershell
python -c "import geo_dl_runtime as r; print(sorted(r.GEO_CAPABILITIES)); r.require_stage('training'); print('training stage: READY')"
```

The capability set must include:

```text
linear
add
mul
scale
rms_norm
gelu
silu_mul
build_rope
apply_rope
causal_attention
cross_entropy
embedding
adamw
```

### 3. Run parity and native tests

```powershell
python -m pytest -q
python -m pytest -q -m cuda
python -m pytest -q -m native --maxfail=1
```

No CUDA test may be silently skipped. Review the pytest summary and treat unexpected skips as a failure.

### 4. Run the native GEOSDP training gate

```powershell
$geosdp = Join-Path $workspace "GEOSDP"
Set-Location $geosdp
python -m pip install -e ".[dev]"
python -m pytest -q
python -m pytest -q tests/test_native_training_step.py -s
python -m geosdp.train --config configs/smoke.yaml --backend native --smoke
```

Expected behavior:

- The native backend reports `GeometricElementaryOperators`.
- The training capability gate succeeds.
- Loss is finite on both smoke steps.
- Parameters receive finite gradients.
- GEO AdamW updates at least one parameter.
- No operation falls back to the reference backend.

## Acceptance criteria

The CUDA runtime is accepted only when all of the following are true:

- GEO host CTest passes.
- Every runtime extension compiles for compute capability 12.0.
- `require_stage("training")` succeeds.
- CPU parity tests pass.
- CUDA forward parity tests pass.
- CUDA VJP parity tests pass.
- Repeated-token embedding gradient accumulation passes.
- Causal-attention gradients for `q`, `k`, and `v` pass.
- Cross-entropy, including `ignore_index`, passes.
- GEO AdamW matches the reference update within the declared tolerance.
- End-to-end native GEOSDP training completes at least two steps with finite loss.
- The acceptance log identifies the exact commit of all three repositories.

## Failure triage

### `nvcc` cannot find the host compiler

Launch PowerShell from the Visual Studio x64 Native Tools prompt and verify `where.exe cl`.

### Unsupported GPU architecture

Verify:

```powershell
$env:TORCH_CUDA_ARCH_LIST = "12.0"
python -c "import torch; print(torch.cuda.get_device_capability(0))"
```

The installed PyTorch distribution and CUDA toolkit must both support the target architecture.

### Extension imports but `GEO_CUDA_AVAILABLE` is false

The runtime was probably built in CPU-only mode. Remove build products and reinstall with `FORCE_CUDA=1`.

```powershell
Remove-Item -Recurse -Force .\build -ErrorAction SilentlyContinue
Get-ChildItem .\src\geo_dl_runtime -Filter "*.pyd" | Remove-Item -Force
$env:FORCE_CUDA = "1"
python -m pip install -e ".[dev]" --no-build-isolation --force-reinstall
```

### Numerical parity failure

Record:

- operation name;
- device;
- tensor shape;
- seed;
- maximum absolute error;
- maximum relative error;
- forward or VJP failure;
- repository SHAs;
- driver, toolkit, and PyTorch versions.

Do not increase tolerances until the discrepancy has been explained.

### CUDA out of memory

The parity suite uses small tensors. An out-of-memory error usually indicates another process is occupying the GPU or a kernel/allocation defect. Check `nvidia-smi` before reducing test shapes.

## Artifact retention

Retain the complete acceptance directory with the PR or release artifact. It is the initial source for GLOS backend-parity, gradient, numerical, resource, and provenance certificates.
