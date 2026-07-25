param(
    [string]$Workspace = (Resolve-Path "$PSScriptRoot\..\.."),
    [string]$Python = "python",
    [string]$CudaArch = "12.0",
    [switch]$SkipGeoHostBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$geoRoot = Join-Path $Workspace "GeometricElementaryOperators"
$runtimeRoot = Join-Path $Workspace "Geo-Deep-Learning-Runtim"
$geosdpRoot = Join-Path $Workspace "GEOSDP"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$artifactRoot = Join-Path $runtimeRoot "artifacts\cuda-acceptance-$stamp"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$logPath = Join-Path $artifactRoot "acceptance.log"

function Invoke-Logged {
    param([string]$Label, [scriptblock]$Command)
    "`n===== $Label =====" | Tee-Object -FilePath $logPath -Append
    $oldEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Command 2>&1 | Out-String | Tee-Object -FilePath $logPath -Append
    } finally {
        $ErrorActionPreference = $oldEAP
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

foreach ($path in @($geoRoot, $runtimeRoot, $geosdpRoot)) {
    if (-not (Test-Path $path)) { throw "Required checkout not found: $path" }
    git -C $path diff --quiet
    if ($LASTEXITCODE -ne 0) { throw "Working tree in $path has unstaged changes." }
    git -C $path diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { throw "Working tree in $path has staged uncommitted changes." }
}

$env:GEO_ROOT = $geoRoot
$env:TORCH_CUDA_ARCH_LIST = $CudaArch
$env:FORCE_CUDA = "1"
$env:DISTUTILS_USE_SDK = "1"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"
$env:NVCC_APPEND_FLAGS = "-allow-unsupported-compiler"
$env:MAX_JOBS = [Math]::Max(1, [Environment]::ProcessorCount - 1).ToString()

Invoke-Logged "NVIDIA driver and GPU" { nvidia-smi }
Invoke-Logged "CUDA compiler" { nvcc --version }
Invoke-Logged "Python and PyTorch CUDA preflight" {
    & $Python -c @'
import json, platform, torch
info = {
    'python': platform.python_version(),
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'device_count': torch.cuda.device_count(),
}
if torch.cuda.is_available():
    info.update({
        'device_name': torch.cuda.get_device_name(0),
        'device_capability': torch.cuda.get_device_capability(0),
    })
print(json.dumps(info, indent=2))
assert torch.cuda.is_available(), 'PyTorch cannot access CUDA'
major, minor = torch.cuda.get_device_capability(0)
assert (major, minor) == (12, 0), f'Expected RTX 5070 capability (12, 0), got {(major, minor)}'
'@
}

Invoke-Logged "Record repository revisions" {
    foreach ($repo in @($geoRoot, $runtimeRoot, $geosdpRoot)) {
        Write-Host "Repo: $repo"
        git -C $repo remote get-url origin
        git -C $repo rev-parse HEAD
        git -C $repo status --porcelain
    }
}

if (-not $SkipGeoHostBuild) {
    $geoBuild = Join-Path $artifactRoot "geo-host-build"
    Invoke-Logged "Configure GEO deterministic host build" {
        cmake -S $geoRoot -B $geoBuild -DGEO_BUILD_TESTS=ON -DGEO_BUILD_BENCHMARKS=OFF -DGEO_BUILD_TOOLS=OFF -DGEO_USE_DOUBLE=OFF
    }
    Invoke-Logged "Build GEO deterministic host tests" { cmake --build $geoBuild --config Release --parallel }
    Invoke-Logged "Run GEO deterministic host tests" { ctest --test-dir $geoBuild -C Release --output-on-failure }
}

Push-Location $runtimeRoot
try {
    Invoke-Logged "Install runtime build dependencies" {
        & $Python -m pip install -U setuptools wheel ninja pytest
    }
    Invoke-Logged "Build and install CUDA runtime" {
        & $Python -m pip install -e ".[dev]" --no-build-isolation --no-deps
    }
    Invoke-Logged "Verify runtime capability contract" {
        & $Python -c "import geo_dl_runtime as r; print(sorted(r.GEO_CAPABILITIES)); r.require_stage('training'); print('training stage: READY')"
    }
    Invoke-Logged "Run complete runtime test suite" { & $Python -m pytest -q }
    Invoke-Logged "Run CUDA parity tests only" { & $Python -m pytest -q -k device }
    Invoke-Logged "Run native tests with verbose failures" { & $Python -m pytest -q -k test --maxfail=1 }
} finally {
    Pop-Location
}

Push-Location $geosdpRoot
try {
    Invoke-Logged "Install GEOSDP" { & $Python -m pip install -e ".[dev]" --no-deps }
    Invoke-Logged "Run GEOSDP tests" { & $Python -m pytest -q }
    Invoke-Logged "Run end-to-end native training test" { & $Python -m pytest -q tests/test_native_training.py -s }
    Invoke-Logged "Run two-step CUDA training smoke" {
        & $Python -m geosdp.train --config configs/smoke.yaml --backend native --smoke
    }
} finally {
    Pop-Location
}

Invoke-Logged "Post-test CUDA synchronization and memory report" {
    & $Python -c @'
import json, torch
assert torch.cuda.is_available()
torch.cuda.synchronize()
print(json.dumps({
  'allocated_bytes': torch.cuda.memory_allocated(),
  'reserved_bytes': torch.cuda.memory_reserved(),
  'max_allocated_bytes': torch.cuda.max_memory_allocated(),
  'max_reserved_bytes': torch.cuda.max_memory_reserved(),
}, indent=2))
'@
}

$logHash = (Get-FileHash -Path $logPath -Algorithm SHA256).Hash

$summary = [ordered]@{
    status = "PASS"
    timestamp = (Get-Date).ToString("o")
    workspace = $Workspace
    geo_commit = (git -C $geoRoot rev-parse HEAD)
    runtime_commit = (git -C $runtimeRoot rev-parse HEAD)
    geosdp_commit = (git -C $geosdpRoot rev-parse HEAD)
    cuda_arch = $CudaArch
    log = $logPath
    log_sha256 = $logHash
}
$summary | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $artifactRoot "summary.json")
Write-Host "CUDA acceptance PASS. Artifacts: $artifactRoot"
