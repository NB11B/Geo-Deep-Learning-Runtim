from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
GEO_ROOT = Path(os.environ.get("GEO_ROOT", ROOT.parent / "GeometricElementaryOperators")).resolve()

required = [
    GEO_ROOT / "include" / "geo" / "tensor_linear.h",
    GEO_ROOT / "include" / "geo" / "tensor_linear_cuda.h",
    GEO_ROOT / "include" / "geo" / "tensor_core.h",
    GEO_ROOT / "include" / "geo" / "tensor_core_cuda.h",
    GEO_ROOT / "include" / "geo" / "tensor_activation.h",
    GEO_ROOT / "include" / "geo" / "tensor_activation_cuda.h",
    GEO_ROOT / "include" / "geo" / "tensor_rope.h",
    GEO_ROOT / "include" / "geo" / "tensor_rope_cuda.h",
    GEO_ROOT / "include" / "geo" / "tensor_attention.h",
    GEO_ROOT / "include" / "geo" / "tensor_attention_cuda.h",
    GEO_ROOT / "include" / "geo" / "tensor_loss.h",
    GEO_ROOT / "include" / "geo" / "tensor_loss_cuda.h",
    GEO_ROOT / "include" / "geo" / "tensor_embedding.h",
    GEO_ROOT / "include" / "geo" / "tensor_embedding_cuda.h",
    GEO_ROOT / "include" / "geo" / "tensor_optimizer.h",
    GEO_ROOT / "include" / "geo" / "tensor_optimizer_cuda.h",
    GEO_ROOT / "src" / "tensor_linear.c",
    GEO_ROOT / "src" / "tensor_linear_cuda.cu",
    GEO_ROOT / "src" / "tensor_core.c",
    GEO_ROOT / "src" / "tensor_core_cuda.cu",
    GEO_ROOT / "src" / "tensor_activation.c",
    GEO_ROOT / "src" / "tensor_activation_cuda.cu",
    GEO_ROOT / "src" / "tensor_rope.c",
    GEO_ROOT / "src" / "tensor_rope_cuda.cu",
    GEO_ROOT / "src" / "tensor_attention.c",
    GEO_ROOT / "src" / "tensor_attention_cuda.cu",
    GEO_ROOT / "src" / "tensor_loss.c",
    GEO_ROOT / "src" / "tensor_loss_cuda.cu",
    GEO_ROOT / "src" / "tensor_embedding.c",
    GEO_ROOT / "src" / "tensor_embedding_cuda.cu",
    GEO_ROOT / "src" / "tensor_optimizer.c",
    GEO_ROOT / "src" / "tensor_optimizer_cuda.cu",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise RuntimeError(
        "GEO_ROOT must reference a GeometricElementaryOperators checkout containing the tensor runtime. "
        f"Missing: {', '.join(missing)}"
    )

common_macros = [("WITH_CUDA", "1"), ("GEO_REAL_IS_DOUBLE", "0")]
common_compile_args = {"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math", "-lineinfo"]}


def extension(name: str, bridge: str, cpu_sources: list[str], cuda_sources: list[str]) -> CUDAExtension:
    return CUDAExtension(
        name=name,
        sources=[
            str(ROOT / "native" / bridge),
            str(GEO_ROOT / "src" / "tensor_linear.c"),
            *[str(GEO_ROOT / "src" / source) for source in cpu_sources],
            *[str(GEO_ROOT / "src" / source) for source in cuda_sources],
        ],
        include_dirs=[str(GEO_ROOT / "include")],
        define_macros=common_macros,
        extra_compile_args=common_compile_args,
    )


setup(
    ext_modules=[
        extension(
            "geo_dl_runtime._C",
            "geo_dl_bridge.cpp",
            ["tensor_core.c", "tensor_activation.c"],
            ["tensor_linear_cuda.cu", "tensor_core_cuda.cu", "tensor_activation_cuda.cu"],
        ),
        extension("geo_dl_runtime._rope", "geo_rope_bridge.cpp", ["tensor_rope.c"], ["tensor_rope_cuda.cu"]),
        extension("geo_dl_runtime._attention", "geo_attention_bridge.cpp", ["tensor_attention.c"], ["tensor_attention_cuda.cu"]),
        extension("geo_dl_runtime._loss", "geo_loss_bridge.cpp", ["tensor_loss.c"], ["tensor_loss_cuda.cu"]),
        extension("geo_dl_runtime._embedding", "geo_embedding_bridge.cpp", ["tensor_embedding.c"], ["tensor_embedding_cuda.cu"]),
        extension("geo_dl_runtime._optimizer", "geo_optimizer_bridge.cpp", ["tensor_optimizer.c"], ["tensor_optimizer_cuda.cu"]),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
