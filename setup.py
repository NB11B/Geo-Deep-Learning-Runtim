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
    GEO_ROOT / "src" / "tensor_linear.c",
    GEO_ROOT / "src" / "tensor_linear_cuda.cu",
    GEO_ROOT / "src" / "tensor_core.c",
    GEO_ROOT / "src" / "tensor_core_cuda.cu",
    GEO_ROOT / "src" / "tensor_activation.c",
    GEO_ROOT / "src" / "tensor_activation_cuda.cu",
    GEO_ROOT / "src" / "tensor_rope.c",
    GEO_ROOT / "src" / "tensor_rope_cuda.cu",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise RuntimeError(
        "GEO_ROOT must reference a GeometricElementaryOperators checkout containing the tensor runtime. "
        f"Missing: {', '.join(missing)}"
    )

common_macros = [("WITH_CUDA", "1"), ("GEO_REAL_IS_DOUBLE", "0")]
common_compile_args = {
    "cxx": ["-O3"],
    "nvcc": ["-O3", "--use_fast_math", "-lineinfo"],
}

setup(
    ext_modules=[
        CUDAExtension(
            name="geo_dl_runtime._C",
            sources=[
                str(ROOT / "native" / "geo_dl_bridge.cpp"),
                str(GEO_ROOT / "src" / "tensor_linear.c"),
                str(GEO_ROOT / "src" / "tensor_linear_cuda.cu"),
                str(GEO_ROOT / "src" / "tensor_core.c"),
                str(GEO_ROOT / "src" / "tensor_core_cuda.cu"),
                str(GEO_ROOT / "src" / "tensor_activation.c"),
                str(GEO_ROOT / "src" / "tensor_activation_cuda.cu"),
            ],
            include_dirs=[str(GEO_ROOT / "include")],
            define_macros=common_macros,
            extra_compile_args=common_compile_args,
        ),
        CUDAExtension(
            name="geo_dl_runtime._rope",
            sources=[
                str(ROOT / "native" / "geo_rope_bridge.cpp"),
                str(GEO_ROOT / "src" / "tensor_linear.c"),
                str(GEO_ROOT / "src" / "tensor_rope.c"),
                str(GEO_ROOT / "src" / "tensor_rope_cuda.cu"),
            ],
            include_dirs=[str(GEO_ROOT / "include")],
            define_macros=common_macros,
            extra_compile_args=common_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
)
