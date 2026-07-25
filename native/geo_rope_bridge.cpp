#include <torch/extension.h>

#include <cstdint>
#include <string>
#include <vector>

#include "geo/tensor_rope.h"
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "geo/tensor_rope_cuda.h"
#endif

namespace {

void check_tensor(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_status(geo_tensor_status status, const char *operation) {
    TORCH_CHECK(status == GEO_TENSOR_OK, operation, " failed: ", geo_tensor_status_string(status));
}

#ifdef WITH_CUDA
void *current_stream(const torch::Tensor &tensor) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    return reinterpret_cast<void *>(stream);
}
#endif

geo_tensor_rope_apply_shape make_apply_shape(
    const torch::Tensor &x,
    const torch::Tensor &cos_table,
    const torch::Tensor &sin_table
) {
    TORCH_CHECK(x.dim() >= 2, "RoPE input must have at least two dimensions");
    TORCH_CHECK(cos_table.dim() == 2 && sin_table.dim() == 2,
                "RoPE cosine and sine tables must be rank two");
    TORCH_CHECK(cos_table.sizes() == sin_table.sizes(),
                "RoPE cosine and sine table shapes must match");
    TORCH_CHECK(x.device() == cos_table.device() && x.device() == sin_table.device(),
                "RoPE input and tables must share a device");

    const std::int64_t tokens = x.size(-2);
    const std::int64_t head_dim = x.size(-1);
    TORCH_CHECK(tokens > 0 && head_dim > 0 && (head_dim % 2) == 0,
                "RoPE requires positive tokens and an even head dimension");
    TORCH_CHECK(cos_table.size(0) >= tokens,
                "RoPE table sequence length is shorter than the input");
    TORCH_CHECK(cos_table.size(1) * 2 == head_dim,
                "RoPE table width must equal half the input head dimension");

    const std::int64_t row_width = tokens * head_dim;
    TORCH_CHECK(row_width > 0 && x.numel() % row_width == 0,
                "RoPE input cannot be flattened into [outer, tokens, head_dim]");
    const std::int64_t outer = x.numel() / row_width;
    TORCH_CHECK(outer > 0, "RoPE outer dimension must be positive");

    return {
        static_cast<size_t>(outer),
        static_cast<size_t>(tokens),
        static_cast<size_t>(head_dim),
        static_cast<size_t>(cos_table.size(0)),
    };
}

}  // namespace

std::vector<torch::Tensor> geo_rope_build(
    std::int64_t seq_len,
    std::int64_t head_dim,
    double theta,
    const std::string &device_string
) {
    TORCH_CHECK(seq_len > 0, "RoPE sequence length must be positive");
    TORCH_CHECK(head_dim > 0 && (head_dim % 2) == 0,
                "RoPE head dimension must be positive and even");
    TORCH_CHECK(theta > 0.0, "RoPE theta must be positive");

    const c10::Device device(device_string);
    TORCH_CHECK(device.is_cpu() || device.is_cuda(),
                "RoPE supports only CPU and CUDA devices");
    const auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    torch::Tensor cos_table = torch::empty({seq_len, head_dim / 2}, options);
    torch::Tensor sin_table = torch::empty({seq_len, head_dim / 2}, options);
    const geo_tensor_rope_table_shape shape = {
        static_cast<size_t>(seq_len),
        static_cast<size_t>(head_dim),
    };

    if (cos_table.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_rope_cuda_build(
                static_cast<float>(theta),
                cos_table.data_ptr<float>(),
                sin_table.data_ptr<float>(),
                shape,
                current_stream(cos_table)
            ),
            "geo_tensor_rope_cuda_build"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_rope_build(
                static_cast<float>(theta),
                cos_table.data_ptr<float>(),
                sin_table.data_ptr<float>(),
                shape
            ),
            "geo_tensor_rope_build"
        );
    }
    return {cos_table, sin_table};
}

torch::Tensor geo_rope_apply_forward(
    torch::Tensor x,
    torch::Tensor cos_table,
    torch::Tensor sin_table
) {
    check_tensor(x, "x");
    check_tensor(cos_table, "cos_table");
    check_tensor(sin_table, "sin_table");
    const auto shape = make_apply_shape(x, cos_table, sin_table);
    torch::Tensor out = torch::empty_like(x);

    if (x.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_rope_cuda_apply_forward(
                x.data_ptr<float>(),
                cos_table.data_ptr<float>(),
                sin_table.data_ptr<float>(),
                out.data_ptr<float>(),
                shape,
                current_stream(x)
            ),
            "geo_tensor_rope_cuda_apply_forward"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_rope_apply_forward(
                x.data_ptr<float>(),
                cos_table.data_ptr<float>(),
                sin_table.data_ptr<float>(),
                out.data_ptr<float>(),
                shape
            ),
            "geo_tensor_rope_apply_forward"
        );
    }
    return out;
}

torch::Tensor geo_rope_apply_backward(
    torch::Tensor grad_output,
    torch::Tensor cos_table,
    torch::Tensor sin_table
) {
    check_tensor(grad_output, "grad_output");
    check_tensor(cos_table, "cos_table");
    check_tensor(sin_table, "sin_table");
    const auto shape = make_apply_shape(grad_output, cos_table, sin_table);
    torch::Tensor grad_x = torch::empty_like(grad_output);

    if (grad_output.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_rope_cuda_apply_vjp(
                grad_output.data_ptr<float>(),
                cos_table.data_ptr<float>(),
                sin_table.data_ptr<float>(),
                grad_x.data_ptr<float>(),
                shape,
                current_stream(grad_output)
            ),
            "geo_tensor_rope_cuda_apply_vjp"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_rope_apply_vjp(
                grad_output.data_ptr<float>(),
                cos_table.data_ptr<float>(),
                sin_table.data_ptr<float>(),
                grad_x.data_ptr<float>(),
                shape
            ),
            "geo_tensor_rope_apply_vjp"
        );
    }
    return grad_x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("build", &geo_rope_build);
    module.def("apply_forward", &geo_rope_apply_forward);
    module.def("apply_backward", &geo_rope_apply_backward);
    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple("build_rope", "apply_rope");
#ifdef WITH_CUDA
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
}
