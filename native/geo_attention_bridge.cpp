#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "geo/tensor_attention.h"
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "geo/tensor_attention_cuda.h"
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

geo_tensor_attention_shape make_attention_shape(
    const torch::Tensor &q,
    const torch::Tensor &k,
    const torch::Tensor &v
) {
    TORCH_CHECK(q.dim() >= 2, "attention tensors must have at least two dimensions");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "q, k, and v shapes must match");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, and v must share a device");
    const std::int64_t tokens = q.size(-2);
    const std::int64_t head_dim = q.size(-1);
    TORCH_CHECK(tokens > 0 && head_dim > 0,
                "attention token and head dimensions must be positive");
    TORCH_CHECK(tokens <= INT64_MAX / head_dim,
                "attention row width overflows int64");
    const std::int64_t row_width = tokens * head_dim;
    TORCH_CHECK(q.numel() % row_width == 0,
                "attention tensors cannot be flattened into [outer, tokens, head_dim]");
    const std::int64_t outer = q.numel() / row_width;
    TORCH_CHECK(outer > 0, "attention outer dimension must be positive");
    return {
        static_cast<size_t>(outer),
        static_cast<size_t>(tokens),
        static_cast<size_t>(head_dim),
    };
}

}  // namespace

std::vector<torch::Tensor> geo_attention_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
) {
    check_tensor(q, "q");
    check_tensor(k, "k");
    check_tensor(v, "v");
    const auto shape = make_attention_shape(q, k, v);
    torch::Tensor out = torch::empty_like(q);
    torch::Tensor probabilities = torch::empty(
        {
            static_cast<std::int64_t>(shape.outer),
            static_cast<std::int64_t>(shape.tokens),
            static_cast<std::int64_t>(shape.tokens),
        },
        q.options()
    );

    if (q.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_causal_attention_cuda_forward(
                q.data_ptr<float>(),
                k.data_ptr<float>(),
                v.data_ptr<float>(),
                out.data_ptr<float>(),
                probabilities.data_ptr<float>(),
                shape,
                current_stream(q)
            ),
            "geo_tensor_causal_attention_cuda_forward"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_causal_attention_forward(
                q.data_ptr<float>(),
                k.data_ptr<float>(),
                v.data_ptr<float>(),
                out.data_ptr<float>(),
                probabilities.data_ptr<float>(),
                shape
            ),
            "geo_tensor_causal_attention_forward"
        );
    }
    return {out, probabilities};
}

std::vector<torch::Tensor> geo_attention_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor probabilities,
    torch::Tensor grad_output
) {
    check_tensor(q, "q");
    check_tensor(k, "k");
    check_tensor(v, "v");
    check_tensor(probabilities, "probabilities");
    check_tensor(grad_output, "grad_output");
    const auto shape = make_attention_shape(q, k, v);
    TORCH_CHECK(grad_output.sizes() == q.sizes(),
                "attention grad_output shape mismatch");
    TORCH_CHECK(grad_output.device() == q.device(),
                "attention grad_output device mismatch");
    TORCH_CHECK(
        probabilities.dim() == 3 &&
        probabilities.size(0) == static_cast<std::int64_t>(shape.outer) &&
        probabilities.size(1) == static_cast<std::int64_t>(shape.tokens) &&
        probabilities.size(2) == static_cast<std::int64_t>(shape.tokens),
        "attention probability shape mismatch"
    );
    TORCH_CHECK(probabilities.device() == q.device(),
                "attention probability device mismatch");

    torch::Tensor grad_q = torch::empty_like(q);
    torch::Tensor grad_k = torch::empty_like(k);
    torch::Tensor grad_v = torch::empty_like(v);

    if (q.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_causal_attention_cuda_vjp(
                q.data_ptr<float>(),
                k.data_ptr<float>(),
                v.data_ptr<float>(),
                probabilities.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                grad_q.data_ptr<float>(),
                grad_k.data_ptr<float>(),
                grad_v.data_ptr<float>(),
                shape,
                current_stream(q)
            ),
            "geo_tensor_causal_attention_cuda_vjp"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_causal_attention_vjp(
                q.data_ptr<float>(),
                k.data_ptr<float>(),
                v.data_ptr<float>(),
                probabilities.data_ptr<float>(),
                grad_output.data_ptr<float>(),
                grad_q.data_ptr<float>(),
                grad_k.data_ptr<float>(),
                grad_v.data_ptr<float>(),
                shape
            ),
            "geo_tensor_causal_attention_vjp"
        );
    }
    return {grad_q, grad_k, grad_v};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &geo_attention_forward);
    module.def("backward", &geo_attention_backward);
    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple("causal_attention");
#ifdef WITH_CUDA
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
}
