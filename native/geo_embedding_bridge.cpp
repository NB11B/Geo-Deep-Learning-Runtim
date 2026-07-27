#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "geo/tensor_embedding.h"
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "geo/tensor_embedding_cuda.h"
#endif

namespace {

void check_weight(const torch::Tensor &weight) {
    TORCH_CHECK(weight.defined(), "embedding weight must be defined");
    TORCH_CHECK(weight.is_contiguous(), "embedding weight must be contiguous");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32,
                "embedding weight must be float32");
    TORCH_CHECK(weight.dim() == 2, "embedding weight must be rank two");
    TORCH_CHECK(weight.size(0) > 0 && weight.size(1) > 0,
                "embedding dimensions must be positive");
}

void check_indices(const torch::Tensor &indices) {
    TORCH_CHECK(indices.defined(), "embedding indices must be defined");
    TORCH_CHECK(indices.is_contiguous(), "embedding indices must be contiguous");
    TORCH_CHECK(indices.scalar_type() == torch::kInt64,
                "embedding indices must be int64");
    TORCH_CHECK(indices.numel() > 0, "embedding indices must be nonempty");
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

geo_tensor_embedding_shape make_shape(
    const torch::Tensor &indices,
    const torch::Tensor &weight
) {
    TORCH_CHECK(indices.device() == weight.device(),
                "embedding indices and weight must share a device");
    return {
        static_cast<size_t>(indices.numel()),
        static_cast<size_t>(weight.size(0)),
        static_cast<size_t>(weight.size(1)),
    };
}

}  // namespace

torch::Tensor geo_embedding_forward(
    torch::Tensor indices,
    torch::Tensor weight
) {
    check_indices(indices);
    check_weight(weight);
    const auto shape = make_shape(indices, weight);
    std::vector<std::int64_t> output_shape(
        indices.sizes().begin(), indices.sizes().end()
    );
    output_shape.push_back(weight.size(1));
    torch::Tensor output = torch::empty(output_shape, weight.options());

    if (weight.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_embedding_cuda_forward(
                indices.data_ptr<std::int64_t>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                shape,
                current_stream(weight)
            ),
            "geo_tensor_embedding_cuda_forward"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_embedding_forward(
                indices.data_ptr<std::int64_t>(),
                weight.data_ptr<float>(),
                output.data_ptr<float>(),
                shape
            ),
            "geo_tensor_embedding_forward"
        );
    }
    return output;
}

torch::Tensor geo_embedding_backward(
    torch::Tensor indices,
    torch::Tensor grad_output,
    std::int64_t vocabulary,
    std::int64_t dimension
) {
    check_indices(indices);
    TORCH_CHECK(grad_output.defined(), "embedding grad_output must be defined");
    TORCH_CHECK(grad_output.is_contiguous(), "embedding grad_output must be contiguous");
    TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32,
                "embedding grad_output must be float32");
    TORCH_CHECK(vocabulary > 0 && dimension > 0,
                "embedding dimensions must be positive");
    TORCH_CHECK(grad_output.device() == indices.device(),
                "embedding indices and grad_output must share a device");
    TORCH_CHECK(
        grad_output.numel() == indices.numel() * dimension,
        "embedding grad_output shape mismatch"
    );

    const geo_tensor_embedding_shape shape = {
        static_cast<size_t>(indices.numel()),
        static_cast<size_t>(vocabulary),
        static_cast<size_t>(dimension),
    };
    torch::Tensor grad_weight = torch::empty(
        {vocabulary, dimension}, grad_output.options()
    );

    if (grad_output.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_embedding_cuda_vjp(
                indices.data_ptr<std::int64_t>(),
                grad_output.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                shape,
                current_stream(grad_output)
            ),
            "geo_tensor_embedding_cuda_vjp"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_embedding_vjp(
                indices.data_ptr<std::int64_t>(),
                grad_output.data_ptr<float>(),
                grad_weight.data_ptr<float>(),
                shape
            ),
            "geo_tensor_embedding_vjp"
        );
    }
    return grad_weight;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &geo_embedding_forward);
    module.def("backward", &geo_embedding_backward);
    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple("embedding");
#ifdef WITH_CUDA
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
}
