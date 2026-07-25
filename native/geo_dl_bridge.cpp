#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "geo/tensor_linear.h"
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "geo/tensor_linear_cuda.h"
#endif

namespace {

void check_tensor(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

geo_tensor_linear_shape make_shape(const torch::Tensor &x, const torch::Tensor &weight) {
    TORCH_CHECK(x.dim() >= 2, "x must have at least two dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must be rank two");
    TORCH_CHECK(x.size(-1) == weight.size(1), "linear feature dimensions do not match");

    std::int64_t rows = 1;
    for (std::int64_t dim = 0; dim < x.dim() - 1; ++dim) {
        TORCH_CHECK(x.size(dim) > 0, "x dimensions must be nonzero");
        TORCH_CHECK(rows <= INT64_MAX / x.size(dim), "flattened row count overflows int64");
        rows *= x.size(dim);
    }
    TORCH_CHECK(rows > 0 && weight.size(0) > 0 && weight.size(1) > 0,
                "linear dimensions must be nonzero");
    return geo_tensor_linear_shape{
        static_cast<size_t>(rows),
        static_cast<size_t>(weight.size(1)),
        static_cast<size_t>(weight.size(0))
    };
}

void check_status(geo_tensor_status status, const char *operation) {
    TORCH_CHECK(status == GEO_TENSOR_OK, operation, " failed: ", geo_tensor_status_string(status));
}

}  // namespace

torch::Tensor geo_linear_forward(torch::Tensor x, torch::Tensor weight) {
    check_tensor(x, "x");
    check_tensor(weight, "weight");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");
    const geo_tensor_linear_shape shape = make_shape(x, weight);

    std::vector<std::int64_t> output_shape(x.sizes().begin(), x.sizes().end());
    output_shape.back() = weight.size(0);
    torch::Tensor output = torch::empty(output_shape, x.options());

    if (x.is_cuda()) {
#ifdef WITH_CUDA
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
        check_status(
            geo_tensor_linear_cuda_forward(
                x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), shape,
                reinterpret_cast<void *>(stream)
            ),
            "geo_tensor_linear_cuda_forward"
        );
#else
        TORCH_CHECK(false, "GEO deep-learning runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_linear_forward(
                x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), shape
            ),
            "geo_tensor_linear_forward"
        );
    }
    return output;
}

std::vector<torch::Tensor> geo_linear_backward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor grad_output
) {
    check_tensor(x, "x");
    check_tensor(weight, "weight");
    check_tensor(grad_output, "grad_output");
    TORCH_CHECK(x.device() == weight.device() && x.device() == grad_output.device(),
                "all tensors must be on the same device");
    const geo_tensor_linear_shape shape = make_shape(x, weight);
    TORCH_CHECK(grad_output.numel() == static_cast<std::int64_t>(shape.rows * shape.out_features),
                "grad_output has the wrong number of elements");

    torch::Tensor grad_x = torch::empty_like(x);
    torch::Tensor grad_weight = torch::empty_like(weight);

    if (x.is_cuda()) {
#ifdef WITH_CUDA
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
        check_status(
            geo_tensor_linear_cuda_vjp(
                x.data_ptr<float>(), weight.data_ptr<float>(), grad_output.data_ptr<float>(),
                grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), shape,
                reinterpret_cast<void *>(stream)
            ),
            "geo_tensor_linear_cuda_vjp"
        );
#else
        TORCH_CHECK(false, "GEO deep-learning runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_linear_vjp(
                x.data_ptr<float>(), weight.data_ptr<float>(), grad_output.data_ptr<float>(),
                grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), shape
            ),
            "geo_tensor_linear_vjp"
        );
    }
    return {grad_x, grad_weight};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("linear_forward", &geo_linear_forward, "GEO tensor linear forward");
    module.def("linear_backward", &geo_linear_backward, "GEO tensor linear VJP");
    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple("linear");
#ifdef WITH_CUDA
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
}
