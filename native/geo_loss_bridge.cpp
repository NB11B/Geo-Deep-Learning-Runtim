#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "geo/tensor_loss.h"
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "geo/tensor_loss_cuda.h"
#endif

namespace {

void check_float_tensor(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_target_tensor(const torch::Tensor &tensor) {
    TORCH_CHECK(tensor.defined(), "targets must be defined");
    TORCH_CHECK(tensor.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kInt64, "targets must be int64");
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

geo_tensor_cross_entropy_shape make_loss_shape(
    const torch::Tensor &values,
    const torch::Tensor &targets
) {
    TORCH_CHECK(values.dim() == 2, "cross-entropy logits/probabilities must be rank two");
    TORCH_CHECK(targets.dim() == 1, "cross-entropy targets must be rank one");
    TORCH_CHECK(values.size(0) == targets.size(0),
                "cross-entropy row and target counts must match");
    TORCH_CHECK(values.size(0) > 0 && values.size(1) > 0,
                "cross-entropy dimensions must be positive");
    TORCH_CHECK(values.device() == targets.device(),
                "cross-entropy values and targets must share a device");
    return {
        static_cast<size_t>(values.size(0)),
        static_cast<size_t>(values.size(1)),
    };
}

}  // namespace

std::vector<torch::Tensor> geo_cross_entropy_forward(
    torch::Tensor logits,
    torch::Tensor targets,
    std::int64_t ignore_index
) {
    check_float_tensor(logits, "logits");
    check_target_tensor(targets);
    const auto shape = make_loss_shape(logits, targets);

    torch::Tensor loss = torch::empty({}, logits.options());
    torch::Tensor probabilities = torch::empty_like(logits);
    torch::Tensor normalizer = torch::empty({}, logits.options());

    if (logits.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_cross_entropy_cuda_forward(
                logits.data_ptr<float>(),
                targets.data_ptr<std::int64_t>(),
                ignore_index,
                loss.data_ptr<float>(),
                probabilities.data_ptr<float>(),
                normalizer.data_ptr<float>(),
                shape,
                current_stream(logits)
            ),
            "geo_tensor_cross_entropy_cuda_forward"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_cross_entropy_forward(
                logits.data_ptr<float>(),
                targets.data_ptr<std::int64_t>(),
                ignore_index,
                loss.data_ptr<float>(),
                probabilities.data_ptr<float>(),
                normalizer.data_ptr<float>(),
                shape
            ),
            "geo_tensor_cross_entropy_forward"
        );
    }
    return {loss, probabilities, normalizer};
}

torch::Tensor geo_cross_entropy_backward(
    torch::Tensor probabilities,
    torch::Tensor targets,
    std::int64_t ignore_index,
    torch::Tensor normalizer,
    torch::Tensor grad_loss
) {
    check_float_tensor(probabilities, "probabilities");
    check_target_tensor(targets);
    check_float_tensor(normalizer, "normalizer");
    check_float_tensor(grad_loss, "grad_loss");
    TORCH_CHECK(normalizer.numel() == 1, "normalizer must be scalar");
    TORCH_CHECK(grad_loss.numel() == 1, "grad_loss must be scalar");
    const auto shape = make_loss_shape(probabilities, targets);
    TORCH_CHECK(probabilities.device() == normalizer.device() &&
                probabilities.device() == grad_loss.device(),
                "cross-entropy backward tensors must share a device");

    torch::Tensor grad_logits = torch::empty_like(probabilities);
    if (probabilities.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_tensor_cross_entropy_cuda_vjp(
                probabilities.data_ptr<float>(),
                targets.data_ptr<std::int64_t>(),
                ignore_index,
                normalizer.data_ptr<float>(),
                grad_loss.data_ptr<float>(),
                grad_logits.data_ptr<float>(),
                shape,
                current_stream(probabilities)
            ),
            "geo_tensor_cross_entropy_cuda_vjp"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_tensor_cross_entropy_vjp(
                probabilities.data_ptr<float>(),
                targets.data_ptr<std::int64_t>(),
                ignore_index,
                normalizer.item<float>(),
                grad_loss.item<float>(),
                grad_logits.data_ptr<float>(),
                shape
            ),
            "geo_tensor_cross_entropy_vjp"
        );
    }
    return grad_logits;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("forward", &geo_cross_entropy_forward);
    module.def("backward", &geo_cross_entropy_backward);
    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple("cross_entropy");
#ifdef WITH_CUDA
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
}
