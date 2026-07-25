#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "geo/tensor_optimizer.h"
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "geo/tensor_optimizer_cuda.h"
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

void validate_lists(
    const std::vector<torch::Tensor> &parameters,
    const std::vector<torch::Tensor> &gradients,
    const std::vector<torch::Tensor> &first_moments,
    const std::vector<torch::Tensor> &second_moments
) {
    const size_t count = parameters.size();
    TORCH_CHECK(count > 0u, "AdamW requires at least one parameter");
    TORCH_CHECK(
        gradients.size() == count && first_moments.size() == count &&
        second_moments.size() == count,
        "AdamW parameter, gradient, and state list lengths must match"
    );
    const torch::Device device = parameters[0].device();
    for (size_t index = 0u; index < count; ++index) {
        check_tensor(parameters[index], "parameter");
        check_tensor(gradients[index], "gradient");
        check_tensor(first_moments[index], "first_moment");
        check_tensor(second_moments[index], "second_moment");
        TORCH_CHECK(
            parameters[index].sizes() == gradients[index].sizes() &&
            parameters[index].sizes() == first_moments[index].sizes() &&
            parameters[index].sizes() == second_moments[index].sizes(),
            "AdamW parameter, gradient, and state shapes must match"
        );
        TORCH_CHECK(
            parameters[index].device() == device &&
            gradients[index].device() == device &&
            first_moments[index].device() == device &&
            second_moments[index].device() == device,
            "all AdamW tensors must share one device"
        );
    }
}

}  // namespace

torch::Tensor geo_adamw_step(
    std::vector<torch::Tensor> parameters,
    std::vector<torch::Tensor> gradients,
    std::vector<torch::Tensor> first_moments,
    std::vector<torch::Tensor> second_moments,
    std::uint64_t step,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    double weight_decay,
    double max_grad_norm
) {
    validate_lists(parameters, gradients, first_moments, second_moments);
    TORCH_CHECK(step > 0u, "AdamW step must be positive");
    const geo_tensor_adamw_config config = {
        static_cast<float>(learning_rate),
        static_cast<float>(beta1),
        static_cast<float>(beta2),
        static_cast<float>(epsilon),
        static_cast<float>(weight_decay),
        static_cast<float>(max_grad_norm),
        step,
    };
    torch::Tensor clip_scale = torch::empty({}, parameters[0].options());

    if (parameters[0].is_cuda()) {
#ifdef WITH_CUDA
        torch::Tensor sum_square = torch::empty({}, parameters[0].options());
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(
            parameters[0].get_device()
        ).stream();
        TORCH_CHECK(
            cudaMemsetAsync(sum_square.data_ptr<float>(), 0, sizeof(float), stream) == cudaSuccess,
            "failed to initialize GEO AdamW gradient accumulator"
        );
        for (size_t index = 0u; index < parameters.size(); ++index) {
            check_status(
                geo_tensor_grad_square_cuda_accumulate(
                    gradients[index].data_ptr<float>(),
                    static_cast<size_t>(gradients[index].numel()),
                    sum_square.data_ptr<float>(),
                    reinterpret_cast<void *>(stream)
                ),
                "geo_tensor_grad_square_cuda_accumulate"
            );
        }
        check_status(
            geo_tensor_grad_clip_cuda_finalize(
                sum_square.data_ptr<float>(),
                static_cast<float>(max_grad_norm),
                clip_scale.data_ptr<float>(),
                reinterpret_cast<void *>(stream)
            ),
            "geo_tensor_grad_clip_cuda_finalize"
        );
        for (size_t index = 0u; index < parameters.size(); ++index) {
            check_status(
                geo_tensor_adamw_cuda_step(
                    parameters[index].data_ptr<float>(),
                    gradients[index].data_ptr<float>(),
                    first_moments[index].data_ptr<float>(),
                    second_moments[index].data_ptr<float>(),
                    static_cast<size_t>(parameters[index].numel()),
                    clip_scale.data_ptr<float>(),
                    config,
                    reinterpret_cast<void *>(stream)
                ),
                "geo_tensor_adamw_cuda_step"
            );
        }
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        float sum_square = 0.0f;
        for (size_t index = 0u; index < parameters.size(); ++index) {
            check_status(
                geo_tensor_grad_square_accumulate(
                    gradients[index].data_ptr<float>(),
                    static_cast<size_t>(gradients[index].numel()),
                    &sum_square
                ),
                "geo_tensor_grad_square_accumulate"
            );
        }
        float clip = 1.0f;
        check_status(
            geo_tensor_grad_clip_scale(
                sum_square, static_cast<float>(max_grad_norm), &clip
            ),
            "geo_tensor_grad_clip_scale"
        );
        *clip_scale.data_ptr<float>() = clip;
        for (size_t index = 0u; index < parameters.size(); ++index) {
            check_status(
                geo_tensor_adamw_step(
                    parameters[index].data_ptr<float>(),
                    gradients[index].data_ptr<float>(),
                    first_moments[index].data_ptr<float>(),
                    second_moments[index].data_ptr<float>(),
                    static_cast<size_t>(parameters[index].numel()),
                    clip,
                    config
                ),
                "geo_tensor_adamw_step"
            );
        }
    }
    return clip_scale;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("adamw_step", &geo_adamw_step);
    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple("adamw");
#ifdef WITH_CUDA
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
}
