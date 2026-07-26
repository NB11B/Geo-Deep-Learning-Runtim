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
    const std::vector<torch::Tensor> &parameters,
    const std::vector<torch::Tensor> &gradients,
    const std::vector<torch::Tensor> &first_moments,
    const std::vector<torch::Tensor> &second_moments,
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
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(
            parameters[0].get_device()
        ).stream();

        const size_t num_tensors = parameters.size();
        std::vector<float *> h_params(num_tensors);
        std::vector<const float *> h_grads(num_tensors);
        std::vector<float *> h_m1(num_tensors);
        std::vector<float *> h_m2(num_tensors);
        std::vector<size_t> h_counts(num_tensors);

        for (size_t i = 0u; i < num_tensors; ++i) {
            h_params[i] = parameters[i].data_ptr<float>();
            h_grads[i] = gradients[i].data_ptr<float>();
            h_m1[i] = first_moments[i].data_ptr<float>();
            h_m2[i] = second_moments[i].data_ptr<float>();
            h_counts[i] = static_cast<size_t>(parameters[i].numel());
        }

        struct DevicePointerCache {
            float **d_params = nullptr;
            const float **d_grads = nullptr;
            float **d_m1 = nullptr;
            float **d_m2 = nullptr;
            size_t *d_counts = nullptr;
            size_t capacity = 0u;

            void reserve(size_t n, cudaStream_t stream) {
                if (n > capacity) {
                    if (d_params) cudaFreeAsync(d_params, stream);
                    if (d_grads) cudaFreeAsync(const_cast<float**>(d_grads), stream);
                    if (d_m1) cudaFreeAsync(d_m1, stream);
                    if (d_m2) cudaFreeAsync(d_m2, stream);
                    if (d_counts) cudaFreeAsync(d_counts, stream);

                    cudaMallocAsync(&d_params, n * sizeof(float *), stream);
                    cudaMallocAsync(&d_grads, n * sizeof(float *), stream);
                    cudaMallocAsync(&d_m1, n * sizeof(float *), stream);
                    cudaMallocAsync(&d_m2, n * sizeof(float *), stream);
                    cudaMallocAsync(&d_counts, n * sizeof(size_t), stream);
                    capacity = n;
                }
            }
        };

        thread_local DevicePointerCache cache;
        cache.reserve(num_tensors, stream);

        cudaMemcpyAsync(cache.d_params, h_params.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(const_cast<float**>(cache.d_grads), h_grads.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cache.d_m1, h_m1.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cache.d_m2, h_m2.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cache.d_counts, h_counts.data(), num_tensors * sizeof(size_t), cudaMemcpyHostToDevice, stream);

        if (max_grad_norm > 0.0f) {
            torch::Tensor sum_square = torch::zeros({}, parameters[0].options());
            check_status(
                geo_tensor_grad_square_cuda_accumulate_fused(
                    cache.d_grads,
                    cache.d_counts,
                    num_tensors,
                    sum_square.data_ptr<float>(),
                    reinterpret_cast<void *>(stream)
                ),
                "geo_tensor_grad_square_cuda_accumulate_fused"
            );
            check_status(
                geo_tensor_grad_clip_cuda_finalize(
                    sum_square.data_ptr<float>(),
                    static_cast<float>(max_grad_norm),
                    clip_scale.data_ptr<float>(),
                    reinterpret_cast<void *>(stream)
                ),
                "geo_tensor_grad_clip_cuda_finalize"
            );
        }

        const float *clip_ptr = (max_grad_norm > 0.0f) ? clip_scale.data_ptr<float>() : nullptr;

        check_status(
            geo_tensor_adamw_cuda_step_fused(
                cache.d_params,
                cache.d_grads,
                cache.d_m1,
                cache.d_m2,
                cache.d_counts,
                num_tensors,
                clip_ptr,
                config,
                reinterpret_cast<void *>(stream)
            ),
            "geo_tensor_adamw_cuda_step_fused"
        );
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

#ifdef WITH_CUDA
std::map<std::string, double> geo_adamw_step_decomposed_profile(
    const std::vector<torch::Tensor> &parameters,
    const std::vector<torch::Tensor> &gradients,
    const std::vector<torch::Tensor> &first_moments,
    const std::vector<torch::Tensor> &second_moments,
    std::uint64_t step,
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon,
    double weight_decay,
    double max_grad_norm
) {
    auto t_cpp_start = std::chrono::high_resolution_clock::now();
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
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(parameters[0].get_device()).stream();

    cudaEvent_t ev_clip_start, ev_clip_end, ev_kernel_start, ev_kernel_end;
    cudaEventCreate(&ev_clip_start);
    cudaEventCreate(&ev_clip_end);
    cudaEventCreate(&ev_kernel_start);
    cudaEventCreate(&ev_kernel_end);

    const size_t num_tensors = parameters.size();

    auto t_extract_start = std::chrono::high_resolution_clock::now();
    std::vector<float *> h_params(num_tensors);
    std::vector<const float *> h_grads(num_tensors);
    std::vector<float *> h_m1(num_tensors);
    std::vector<float *> h_m2(num_tensors);
    std::vector<size_t> h_counts(num_tensors);

    for (size_t i = 0u; i < num_tensors; ++i) {
        h_params[i] = parameters[i].data_ptr<float>();
        h_grads[i] = gradients[i].data_ptr<float>();
        h_m1[i] = first_moments[i].data_ptr<float>();
        h_m2[i] = second_moments[i].data_ptr<float>();
        h_counts[i] = static_cast<size_t>(parameters[i].numel());
    }
    auto t_extract_end = std::chrono::high_resolution_clock::now();

    struct DevicePointerCache {
        float **d_params = nullptr;
        const float **d_grads = nullptr;
        float **d_m1 = nullptr;
        float **d_m2 = nullptr;
        size_t *d_counts = nullptr;
        size_t capacity = 0u;

        void reserve(size_t n, cudaStream_t stream) {
            if (n > capacity) {
                if (d_params) cudaFreeAsync(d_params, stream);
                if (d_grads) cudaFreeAsync(const_cast<float**>(d_grads), stream);
                if (d_m1) cudaFreeAsync(d_m1, stream);
                if (d_m2) cudaFreeAsync(d_m2, stream);
                if (d_counts) cudaFreeAsync(d_counts, stream);

                cudaMallocAsync(&d_params, n * sizeof(float *), stream);
                cudaMallocAsync(&d_grads, n * sizeof(float *), stream);
                cudaMallocAsync(&d_m1, n * sizeof(float *), stream);
                cudaMallocAsync(&d_m2, n * sizeof(float *), stream);
                cudaMallocAsync(&d_counts, n * sizeof(size_t), stream);
                capacity = n;
            }
        }
    };

    thread_local DevicePointerCache cache;
    cache.reserve(num_tensors, stream);

    auto t_memcpy_start = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(cache.d_params, h_params.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(const_cast<float**>(cache.d_grads), h_grads.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(cache.d_m1, h_m1.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(cache.d_m2, h_m2.data(), num_tensors * sizeof(float *), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(cache.d_counts, h_counts.data(), num_tensors * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    auto t_memcpy_end = std::chrono::high_resolution_clock::now();

    cudaEventRecord(ev_clip_start, stream);
    if (max_grad_norm > 0.0f) {
        torch::Tensor sum_square = torch::zeros({}, parameters[0].options());
        check_status(
            geo_tensor_grad_square_cuda_accumulate_fused(
                cache.d_grads,
                cache.d_counts,
                num_tensors,
                sum_square.data_ptr<float>(),
                reinterpret_cast<void *>(stream)
            ),
            "geo_tensor_grad_square_cuda_accumulate_fused"
        );
        check_status(
            geo_tensor_grad_clip_cuda_finalize(
                sum_square.data_ptr<float>(),
                static_cast<float>(max_grad_norm),
                clip_scale.data_ptr<float>(),
                reinterpret_cast<void *>(stream)
            ),
            "geo_tensor_grad_clip_cuda_finalize"
        );
    }
    cudaEventRecord(ev_clip_end, stream);

    const float *clip_ptr = (max_grad_norm > 0.0f) ? clip_scale.data_ptr<float>() : nullptr;

    cudaEventRecord(ev_kernel_start, stream);
    check_status(
        geo_tensor_adamw_cuda_step_fused(
            cache.d_params,
            cache.d_grads,
            cache.d_m1,
            cache.d_m2,
            cache.d_counts,
            num_tensors,
            clip_ptr,
            config,
            reinterpret_cast<void *>(stream)
        ),
        "geo_tensor_adamw_cuda_step_fused"
    );
    cudaEventRecord(ev_kernel_end, stream);

    cudaStreamSynchronize(stream);
    auto t_cpp_end = std::chrono::high_resolution_clock::now();

    float clip_ms = 0.0f;
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&clip_ms, ev_clip_start, ev_clip_end);
    cudaEventElapsedTime(&kernel_ms, ev_kernel_start, ev_kernel_end);

    cudaEventDestroy(ev_clip_start);
    cudaEventDestroy(ev_clip_end);
    cudaEventDestroy(ev_kernel_start);
    cudaEventDestroy(ev_kernel_end);

    double cpp_total_ms = std::chrono::duration<double, std::milli>(t_cpp_end - t_cpp_start).count();
    double extract_ms = std::chrono::duration<double, std::milli>(t_extract_end - t_extract_start).count();
    double memcpy_ms = std::chrono::duration<double, std::milli>(t_memcpy_end - t_memcpy_start).count();

    std::map<std::string, double> profile;
    profile["clip_calc_cuda_ms"] = static_cast<double>(clip_ms);
    profile["pure_cuda_kernel_ms"] = static_cast<double>(kernel_ms);
    profile["host_extract_ms"] = extract_ms;
    profile["host_memcpy_ms"] = memcpy_ms;
    profile["cpp_bridge_total_ms"] = cpp_total_ms;
    return profile;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("adamw_step", &geo_adamw_step);
#ifdef WITH_CUDA
    module.def("adamw_step_decomposed_profile", &geo_adamw_step_decomposed_profile);
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple("adamw");
}
