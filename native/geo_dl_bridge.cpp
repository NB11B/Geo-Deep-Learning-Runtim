#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "geo/tensor_activation.h"
#include "geo/tensor_core.h"
#include "geo/tensor_linear.h"
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "geo/tensor_activation_cuda.h"
#include "geo/tensor_core_cuda.h"
#include "geo/tensor_linear_cuda.h"
#include "geo/geo_implicit_cuda.h"
#endif


namespace {

void check_tensor(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_same(const torch::Tensor &a, const torch::Tensor &b, const char *message) {
    TORCH_CHECK(a.device() == b.device(), message, ": device mismatch");
    TORCH_CHECK(a.sizes() == b.sizes(), message, ": shape mismatch");
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

geo_tensor_linear_shape make_linear_shape(const torch::Tensor &x, const torch::Tensor &weight) {
    TORCH_CHECK(x.dim() >= 2, "x must have at least two dimensions");
    TORCH_CHECK(weight.dim() == 2, "weight must be rank two");
    TORCH_CHECK(x.size(-1) == weight.size(1), "linear feature dimensions do not match");
    const std::int64_t rows = x.numel() / x.size(-1);
    TORCH_CHECK(rows > 0 && weight.size(0) > 0 && weight.size(1) > 0, "linear dimensions must be nonzero");
    return {static_cast<size_t>(rows), static_cast<size_t>(weight.size(1)), static_cast<size_t>(weight.size(0))};
}

geo_tensor_norm_shape make_norm_shape(const torch::Tensor &x, const torch::Tensor &weight) {
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    TORCH_CHECK(weight.dim() == 1, "RMSNorm weight must be rank one");
    TORCH_CHECK(x.size(-1) == weight.size(0), "RMSNorm feature dimensions do not match");
    const std::int64_t rows = x.numel() / x.size(-1);
    TORCH_CHECK(rows > 0 && x.size(-1) > 0, "RMSNorm dimensions must be nonzero");
    return {static_cast<size_t>(rows), static_cast<size_t>(x.size(-1))};
}

struct GeoDispatchTelemetry {
    std::string op_name = "none";
    std::string shape = "none";
    std::string requested_mode = "default";
    std::string selected_backend = "none";
    std::string reason = "none";
    std::string algorithm_id = "none";
    size_t workspace_bytes = 0;
    bool full_matrix_allocated = false;
    bool fallback = false;
    std::string fallback_reason = "";
};

static thread_local GeoDispatchTelemetry tls_last_dispatch;

pybind11::dict get_last_dispatch_telemetry() {
    pybind11::dict res;
    res["operator"] = tls_last_dispatch.op_name;
    res["shape"] = tls_last_dispatch.shape;
    res["requested_mode"] = tls_last_dispatch.requested_mode;
    res["selected_backend"] = tls_last_dispatch.selected_backend;
    res["reason"] = tls_last_dispatch.reason;
    res["algorithm_id"] = tls_last_dispatch.algorithm_id;
    res["workspace_bytes"] = tls_last_dispatch.workspace_bytes;
    res["full_matrix_allocated"] = tls_last_dispatch.full_matrix_allocated;
    res["fallback"] = tls_last_dispatch.fallback;
    res["fallback_reason"] = tls_last_dispatch.fallback_reason;
    return res;
}

void record_linear_telemetry(const char *op, const geo_tensor_linear_shape &shape, bool is_cuda) {
    tls_last_dispatch.op_name = op;
    tls_last_dispatch.shape = "M=" + std::to_string(shape.rows) + ",K=" + std::to_string(shape.in_features) + ",N=" + std::to_string(shape.out_features);
    tls_last_dispatch.requested_mode = "default";
    if (is_cuda && shape.out_features >= 512) {
        tls_last_dispatch.selected_backend = "cuBLAS Tensor Cores";
        tls_last_dispatch.reason = "wide_output_threshold_ge_512";
        tls_last_dispatch.algorithm_id = "cublasSgemm_v2";
    } else if (is_cuda) {
        tls_last_dispatch.selected_backend = "GEO Vectorized CUDA Kernel";
        tls_last_dispatch.reason = "small_output_dim_lt_512";
        tls_last_dispatch.algorithm_id = "geo_tensor_linear_forward_kernel";
    } else {
        tls_last_dispatch.selected_backend = "GEO CPU C Kernel";
        tls_last_dispatch.reason = "host_cpu_execution";
        tls_last_dispatch.algorithm_id = "geo_tensor_linear_forward";
    }
    tls_last_dispatch.workspace_bytes = 0;
    tls_last_dispatch.full_matrix_allocated = false;
    tls_last_dispatch.fallback = false;
    tls_last_dispatch.fallback_reason = "";
}

}  // namespace

torch::Tensor geo_linear_forward(torch::Tensor x, torch::Tensor weight) {
    check_tensor(x, "x");
    check_tensor(weight, "weight");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");
    const auto shape = make_linear_shape(x, weight);
    record_linear_telemetry("linear_forward", shape, x.is_cuda());
    std::vector<std::int64_t> output_shape(x.sizes().begin(), x.sizes().end());
    output_shape.back() = weight.size(0);
    torch::Tensor output = torch::empty(output_shape, x.options());
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_linear_cuda_forward(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), &shape, current_stream(x)), "geo_tensor_linear_cuda_forward");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_linear_forward(x.data_ptr<float>(), weight.data_ptr<float>(), output.data_ptr<float>(), &shape), "geo_tensor_linear_forward");
    }
    return output;
}

std::vector<torch::Tensor> geo_linear_backward(torch::Tensor x, torch::Tensor weight, torch::Tensor grad_output) {
    check_tensor(x, "x");
    check_tensor(weight, "weight");
    check_tensor(grad_output, "grad_output");
    TORCH_CHECK(x.device() == weight.device() && x.device() == grad_output.device(), "all tensors must share a device");
    const auto shape = make_linear_shape(x, weight);
    record_linear_telemetry("linear_backward", shape, x.is_cuda());
    torch::Tensor grad_x = torch::empty_like(x);
    torch::Tensor grad_weight = torch::empty_like(weight);
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_linear_cuda_vjp(x.data_ptr<float>(), weight.data_ptr<float>(), grad_output.data_ptr<float>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), &shape, current_stream(x)), "geo_tensor_linear_cuda_vjp");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_linear_vjp(x.data_ptr<float>(), weight.data_ptr<float>(), grad_output.data_ptr<float>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), &shape), "geo_tensor_linear_vjp");
    }
    return {grad_x, grad_weight};
}

torch::Tensor geo_add_forward(torch::Tensor a, torch::Tensor b) {
    check_tensor(a, "a");
    check_tensor(b, "b");
    check_same(a, b, "add");
    torch::Tensor out = torch::empty_like(a);
    if (a.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_add_cuda_forward(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), static_cast<size_t>(a.numel()), current_stream(a)), "geo_tensor_add_cuda_forward");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_add_forward(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), static_cast<size_t>(a.numel())), "geo_tensor_add_forward");
    }
    return out;
}

std::vector<torch::Tensor> geo_add_backward(torch::Tensor grad_output) {
    check_tensor(grad_output, "grad_output");
    torch::Tensor grad_a = torch::empty_like(grad_output);
    torch::Tensor grad_b = torch::empty_like(grad_output);
    if (grad_output.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_add_cuda_vjp(grad_output.data_ptr<float>(), grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), static_cast<size_t>(grad_output.numel()), current_stream(grad_output)), "geo_tensor_add_cuda_vjp");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_add_vjp(grad_output.data_ptr<float>(), grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), static_cast<size_t>(grad_output.numel())), "geo_tensor_add_vjp");
    }
    return {grad_a, grad_b};
}

torch::Tensor geo_mul_forward(torch::Tensor a, torch::Tensor b) {
    check_tensor(a, "a");
    check_tensor(b, "b");
    check_same(a, b, "mul");
    torch::Tensor out = torch::empty_like(a);
    if (a.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_mul_cuda_forward(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), static_cast<size_t>(a.numel()), current_stream(a)), "geo_tensor_mul_cuda_forward");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_mul_forward(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), static_cast<size_t>(a.numel())), "geo_tensor_mul_forward");
    }
    return out;
}

std::vector<torch::Tensor> geo_mul_backward(torch::Tensor a, torch::Tensor b, torch::Tensor grad_output) {
    check_tensor(a, "a");
    check_tensor(b, "b");
    check_tensor(grad_output, "grad_output");
    check_same(a, b, "mul backward");
    check_same(a, grad_output, "mul backward");
    torch::Tensor grad_a = torch::empty_like(a);
    torch::Tensor grad_b = torch::empty_like(b);
    if (a.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_mul_cuda_vjp(a.data_ptr<float>(), b.data_ptr<float>(), grad_output.data_ptr<float>(), grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), static_cast<size_t>(a.numel()), current_stream(a)), "geo_tensor_mul_cuda_vjp");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_mul_vjp(a.data_ptr<float>(), b.data_ptr<float>(), grad_output.data_ptr<float>(), grad_a.data_ptr<float>(), grad_b.data_ptr<float>(), static_cast<size_t>(a.numel())), "geo_tensor_mul_vjp");
    }
    return {grad_a, grad_b};
}

torch::Tensor geo_scale_forward(torch::Tensor x, double scalar) {
    check_tensor(x, "x");
    torch::Tensor out = torch::empty_like(x);
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_scale_cuda_forward(x.data_ptr<float>(), static_cast<float>(scalar), out.data_ptr<float>(), static_cast<size_t>(x.numel()), current_stream(x)), "geo_tensor_scale_cuda_forward");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_scale_forward(x.data_ptr<float>(), static_cast<float>(scalar), out.data_ptr<float>(), static_cast<size_t>(x.numel())), "geo_tensor_scale_forward");
    }
    return out;
}

torch::Tensor geo_scale_backward(torch::Tensor grad_output, double scalar) {
    check_tensor(grad_output, "grad_output");
    torch::Tensor grad_x = torch::empty_like(grad_output);
    if (grad_output.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_scale_cuda_vjp(grad_output.data_ptr<float>(), static_cast<float>(scalar), grad_x.data_ptr<float>(), static_cast<size_t>(grad_output.numel()), current_stream(grad_output)), "geo_tensor_scale_cuda_vjp");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_scale_vjp(grad_output.data_ptr<float>(), static_cast<float>(scalar), grad_x.data_ptr<float>(), static_cast<size_t>(grad_output.numel())), "geo_tensor_scale_vjp");
    }
    return grad_x;
}

std::vector<torch::Tensor> geo_rms_norm_forward(torch::Tensor x, torch::Tensor weight, double epsilon) {
    check_tensor(x, "x");
    check_tensor(weight, "weight");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must share a device");
    const auto shape = make_norm_shape(x, weight);
    torch::Tensor out = torch::empty_like(x);
    torch::Tensor inv_rms = torch::empty({static_cast<std::int64_t>(shape.rows)}, x.options());
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_rms_norm_cuda_forward(x.data_ptr<float>(), weight.data_ptr<float>(), static_cast<float>(epsilon), out.data_ptr<float>(), inv_rms.data_ptr<float>(), shape, current_stream(x)), "geo_tensor_rms_norm_cuda_forward");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_rms_norm_forward(x.data_ptr<float>(), weight.data_ptr<float>(), static_cast<float>(epsilon), out.data_ptr<float>(), inv_rms.data_ptr<float>(), shape), "geo_tensor_rms_norm_forward");
    }
    return {out, inv_rms};
}

std::vector<torch::Tensor> geo_rms_norm_backward(torch::Tensor x, torch::Tensor weight, torch::Tensor grad_output, torch::Tensor inv_rms) {
    check_tensor(x, "x");
    check_tensor(weight, "weight");
    check_tensor(grad_output, "grad_output");
    check_tensor(inv_rms, "inv_rms");
    TORCH_CHECK(x.device() == weight.device() && x.device() == grad_output.device() && x.device() == inv_rms.device(), "all RMSNorm tensors must share a device");
    const auto shape = make_norm_shape(x, weight);
    TORCH_CHECK(grad_output.sizes() == x.sizes(), "RMSNorm grad_output shape mismatch");
    TORCH_CHECK(inv_rms.numel() == static_cast<std::int64_t>(shape.rows), "RMSNorm inv_rms shape mismatch");
    torch::Tensor grad_x = torch::empty_like(x);
    torch::Tensor grad_weight = torch::empty_like(weight);
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_rms_norm_cuda_vjp(x.data_ptr<float>(), weight.data_ptr<float>(), grad_output.data_ptr<float>(), inv_rms.data_ptr<float>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), shape, current_stream(x)), "geo_tensor_rms_norm_cuda_vjp");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_rms_norm_vjp(x.data_ptr<float>(), weight.data_ptr<float>(), grad_output.data_ptr<float>(), inv_rms.data_ptr<float>(), grad_x.data_ptr<float>(), grad_weight.data_ptr<float>(), shape), "geo_tensor_rms_norm_vjp");
    }
    return {grad_x, grad_weight};
}

torch::Tensor geo_gelu_forward(torch::Tensor x) {
    check_tensor(x, "x");
    torch::Tensor out = torch::empty_like(x);
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_gelu_cuda_forward(x.data_ptr<float>(), out.data_ptr<float>(), static_cast<size_t>(x.numel()), current_stream(x)), "geo_tensor_gelu_cuda_forward");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_gelu_forward(x.data_ptr<float>(), out.data_ptr<float>(), static_cast<size_t>(x.numel())), "geo_tensor_gelu_forward");
    }
    return out;
}

torch::Tensor geo_gelu_backward(torch::Tensor x, torch::Tensor grad_output) {
    check_tensor(x, "x");
    check_tensor(grad_output, "grad_output");
    check_same(x, grad_output, "GELU backward");
    torch::Tensor grad_x = torch::empty_like(x);
    if (x.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_gelu_cuda_vjp(x.data_ptr<float>(), grad_output.data_ptr<float>(), grad_x.data_ptr<float>(), static_cast<size_t>(x.numel()), current_stream(x)), "geo_tensor_gelu_cuda_vjp");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_gelu_vjp(x.data_ptr<float>(), grad_output.data_ptr<float>(), grad_x.data_ptr<float>(), static_cast<size_t>(x.numel())), "geo_tensor_gelu_vjp");
    }
    return grad_x;
}

torch::Tensor geo_silu_mul_forward(torch::Tensor gate, torch::Tensor up) {
    check_tensor(gate, "gate");
    check_tensor(up, "up");
    check_same(gate, up, "SiLU-multiply");
    torch::Tensor out = torch::empty_like(gate);
    if (gate.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_silu_mul_cuda_forward(gate.data_ptr<float>(), up.data_ptr<float>(), out.data_ptr<float>(), static_cast<size_t>(gate.numel()), current_stream(gate)), "geo_tensor_silu_mul_cuda_forward");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_silu_mul_forward(gate.data_ptr<float>(), up.data_ptr<float>(), out.data_ptr<float>(), static_cast<size_t>(gate.numel())), "geo_tensor_silu_mul_forward");
    }
    return out;
}

std::vector<torch::Tensor> geo_silu_mul_backward(torch::Tensor gate, torch::Tensor up, torch::Tensor grad_output) {
    check_tensor(gate, "gate");
    check_tensor(up, "up");
    check_tensor(grad_output, "grad_output");
    check_same(gate, up, "SiLU-multiply backward");
    check_same(gate, grad_output, "SiLU-multiply backward");
    torch::Tensor grad_gate = torch::empty_like(gate);
    torch::Tensor grad_up = torch::empty_like(up);
    if (gate.is_cuda()) {
#ifdef WITH_CUDA
        check_status(geo_tensor_silu_mul_cuda_vjp(gate.data_ptr<float>(), up.data_ptr<float>(), grad_output.data_ptr<float>(), grad_gate.data_ptr<float>(), grad_up.data_ptr<float>(), static_cast<size_t>(gate.numel()), current_stream(gate)), "geo_tensor_silu_mul_cuda_vjp");
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(geo_tensor_silu_mul_vjp(gate.data_ptr<float>(), up.data_ptr<float>(), grad_output.data_ptr<float>(), grad_gate.data_ptr<float>(), grad_up.data_ptr<float>(), static_cast<size_t>(gate.numel())), "geo_tensor_silu_mul_vjp");
    }
    return {grad_gate, grad_up};
}

#ifdef WITH_CUDA
std::map<std::string, double> geo_linear_backward_decomposed_profile(torch::Tensor x, torch::Tensor weight, torch::Tensor grad_output) {
    check_tensor(x, "x");
    check_tensor(weight, "weight");
    check_tensor(grad_output, "grad_output");
    TORCH_CHECK(x.device() == weight.device() && x.device() == grad_output.device(), "all tensors must share a device");
    const auto shape = make_linear_shape(x, weight);
    torch::Tensor grad_x = torch::empty_like(x);
    torch::Tensor grad_weight = torch::empty_like(weight);

    float dx_ms = 0.0f;
    float dw_ms = 0.0f;

    check_status(
        geo_tensor_linear_cuda_vjp_decomposed_profile(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            grad_output.data_ptr<float>(),
            grad_x.data_ptr<float>(),
            grad_weight.data_ptr<float>(),
            &shape,
            &dx_ms,
            &dw_ms,
            current_stream(x)
        ),
        "geo_tensor_linear_cuda_vjp_decomposed_profile"
    );

    std::map<std::string, double> res;
    res["dx_ms"] = static_cast<double>(dx_ms);
    res["dw_ms"] = static_cast<double>(dw_ms);
    return res;
}
#endif

#ifdef WITH_CUDA
torch::Tensor geo_implicit_linear_forward(
    const torch::Tensor &x,
    const torch::Tensor &u,
    const torch::Tensor &v,
    const torch::Tensor &alpha,
    const torch::Tensor &perm_indices,
    const torch::Tensor &sign_masks
) {
    check_tensor(x, "x");
    check_tensor(u, "u");
    check_tensor(v, "v");
    check_tensor(alpha, "alpha");

    const int64_t batch_tokens = x.numel() / x.size(-1);
    const int64_t in_features = x.size(-1);
    const int64_t out_features = u.size(1);
    const int64_t rank = u.size(0);

    auto y = torch::empty({x.size(0), x.size(1), out_features}, x.options());
    geo_implicit_shape shape = {
        static_cast<size_t>(batch_tokens),
        static_cast<size_t>(in_features),
        static_cast<size_t>(out_features),
        static_cast<size_t>(rank)
    };

    geo_tensor_status status = geo_implicit_linear_cuda_forward(
        x.data_ptr<float>(),
        u.data_ptr<float>(),
        v.data_ptr<float>(),
        alpha.data_ptr<float>(),
        perm_indices.data_ptr<int32_t>(),
        sign_masks.data_ptr<float>(),
        y.data_ptr<float>(),
        &shape,
        current_stream(x)
    );
    check_status(status, "geo_implicit_linear_cuda_forward");
    return y;
}

std::vector<torch::Tensor> geo_implicit_linear_backward(
    const torch::Tensor &x,
    const torch::Tensor &u,
    const torch::Tensor &v,
    const torch::Tensor &alpha,
    const torch::Tensor &perm_indices,
    const torch::Tensor &inv_perm,
    const torch::Tensor &sign_masks,
    const torch::Tensor &grad_y
) {
    check_tensor(x, "x");
    check_tensor(u, "u");
    check_tensor(v, "v");
    check_tensor(alpha, "alpha");
    check_tensor(grad_y, "grad_y");

    const int64_t batch_tokens = x.numel() / x.size(-1);
    const int64_t in_features = x.size(-1);
    const int64_t out_features = u.size(1);
    const int64_t rank = u.size(0);

    auto grad_x = torch::zeros_like(x);
    auto grad_u = torch::zeros_like(u);
    auto grad_v = torch::zeros_like(v);
    auto grad_alpha = torch::zeros_like(alpha);

    geo_implicit_shape shape = {
        static_cast<size_t>(batch_tokens),
        static_cast<size_t>(in_features),
        static_cast<size_t>(out_features),
        static_cast<size_t>(rank)
    };

    geo_tensor_status status = geo_implicit_linear_cuda_vjp(
        x.data_ptr<float>(),
        u.data_ptr<float>(),
        v.data_ptr<float>(),
        alpha.data_ptr<float>(),
        perm_indices.data_ptr<int32_t>(),
        inv_perm.data_ptr<int32_t>(),
        sign_masks.data_ptr<float>(),
        grad_y.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_u.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        grad_alpha.data_ptr<float>(),
        &shape,
        current_stream(x)
    );
    check_status(status, "geo_implicit_linear_cuda_vjp");
    return {grad_x, grad_u, grad_v, grad_alpha};
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("get_last_dispatch_telemetry", &get_last_dispatch_telemetry);
    module.def("linear_forward", &geo_linear_forward);
    module.def("linear_backward", &geo_linear_backward);
#ifdef WITH_CUDA
    module.def("linear_backward_decomposed_profile", &geo_linear_backward_decomposed_profile);
    module.def("implicit_linear_forward", &geo_implicit_linear_forward);
    module.def("implicit_linear_backward", &geo_implicit_linear_backward);
#endif

    module.def("add_forward", &geo_add_forward);
    module.def("add_backward", &geo_add_backward);
    module.def("mul_forward", &geo_mul_forward);
    module.def("mul_backward", &geo_mul_backward);
    module.def("scale_forward", &geo_scale_forward);
    module.def("scale_backward", &geo_scale_backward);
    module.def("rms_norm_forward", &geo_rms_norm_forward);
    module.def("rms_norm_backward", &geo_rms_norm_backward);
    module.def("gelu_forward", &geo_gelu_forward);
    module.def("gelu_backward", &geo_gelu_backward);
    module.def("silu_mul_forward", &geo_silu_mul_forward);
    module.def("silu_mul_backward", &geo_silu_mul_backward);
    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple("linear", "add", "mul", "scale", "rms_norm", "gelu", "silu_mul");
    
    pybind11::dict dispatcher;
    dispatcher["linear"] = "Adaptive: cuBLAS Tensor Cores (N >= 512) | GEO Vectorized Kernel (N < 512)";
    dispatcher["attention"] = "GEO Recomputation Tiled Causal Attention Kernel";
    dispatcher["loss"] = "GEO Parallel Block-Reduction Cross-Entropy (V > 512) | Serial (V <= 512)";
    dispatcher["optimizer"] = "GEO Single-Launch Fused Multi-Tensor GeoAdamW + Fused Norm Clipping";
    module.attr("GEO_EXECUTION_DISPATCHER") = dispatcher;

#ifdef WITH_CUDA
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
}
