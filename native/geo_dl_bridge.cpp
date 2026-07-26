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

}  // namespace

torch::Tensor geo_linear_forward(torch::Tensor x, torch::Tensor weight) {
    check_tensor(x, "x");
    check_tensor(weight, "weight");
    TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");
    const auto shape = make_linear_shape(x, weight);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("linear_forward", &geo_linear_forward);
    module.def("linear_backward", &geo_linear_backward);
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
