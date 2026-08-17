#include <torch/extension.h>

#include <cstdint>
#include <vector>

#include "geo/tensor_relational.h"
#ifdef WITH_CUDA
#include <ATen/cuda/CUDAContext.h>
#include "geo/tensor_relational_cuda.h"
#endif

namespace {

void check_float_tensor(const torch::Tensor &tensor, const char *name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

void check_status(geo_relational_status status, const char *operation) {
    TORCH_CHECK(status == GEO_RELATIONAL_OK, operation, " failed: ", geo_relational_status_string(status));
}

#ifdef WITH_CUDA
void *current_stream(const torch::Tensor &tensor) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(tensor.get_device()).stream();
    return reinterpret_cast<void *>(stream);
}
#endif

geo_relational_shape make_relational_shape(
    const torch::Tensor &state,
    size_t matrix_count
) {
    TORCH_CHECK(state.dim() == 3, "State tensor must be rank 3 (groups, streams, features)");
    return {
        static_cast<size_t>(state.size(0)),
        static_cast<size_t>(state.size(1)),
        static_cast<size_t>(state.size(2)),
        matrix_count
    };
}

}  // namespace

std::vector<torch::Tensor> birkhoff_project_forward(
    torch::Tensor logits,
    std::int64_t iterations,
    double epsilon,
    bool fail_on_nonfinite,
    bool require_certificate
) {
    check_float_tensor(logits, "logits");
    TORCH_CHECK(logits.dim() == 3, "Logits must be (M, P, P)");
    TORCH_CHECK(logits.size(1) == logits.size(2), "Logits must be square (P x P)");

    size_t M = logits.size(0);
    size_t P = logits.size(1);

    geo_relational_shape shape = { M, P, 1, M };
    geo_relational_projection_options options = {
        GEO_RELATIONAL_ABI_VERSION,
        static_cast<uint32_t>(iterations),
        static_cast<geo_real_t>(epsilon),
        GEO_RELATIONAL_PROJECTION_BIRKHOFF_LOG_SINKHORN,
        static_cast<uint8_t>(fail_on_nonfinite ? 1 : 0),
        static_cast<uint8_t>(require_certificate ? 1 : 0),
        0
    };

    size_t ws_elems = geo_relational_projection_workspace_elements(M, P, static_cast<uint32_t>(iterations), 1);
    torch::Tensor relationship = torch::empty_like(logits);
    torch::Tensor workspace = torch::empty({static_cast<std::int64_t>(ws_elems)}, logits.options());
    torch::Tensor cert_tensor = torch::empty({static_cast<std::int64_t>(M * sizeof(geo_relational_certificate))}, torch::dtype(torch::kUInt8));

    geo_relational_certificate *certs = reinterpret_cast<geo_relational_certificate *>(cert_tensor.data_ptr<uint8_t>());

    if (logits.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_project_forward_cuda(
                logits.data_ptr<float>(),
                relationship.data_ptr<float>(),
                workspace.data_ptr<float>(),
                ws_elems,
                &shape,
                &options,
                certs,
                current_stream(logits)
            ),
            "geo_relational_project_forward_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_project_forward(
                logits.data_ptr<float>(),
                relationship.data_ptr<float>(),
                workspace.data_ptr<float>(),
                ws_elems,
                &shape,
                &options,
                certs
            ),
            "geo_relational_project_forward"
        );
    }

    return {relationship, workspace, cert_tensor};
}

torch::Tensor birkhoff_project_backward(
    torch::Tensor logits,
    torch::Tensor relationship_cotangent,
    torch::Tensor workspace,
    std::int64_t iterations,
    double epsilon
) {
    check_float_tensor(logits, "logits");
    check_float_tensor(relationship_cotangent, "relationship_cotangent");
    check_float_tensor(workspace, "workspace");

    size_t M = logits.size(0);
    size_t P = logits.size(1);

    geo_relational_shape shape = { M, P, 1, M };
    geo_relational_projection_options options = {
        GEO_RELATIONAL_ABI_VERSION,
        static_cast<uint32_t>(iterations),
        static_cast<geo_real_t>(epsilon),
        GEO_RELATIONAL_PROJECTION_BIRKHOFF_LOG_SINKHORN,
        1, 0, 0
    };

    torch::Tensor logits_cotangent = torch::empty_like(logits);
    size_t ws_elems = workspace.numel();

    if (logits.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_project_vjp_cuda(
                logits.data_ptr<float>(),
                relationship_cotangent.data_ptr<float>(),
                logits_cotangent.data_ptr<float>(),
                workspace.data_ptr<float>(),
                ws_elems,
                &shape,
                &options,
                current_stream(logits)
            ),
            "geo_relational_project_vjp_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_project_vjp(
                logits.data_ptr<float>(),
                relationship_cotangent.data_ptr<float>(),
                logits_cotangent.data_ptr<float>(),
                workspace.data_ptr<float>(),
                ws_elems,
                &shape,
                &options
            ),
            "geo_relational_project_vjp"
        );
    }

    return logits_cotangent;
}

torch::Tensor identity_gate_forward(
    torch::Tensor projected,
    torch::Tensor gate
) {
    check_float_tensor(projected, "projected");
    check_float_tensor(gate, "gate");

    size_t M = projected.size(0);
    size_t P = projected.size(1);
    size_t G = gate.numel();
    geo_relational_shape shape = { G, P, 1, M };

    torch::Tensor effective = torch::empty_like(projected);
    if (projected.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_identity_gate_forward_cuda(
                projected.data_ptr<float>(),
                gate.data_ptr<float>(),
                effective.data_ptr<float>(),
                &shape,
                current_stream(projected)
            ),
            "geo_relational_identity_gate_forward_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_identity_gate_forward(
                projected.data_ptr<float>(),
                gate.data_ptr<float>(),
                effective.data_ptr<float>(),
                &shape
            ),
            "geo_relational_identity_gate_forward"
        );
    }
    return effective;
}

std::vector<torch::Tensor> identity_gate_backward(
    torch::Tensor projected,
    torch::Tensor gate,
    torch::Tensor effective_cotangent
) {
    check_float_tensor(projected, "projected");
    check_float_tensor(gate, "gate");
    check_float_tensor(effective_cotangent, "effective_cotangent");

    size_t M = projected.size(0);
    size_t P = projected.size(1);
    size_t G = gate.numel();
    geo_relational_shape shape = { G, P, 1, M };

    torch::Tensor projected_cotangent = torch::empty_like(projected);
    torch::Tensor gate_cotangent = torch::empty_like(gate);

    if (projected.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_identity_gate_vjp_cuda(
                projected.data_ptr<float>(),
                gate.data_ptr<float>(),
                effective_cotangent.data_ptr<float>(),
                projected_cotangent.data_ptr<float>(),
                gate_cotangent.data_ptr<float>(),
                &shape,
                current_stream(projected)
            ),
            "geo_relational_identity_gate_vjp_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_identity_gate_vjp(
                projected.data_ptr<float>(),
                gate.data_ptr<float>(),
                effective_cotangent.data_ptr<float>(),
                projected_cotangent.data_ptr<float>(),
                gate_cotangent.data_ptr<float>(),
                &shape
            ),
            "geo_relational_identity_gate_vjp"
        );
    }
    return {projected_cotangent, gate_cotangent};
}

torch::Tensor relational_mix_forward(
    torch::Tensor state,
    torch::Tensor relationship
) {
    check_float_tensor(state, "state");
    check_float_tensor(relationship, "relationship");

    size_t M = relationship.size(0);
    geo_relational_shape shape = make_relational_shape(state, M);

    torch::Tensor output = torch::empty_like(state);
    if (state.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_mix_forward_cuda(
                state.data_ptr<float>(),
                relationship.data_ptr<float>(),
                output.data_ptr<float>(),
                &shape,
                current_stream(state)
            ),
            "geo_relational_mix_forward_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_mix_forward(
                state.data_ptr<float>(),
                relationship.data_ptr<float>(),
                output.data_ptr<float>(),
                &shape
            ),
            "geo_relational_mix_forward"
        );
    }
    return output;
}

std::vector<torch::Tensor> relational_mix_backward(
    torch::Tensor state,
    torch::Tensor relationship,
    torch::Tensor output_cotangent
) {
    check_float_tensor(state, "state");
    check_float_tensor(relationship, "relationship");
    check_float_tensor(output_cotangent, "output_cotangent");

    size_t M = relationship.size(0);
    geo_relational_shape shape = make_relational_shape(state, M);

    torch::Tensor state_cotangent = torch::empty_like(state);
    torch::Tensor relationship_cotangent = torch::empty_like(relationship);

    if (state.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_mix_vjp_cuda(
                state.data_ptr<float>(),
                relationship.data_ptr<float>(),
                output_cotangent.data_ptr<float>(),
                state_cotangent.data_ptr<float>(),
                relationship_cotangent.data_ptr<float>(),
                &shape,
                current_stream(state)
            ),
            "geo_relational_mix_vjp_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_mix_vjp(
                state.data_ptr<float>(),
                relationship.data_ptr<float>(),
                output_cotangent.data_ptr<float>(),
                state_cotangent.data_ptr<float>(),
                relationship_cotangent.data_ptr<float>(),
                &shape
            ),
            "geo_relational_mix_vjp"
        );
    }
    return {state_cotangent, relationship_cotangent};
}

torch::Tensor relational_read_forward(
    torch::Tensor state,
    torch::Tensor read_weights
) {
    check_float_tensor(state, "state");
    check_float_tensor(read_weights, "read_weights");

    size_t weight_count = read_weights.size(0);
    geo_relational_shape shape = make_relational_shape(state, weight_count);

    torch::Tensor read_state = torch::empty({state.size(0), state.size(2)}, state.options());

    if (state.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_read_forward_cuda(
                state.data_ptr<float>(),
                read_weights.data_ptr<float>(),
                weight_count,
                read_state.data_ptr<float>(),
                &shape,
                current_stream(state)
            ),
            "geo_relational_read_forward_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_read_forward(
                state.data_ptr<float>(),
                read_weights.data_ptr<float>(),
                weight_count,
                read_state.data_ptr<float>(),
                &shape
            ),
            "geo_relational_read_forward"
        );
    }
    return read_state;
}

std::vector<torch::Tensor> relational_read_backward(
    torch::Tensor state,
    torch::Tensor read_weights,
    torch::Tensor read_state_cotangent
) {
    check_float_tensor(state, "state");
    check_float_tensor(read_weights, "read_weights");
    check_float_tensor(read_state_cotangent, "read_state_cotangent");

    size_t weight_count = read_weights.size(0);
    geo_relational_shape shape = make_relational_shape(state, weight_count);

    torch::Tensor state_cotangent = torch::empty_like(state);
    torch::Tensor read_weights_cotangent = torch::empty_like(read_weights);

    if (state.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_read_vjp_cuda(
                state.data_ptr<float>(),
                read_weights.data_ptr<float>(),
                weight_count,
                read_state_cotangent.data_ptr<float>(),
                state_cotangent.data_ptr<float>(),
                read_weights_cotangent.data_ptr<float>(),
                &shape,
                current_stream(state)
            ),
            "geo_relational_read_vjp_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_read_vjp(
                state.data_ptr<float>(),
                read_weights.data_ptr<float>(),
                weight_count,
                read_state_cotangent.data_ptr<float>(),
                state_cotangent.data_ptr<float>(),
                read_weights_cotangent.data_ptr<float>(),
                &shape
            ),
            "geo_relational_read_vjp"
        );
    }
    return {state_cotangent, read_weights_cotangent};
}

torch::Tensor relational_write_add_forward(
    torch::Tensor transported,
    torch::Tensor source,
    torch::Tensor write_weights,
    c10::optional<torch::Tensor> source_scale
) {
    check_float_tensor(transported, "transported");
    check_float_tensor(source, "source");
    check_float_tensor(write_weights, "write_weights");

    size_t weight_count = write_weights.size(0);
    size_t scale_count = source_scale.has_value() ? source_scale->size(0) : 0;
    geo_relational_shape shape = make_relational_shape(transported, weight_count);

    torch::Tensor output = torch::empty_like(transported);
    const float *scale_ptr = source_scale.has_value() ? source_scale->data_ptr<float>() : nullptr;

    if (transported.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_write_add_forward_cuda(
                transported.data_ptr<float>(),
                source.data_ptr<float>(),
                write_weights.data_ptr<float>(),
                weight_count,
                scale_ptr,
                scale_count,
                output.data_ptr<float>(),
                &shape,
                current_stream(transported)
            ),
            "geo_relational_write_add_forward_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_write_add_forward(
                transported.data_ptr<float>(),
                source.data_ptr<float>(),
                write_weights.data_ptr<float>(),
                weight_count,
                scale_ptr,
                scale_count,
                output.data_ptr<float>(),
                &shape
            ),
            "geo_relational_write_add_forward"
        );
    }
    return output;
}

std::vector<torch::Tensor> relational_write_add_backward(
    torch::Tensor source,
    torch::Tensor write_weights,
    c10::optional<torch::Tensor> source_scale,
    torch::Tensor output_cotangent
) {
    check_float_tensor(source, "source");
    check_float_tensor(write_weights, "write_weights");
    check_float_tensor(output_cotangent, "output_cotangent");

    size_t weight_count = write_weights.size(0);
    size_t scale_count = source_scale.has_value() ? source_scale->size(0) : 0;
    size_t G = output_cotangent.size(0);
    size_t P = output_cotangent.size(1);
    size_t D = output_cotangent.size(2);
    geo_relational_shape shape = { G, P, D, weight_count };

    torch::Tensor transported_cotangent = torch::empty_like(output_cotangent);
    torch::Tensor source_cotangent = torch::empty_like(source);
    torch::Tensor write_weights_cotangent = torch::empty_like(write_weights);
    torch::Tensor source_scale_cotangent;
    if (source_scale.has_value()) {
        source_scale_cotangent = torch::empty_like(*source_scale);
    }

    const float *scale_ptr = source_scale.has_value() ? source_scale->data_ptr<float>() : nullptr;
    float *scale_cot_ptr = source_scale.has_value() ? source_scale_cotangent.data_ptr<float>() : nullptr;

    if (output_cotangent.is_cuda()) {
#ifdef WITH_CUDA
        check_status(
            geo_relational_write_add_vjp_cuda(
                source.data_ptr<float>(),
                write_weights.data_ptr<float>(),
                weight_count,
                scale_ptr,
                scale_count,
                output_cotangent.data_ptr<float>(),
                transported_cotangent.data_ptr<float>(),
                source_cotangent.data_ptr<float>(),
                write_weights_cotangent.data_ptr<float>(),
                scale_cot_ptr,
                &shape,
                current_stream(output_cotangent)
            ),
            "geo_relational_write_add_vjp_cuda"
        );
#else
        TORCH_CHECK(false, "runtime was built without CUDA support");
#endif
    } else {
        check_status(
            geo_relational_write_add_vjp(
                source.data_ptr<float>(),
                write_weights.data_ptr<float>(),
                weight_count,
                scale_ptr,
                scale_count,
                output_cotangent.data_ptr<float>(),
                transported_cotangent.data_ptr<float>(),
                source_cotangent.data_ptr<float>(),
                write_weights_cotangent.data_ptr<float>(),
                scale_cot_ptr,
                &shape
            ),
            "geo_relational_write_add_vjp"
        );
    }

    return {transported_cotangent, source_cotangent, write_weights_cotangent, source_scale_cotangent};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("birkhoff_project_forward", &birkhoff_project_forward);
    module.def("birkhoff_project_backward", &birkhoff_project_backward);
    module.def("identity_gate_forward", &identity_gate_forward);
    module.def("identity_gate_backward", &identity_gate_backward);
    module.def("relational_mix_forward", &relational_mix_forward);
    module.def("relational_mix_backward", &relational_mix_backward);
    module.def("relational_read_forward", &relational_read_forward);
    module.def("relational_read_backward", &relational_read_backward);
    module.def("relational_write_add_forward", &relational_write_add_forward);
    module.def("relational_write_add_backward", &relational_write_add_backward);

    module.attr("GEO_DL_RUNTIME_ABI_VERSION") = 1;
    module.attr("GEO_BACKEND") = "GeometricElementaryOperators";
    module.attr("GEO_OWNS_BACKWARD") = true;
    module.attr("GEO_CAPABILITIES") = pybind11::make_tuple(
        "birkhoff_project",
        "relational_identity_gate",
        "relational_mix",
        "relational_read",
        "relational_write_add"
    );
#ifdef WITH_CUDA
    module.attr("GEO_CUDA_AVAILABLE") = true;
#else
    module.attr("GEO_CUDA_AVAILABLE") = false;
#endif
}
