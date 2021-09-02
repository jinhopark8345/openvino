// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey GatherElementsKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

static inline std::vector<std::string> GetDefaultOrder(size_t size) {
    std::vector<std::string> default_order;
    if (size <= 4) {
        default_order = { "b", "f", "y", "x" };
    } else if (size == 5) {
        default_order = { "b", "f", "z", "y", "x" };
    } else if (size == 6) {
        default_order = { "b", "f", "w", "z", "y", "x" };
    }

    return default_order;
}

CommonDispatchData GatherElementsKernelRef::SetDefault(const gather_elements_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;

    auto indices_dims = params.inputs[1].LogicalDims();

    if (indices_dims.size() > 1) {
        std::reverse(indices_dims.begin(), indices_dims.end());
    }

    indices_dims[params.indices_rank - 1] = 1; // set last dim of indices to 1

    switch (params.inputs[1].GetLayout()) {
    case DataLayout::bfyx:
        dispatchData.gws = { indices_dims[3], indices_dims[2], indices_dims[1] * indices_dims[0] };
        break;

    case DataLayout::bfzyx:
        dispatchData.gws = { indices_dims[4] * indices_dims[3], indices_dims[2], indices_dims[1] * indices_dims[0] };
        break;

    case DataLayout::bfwzyx:
        dispatchData.gws = { indices_dims[5] * indices_dims[4], indices_dims[3] * indices_dims[2], indices_dims[1] * indices_dims[0] };
        break;

    default:
        throw std::invalid_argument("Unsupported data layout for scatter elements update primitive");
        break;
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

static size_t GetIndicesLastDim(const gather_elements_params& params) {
    // get indices dims
    auto indices_dims = params.inputs[1].LogicalDims();

    if (indices_dims.size() > 1) {
        std::reverse(indices_dims.begin(), indices_dims.end());
    }

    auto indices_last_dim = indices_dims[params.indices_rank - 1];

    return indices_last_dim;
}

static size_t GetSliceSize(const gather_elements_params& params) {
    // get input dims
    auto input_dims = params.inputs[0].LogicalDims();

    if (input_dims.size() > 1) {
        std::reverse(input_dims.begin(), input_dims.end());
    }

    // get last dim of indices
    auto indices_last_dim = GetIndicesLastDim(params);

    // calculate slize size which is used in kernel to copy
    size_t wi_slice_size = 1;
    for (size_t i = params.batch_dims + indices_last_dim; i < input_dims.size(); i++) {
        wi_slice_size *= input_dims[i];
    }

    return wi_slice_size;
}

JitConstants GatherElementsKernelRef::GetJitConstants(const gather_elements_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("INDICES_RANK", params.indices_rank));
    jit.AddConstant(MakeJitConstant("BATCH_DIMS", params.batch_dims));
    jit.AddConstant(MakeJitConstant("WI_SLICE_SIZE", GetSliceSize(params)));
    jit.AddConstant(MakeJitConstant("INDICES_LAST_DIM", GetIndicesLastDim(params)));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf = { "", GetDefaultOrder(params.output.GetDims().size()), "val", params.inputs[0].GetDType() };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

bool GatherElementsKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::GATHER_ELEMENTS || o.GetType() != KernelType::GATHER_ELEMENTS) {
        return false;
    }

    const gather_elements_params& params = static_cast<const gather_elements_params&>(p);
    auto input_dims = params.inputs[0].LogicalDims();
    auto indices_dims = params.inputs[1].LogicalDims();
    auto indices_rank = params.indices_rank;
    auto batch_dims = params.batch_dims;

    std::reverse(input_dims.begin(), input_dims.end());
    std::reverse(indices_dims.begin(), indices_dims.end());

    if (indices_rank < 1) {
        return false;
    }

    if (batch_dims + indices_dims[indices_rank - 1] > input_dims.size()) {
        return false;
    }

    if (batch_dims >= std::min(input_dims.size(), static_cast<size_t>(indices_rank))) {
        return false;
    }

    for (uint8_t i = 0; i < batch_dims; i++) {
        if (input_dims[i] != indices_dims[i]) {
            return false;
        }
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

KernelsData GatherElementsKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<gather_elements_params>(params);
    gather_elements_params& newParams = *static_cast<gather_elements_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams, options);
    auto cldnn_jit = GetJitConstants(newParams);

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2, GetFusedPrimitiveInputsCount(params));

    return { kd };
}

}  // namespace kernel_selector
