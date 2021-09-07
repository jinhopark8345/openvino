// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_elements_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

static size_t GetGatherElementsChannelIndex(const gather_elements_params& params) {
    Tensor::DataChannelName name = Tensor::DataChannelName::X;

    size_t inputSize = params.inputs[0].GetDims().size();

    switch (params.axis) {
        case GatherAxis::X:
            return inputSize - 1;
        case GatherAxis::Y:
            return inputSize - 2;
        case GatherAxis::Z:
            return inputSize - 3;
        case GatherAxis::W:
            return 2;
        case GatherAxis::FEATURE:
            return 1;
        case GatherAxis::BATCH:
            return 0;
        default:
            break;
    }

    return DataTensor::Channelndex(params.output.GetLayout(), name);
}
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

    const auto& output = params.output;

    switch (params.inputs[1].GetLayout()) {
    case DataLayout::bfyx:
        dispatchData.gws = {output.X().v, output.Y().v, output.Feature().v * output.Batch().v};
        // dispatchData.gws = {1, 1, 1};
        break;

    case DataLayout::bfzyx:
        dispatchData.gws = {output.X().v, output.Y().v * output.Z().v, output.Feature().v * output.Batch().v};
        // dispatchData.gws = {1, 1, 1};
        break;

    case DataLayout::bfwzyx:
        dispatchData.gws = {output.X().v * output.Y().v, output.Z().v * output.W().v, output.Feature().v * output.Batch().v};
        // dispatchData.gws = {1, 1, 1};
        break;

    default:
        throw std::invalid_argument("Unsupported data layout for gather elements primitive");
        break;
    }


    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    // dispatchData.lws = {1, 1, 1};

    return dispatchData;
}

// static size_t GetIndicesLastDim(const gather_elements_params& params) {
//     // get indices dims
//     auto indices_dims = params.inputs[1].LogicalDims();

//     if (indices_dims.size() > 1) {
//         std::reverse(indices_dims.begin(), indices_dims.end());
//     }

//     auto indices_last_dim = indices_dims[params.indices_rank - 1];

//     return indices_last_dim;
// }

// static size_t GetSliceSize(const gather_elements_params& params) {
//     // get input dims
//     auto input_dims = params.inputs[0].LogicalDims();

//     if (input_dims.size() > 1) {
//         std::reverse(input_dims.begin(), input_dims.end());
//     }

//     // get last dim of indices
//     auto indices_last_dim = GetIndicesLastDim(params);

//     // calculate slize size which is used in kernel to copy
//     size_t wi_slice_size = 1;
//     for (size_t i = params.batch_dims + indices_last_dim; i < input_dims.size(); i++) {
//         wi_slice_size *= input_dims[i];
//     }

//     return wi_slice_size;
// }

JitConstants GatherElementsKernelRef::GetJitConstants(const gather_elements_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    // jit.AddConstant(MakeJitConstant("INDICES_RANK", params.indices_rank));
    // jit.AddConstant(MakeJitConstant("BATCH_DIMS", params.batch_dims));
    // jit.AddConstant(MakeJitConstant("WI_SLICE_SIZE", GetSliceSize(params)));
    // jit.AddConstant(MakeJitConstant("INDICES_LAST_DIM", GetIndicesLastDim(params)));

    jit.AddConstant(MakeJitConstant("AXIS", GetGatherElementsChannelIndex(params)));

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

    if (input_dims.size() != indices_dims.size()) {
        return false;
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
