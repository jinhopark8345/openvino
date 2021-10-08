// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_nd8_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_nd8_params : public base_params {
    gather_nd8_params() : base_params(KernelType::GATHER_ND8), indices_rank(0), batch_dims(0) {}

    uint8_t indices_rank;

    uint8_t batch_dims;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_nd8_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_nd8_optional_params : optional_params {
    gather_nd8_optional_params() : optional_params(KernelType::GATHER_ND8) {}
};

class GatherND8KernelRef : public KernelBaseOpenCL {
public:
    GatherND8KernelRef() : KernelBaseOpenCL("gather_nd8_ref") {}
    virtual ~GatherND8KernelRef() {}
    virtual JitConstants GetJitConstants(const gather_nd8_params& params) const;
    virtual CommonDispatchData SetDefault(const gather_nd8_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
