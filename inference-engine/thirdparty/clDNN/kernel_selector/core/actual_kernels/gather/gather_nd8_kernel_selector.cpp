// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_nd8_kernel_selector.h"
#include "gather_nd8_kernel_ref.h"

namespace kernel_selector {

gather_nd8_kernel_selector::gather_nd8_kernel_selector() { Attach<GatherND8KernelRef>(); }

KernelsData gather_nd8_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::GATHER_ND8);
}
}  // namespace kernel_selector
