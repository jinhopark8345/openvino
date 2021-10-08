// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class gather_nd8_kernel_selector : public kernel_selector_base {
public:
    static gather_nd8_kernel_selector& Instance() {
        static gather_nd8_kernel_selector instance_;
        return instance_;
    }

    gather_nd8_kernel_selector();

    virtual ~gather_nd8_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector
