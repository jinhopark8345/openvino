// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "cldnn/primitives/gather_nd8.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<gather_nd8> : public typed_program_node_base<gather_nd8> {
    using parent = typed_program_node_base<gather_nd8>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using gather_nd8_node = typed_program_node<gather_nd8>;

template <>
class typed_primitive_inst<gather_nd8> : public typed_primitive_inst_base<gather_nd8> {
    using parent = typed_primitive_inst_base<gather_nd8>;

public:
    static layout calc_output_layout(gather_nd8_node const& node);
    static std::string to_string(gather_nd8_node const& node);

public:
    typed_primitive_inst(network& network, gather_nd8_node const& desc);
};

using gather_nd8_inst = typed_primitive_inst<gather_nd8>;
}  // namespace cldnn
