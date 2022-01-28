// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <openvino/core/partial_shape.hpp>

#include "openvino/core/type/element_type.hpp"
#include "xml_parse_utils.h"

namespace ov {
void operator>>(const std::stringstream& in, ov::element::Type& type);

bool getStrAttribute(const pugi::xml_node& node, const std::string& name, std::string& value);
Dimension str_to_dimension(const std::string& value);
PartialShape str_to_partial_shape(const std::string& value);
bool get_dimension_from_attribute(const pugi::xml_node& node, const std::string& name, Dimension& value);
bool get_partial_shape_from_attribute(const pugi::xml_node& node, const std::string& name, PartialShape& value);

template <class T>
void str_to_container(const std::string& value, T& res) {
    std::stringstream ss(value);
    std::string field;
    while (getline(ss, field, ',')) {
        if (field.empty())
            IE_THROW() << "Cannot get vector of parameters! \"" << value << "\" is incorrect";
        std::stringstream fs(field);
        typename T::value_type val;
        fs >> val;
        res.insert(res.end(), val);
    }
}

template <class T>
bool getParameters(const pugi::xml_node& node, const std::string& name, std::vector<T>& value) {
    std::string param;
    if (!getStrAttribute(node, name, param))
        return false;
    str_to_container(param, value);
    return true;
}

template <class T>
T stringToType(const std::string& valStr) {
    T ret{0};
    std::istringstream ss(valStr);
    if (!ss.eof()) {
        ss >> ret;
    }
    return ret;
}
}  // namespace ov
