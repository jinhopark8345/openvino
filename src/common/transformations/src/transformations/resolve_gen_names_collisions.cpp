// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <numeric>

#include "transformations/resolve_gen_names_collisions.hpp"

bool ov::pass::ResolveGeneratedNameCollisions::run_on_model(const std::shared_ptr<ov::Model>& model) {
    // Next containers are used to fix collisions in autogenerated names
    // Collect all unique friendly names
    std::unordered_set<std::string> unique_friendly_names;
    // Save nodes with autogenerated names but without conflicts in the candidate list
    std::unordered_map<std::string, Node*> nodes_with_possible_conflicts;
    // The final list of nodes with collisions
    std::vector<Node*> nodes_with_conflicts;

    for (auto& node : model->get_ordered_ops()) {
        // Detect names collisions only for nodes with autogenerated names
        const auto friendly_name = node->get_friendly_name();
        if (unique_friendly_names.find(friendly_name) == unique_friendly_names.end()) {
            unique_friendly_names.insert(friendly_name);
            if (node->m_friendly_name.empty())
                nodes_with_possible_conflicts[friendly_name] = node.get();
        } else if (node->m_friendly_name.empty()) {
            // We have a conflict with autogenerated name
            nodes_with_conflicts.emplace_back(node.get());
        } else if (nodes_with_possible_conflicts.find(friendly_name) != nodes_with_possible_conflicts.end()) {
            // We have a conflict with autogenerated name
            nodes_with_conflicts.emplace_back(nodes_with_possible_conflicts[friendly_name]);
        }
    }

    // Resolve names collisions
    for (const auto& node : nodes_with_conflicts) {
        size_t idx = 2;
        const auto friendly_name = node->get_friendly_name();
        while (unique_friendly_names.find(friendly_name + "_" + std::to_string(idx)) != unique_friendly_names.end())
            idx++;
        const auto new_friendly_name = friendly_name + "_" + std::to_string(idx);
        node->set_friendly_name(new_friendly_name);
        unique_friendly_names.insert(new_friendly_name);
    }
    return true;
}

