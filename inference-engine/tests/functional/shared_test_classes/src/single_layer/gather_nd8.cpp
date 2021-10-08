// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/single_layer/gather_nd8.hpp"

namespace LayerTestsDefinitions {

std::string GatherND8LayerTest::getTestCaseName(const testing::TestParamInfo<GatherND8Params>& obj) {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int batchDims;
    std::string device;
    Config config;
    GatherND8ParamsSubset gatherArgsSubset;
    std::tie(gatherArgsSubset, dPrecision, iPrecision, device, config) = obj.param;
    std::tie(dataShape, indicesShape, batchDims) = gatherArgsSubset;

    std::ostringstream result;
    result << "DS=" << CommonTestUtils::vec2str(dataShape) << "_";
    result << "IS=" << CommonTestUtils::vec2str(indicesShape) << "_";
    result << "BD=" << batchDims << "_";
    result << "DP=" << dPrecision.name() << "_";
    result << "IP=" << iPrecision.name() << "_";
    result << "device=" << device;
    if (!config.empty()) {
        result << "_config=";
        for (const auto& cfg : config) {
            result << "{" << cfg.first << ": " << cfg.second << "}";
        }
    }

    return result.str();
}

void GatherND8LayerTest::SetUp() {
    InferenceEngine::SizeVector dataShape, indicesShape;
    InferenceEngine::Precision dPrecision, iPrecision;
    int batchDims;
    GatherND8ParamsSubset gatherArgsSubset;
    std::tie(gatherArgsSubset, dPrecision, iPrecision, targetDevice, configuration) = this->GetParam();
    std::tie(dataShape, indicesShape, batchDims) = gatherArgsSubset;

    auto ngDPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(dPrecision);
    auto ngIPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iPrecision);

    auto params = ngraph::builder::makeParams(ngDPrc, {dataShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto dataNode = paramOuts[0];
    auto gather = std::dynamic_pointer_cast<ngraph::opset8::GatherND8>(
            ngraph::builder::makeGatherND8(dataNode, indicesShape, ngIPrc, batchDims));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(gather)};
    function = std::make_shared<ngraph::Function>(results, params, "gatherND8");
}
}  // namespace LayerTestsDefinitions
