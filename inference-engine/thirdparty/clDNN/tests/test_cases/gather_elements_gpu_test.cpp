// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/primitives/gather_elements.hpp>


using namespace cldnn;
using namespace ::tests;

inline void DoTest(engine& engine,
    const cldnn::memory::ptr& input0, // data
    const cldnn::memory::ptr& input1, // indices
    const std::vector<float>& expected_results,
    const tensor& output_tensor,
    const cldnn::gather_elements::gather_elements_axis axis) {
    topology topology;
    topology.add(input_layout("InputData", input0->get_layout()));
    topology.add(input_layout("InputIndices", input1->get_layout()));
    topology.add(
        gather_elements("gather_elements", "InputData", "InputIndices", input1->get_layout().format, output_tensor, axis)
    );

    network network(engine, topology);

    network.set_input_data("InputData", input0);
    network.set_input_data("InputIndices", input1);
    auto outputs = network.execute();
    auto output = outputs.at("gather_elements").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < expected_results.size(); ++i) {
        EXPECT_EQ(expected_results[i], float16_to_float32(output_ptr[i]));
    }
}


// this is a real test case for gather_elements, above are test cases from gather_nd
// TEST(gather_elements_gpu_fp16, d22_i22_a0) {
//     auto& engine = get_test_engine();

//     auto axis = cldnn::gather_elements::gather_elements_axis::along_b;
//     auto input0 = engine.allocate_memory({ data_types::f16, format::bf, { 2, 2 } }); // data
//     auto input1 = engine.allocate_memory({ data_types::f16, format::bf, { 2, 2} }); // indices

//     set_values(input0, {
//         FLOAT16(1), FLOAT16(2),
//         FLOAT16(3), FLOAT16(4),
//     });

//     set_values(input1, {
//         FLOAT16(0), FLOAT16(1),
//         FLOAT16(0), FLOAT16(0),
//     });

//     std::vector<float> expected_results = {
//         FLOAT16(1), FLOAT16(4),
//         FLOAT16(1), FLOAT16(2),
//     };

//     DoTest(engine, input0, input1, expected_results, tensor(2, 2), axis);

// }

// edited test cases from: TEST(gather_elements_gpu_fp16, d124251_i124221_an3)
TEST(gather_elements_gpu_fp16, d124251_i124221_an3) {
    auto& engine = get_test_engine();

    auto axis = cldnn::gather_elements::gather_elements_axis::along_z;
    auto input0 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 4, 2, 5, 1 } }); // data
    auto input1 = engine.allocate_memory({ data_types::f16, format::bfwzyx, { 1, 2, 4, 2, 2, 1 } }); // indices

    set_values(input0, {
        FLOAT16(33), FLOAT16(32), FLOAT16(28), FLOAT16(27),
        FLOAT16(34), FLOAT16(31), FLOAT16(29), FLOAT16(26),
        FLOAT16(35), FLOAT16(30), FLOAT16(64), FLOAT16(25),
        FLOAT16(1), FLOAT16(36), FLOAT16(65), FLOAT16(56),
        FLOAT16(2), FLOAT16(3), FLOAT16(66), FLOAT16(24),
        FLOAT16(57), FLOAT16(4), FLOAT16(23), FLOAT16(55),
        FLOAT16(61), FLOAT16(62), FLOAT16(63), FLOAT16(67),
        FLOAT16(58), FLOAT16(5), FLOAT16(80), FLOAT16(54),
        FLOAT16(59), FLOAT16(6), FLOAT16(81), FLOAT16(68),
        FLOAT16(60), FLOAT16(7), FLOAT16(22), FLOAT16(53),
        FLOAT16(8), FLOAT16(37), FLOAT16(21), FLOAT16(69),
        FLOAT16(9), FLOAT16(38), FLOAT16(78), FLOAT16(51),
        FLOAT16(10), FLOAT16(39), FLOAT16(20), FLOAT16(52),
        FLOAT16(75), FLOAT16(76), FLOAT16(19), FLOAT16(49),
        FLOAT16(11), FLOAT16(77), FLOAT16(18), FLOAT16(47),
        FLOAT16(73), FLOAT16(12), FLOAT16(17), FLOAT16(50),
        FLOAT16(74), FLOAT16(13), FLOAT16(40), FLOAT16(46),
        FLOAT16(72), FLOAT16(14), FLOAT16(16), FLOAT16(45),
        FLOAT16(71), FLOAT16(15), FLOAT16(41), FLOAT16(44),
        FLOAT16(70), FLOAT16(79), FLOAT16(42), FLOAT16(43),
    });

    set_values(input1, {
        FLOAT16(0), FLOAT16(2), FLOAT16(4), FLOAT16(3),
        FLOAT16(4), FLOAT16(0), FLOAT16(0), FLOAT16(1),
        FLOAT16(4), FLOAT16(0), FLOAT16(1), FLOAT16(0),
        FLOAT16(1), FLOAT16(0), FLOAT16(1), FLOAT16(1),
        FLOAT16(3), FLOAT16(1), FLOAT16(4), FLOAT16(2),
        FLOAT16(4), FLOAT16(2), FLOAT16(1), FLOAT16(3),
        FLOAT16(2), FLOAT16(1), FLOAT16(2), FLOAT16(4),
        FLOAT16(1), FLOAT16(0), FLOAT16(2), FLOAT16(4),
    });

    std::vector<float> expected_results = {
        FLOAT16(33), FLOAT16(3), FLOAT16(81), FLOAT16(67),
        FLOAT16(60), FLOAT16(34), FLOAT16(31), FLOAT16(56),
        FLOAT16(59), FLOAT16(32), FLOAT16(64), FLOAT16(27),
        FLOAT16(35), FLOAT16(32), FLOAT16(64), FLOAT16(25),
        FLOAT16(61), FLOAT16(30), FLOAT16(81), FLOAT16(24),
        FLOAT16(59), FLOAT16(3), FLOAT16(64), FLOAT16(67),
        FLOAT16(2), FLOAT16(30), FLOAT16(66), FLOAT16(68),
        FLOAT16(35), FLOAT16(32), FLOAT16(66), FLOAT16(68),
    };

    DoTest(engine, input0, input1, expected_results, tensor(1, 2, 4, 2, 2, 1), axis);
}
