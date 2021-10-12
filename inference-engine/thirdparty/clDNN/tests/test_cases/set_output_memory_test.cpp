#include "test_utils.h"
#include <cldnn/primitives/arg_max_min.hpp>
#include <cldnn/primitives/mutable_data.hpp>
#include <cldnn/primitives/data.hpp>
#include <cldnn/primitives/input_layout.hpp>
#include <cldnn/graph/network.hpp>

using namespace cldnn;
using namespace ::tests;

TEST(top_k_layer_tests_jinho, second_input) {
    static const int32_t x_size = 2, y_size = 2, feature_num = 4, batch_num = 2;
    auto& engine = get_test_engine();
    const int top_k = 2;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ batch_num, feature_num, x_size , y_size } });
    auto second_input = engine.allocate_memory({ data_types::f32, format::bfyx, { top_k, feature_num, x_size , y_size } });
    auto final_output = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1 , 1 } });

    // arg_max is a parent of pred/sink_port_0
    // 12220_md_write is dependency of arg_max and
    // will be part of output chain but without impl

    topology topology;
    topology.add(input_layout("Add_1396", input->get_layout()));
    topology.add(cldnn::mutable_data("second_input", second_input));
    topology.add(cldnn::mutable_data("12220_md_write", final_output));
    topology.add(arg_max_min("arg_max", { "Add_1396", "12220_md_write", "second_input" }, arg_max_min::min, top_k, arg_max_min::batch));
    topology.add(cldnn::mutable_data("pred/sink_port_0", {"arg_max"},final_output) );

    std::vector<float> input_vec = {
            //y0x0 y0x1 y1x0 y1x1
            /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
            /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b0f3*/0.2f, 0.2f,  -10.f, 4.2f,

            /*b1f0*/3.f,  0.5f,  7.f,   10.f,
            /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
            /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f,
            /*b1f3*/4.f,  0.5f,  8.f,   8.2f
    };
    set_values(input, input_vec);
    auto prog = program::build_program(engine, topology, build_options());
    network network(prog, 0);
    network.set_input_data("Add_1396", input);

    // to make _reset_arguments false
    network.execute();
    network.execute();
    network.set_output_memory("pred/sink_port_0", final_output);
}
