# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.elementwise import Add
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes


class FakeOutputResolver(BackReplacementPattern):
    """
    This transformation removes FakeOutput nodes. If producer of FakeOutput have only one consumer (FakeOutput itself)
     the name of FakeOutput is inherited by its producer, otherwise FakeOutput is replaced with op which does nothing.
    """
    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for fake_output in graph.get_op_nodes(op='FakeOutput'):
            name = fake_output.soft_get('name', fake_output.id)

            producer = fake_output.in_port(0).get_source().node
            producer_outputs = 0
            for port in producer.out_ports().values():
                if not port.disconnected():
                    producer_outputs += 1
            if producer_outputs != 1:
                # At this stage we don't know the type of output, so we rely on MO transformation which updates the
                # Const type for elementwise operations in case of input data types mismatch
                add = create_op_with_const_inputs(graph, Add, {1: int64_array(0)}, {'can_be_fused': False})
                rename_nodes([(fake_output, name + '/TBD'), (add, name)])

                # Get tensor names incoming to FakeOutput
                tensor_names = fake_output.in_port(0).get_connection().source.get_tensor_names()

                # Remove tensor info from data node
                in_data_node = fake_output.in_node()
                if 'fw_tensor_debug_info' in in_data_node:
                    del in_data_node['fw_tensor_debug_info']

                fake_output.in_port(0).get_connection().set_destination(add.in_port(0))
                fake_output.out_port(0).get_connection().set_source(add.out_port(0))

                # Move tensor names to Add op, which replaces FakeOutput
                if len(tensor_names) > 0:
                    add.out_port(0).add_tensor_names([add.name], [tensor_names])

            else:
                result_in_port = fake_output.out_port(0).get_destination()
                result_in_port.disconnect()
                fake_output.in_port(0).get_connection().set_destination(result_in_port)
                rename_nodes([(fake_output, name + '/TBD'), (producer, name)])
