
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/data_types.cl"
#include "include/fetch_data.cl"

#define GET_OUTPUT_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)

KERNEL(gather_elements_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    for(uint dim2 = 0; dim2 < 1*2; dim2++){
        for(uint dim1 = 0; dim1 < 1*2; dim1++){
            for(uint dim0 = 0 ; dim0 < 2*4 ; dim0 ++ ){

                #define ORDER b,f,w,z,y,x
                const uint x = dim0 % OUTPUT_SIZE_X;
                const uint y = dim0 / OUTPUT_SIZE_X;
                const uint z = dim1 % OUTPUT_SIZE_Z;
                const uint w = dim1 / OUTPUT_SIZE_Z;

                const uint f = dim2 % OUTPUT_FEATURE_NUM;
                const uint b = dim2 / OUTPUT_FEATURE_NUM;

                const int out_idx = GET_OUTPUT_INDEX(INPUT1, ORDER);

                size_t data_shape[6] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
                size_t indices_shape[6] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_W, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};


                size_t max_inner_sum = 1, max_outer_sum = 1, outer_sum_inc_data = 1, outer_sum_inc_indices = 1;
                for (size_t i = AXIS + 1; i < INPUT1_DIMS; i++)
                    max_inner_sum *= indices_shape[i];

                for (int i = 0; i < AXIS; i++)
                    max_outer_sum *= indices_shape[i];

                for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
                    outer_sum_inc_data *= data_shape[i];
                }
                max_outer_sum *= outer_sum_inc_data;

                for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
                    outer_sum_inc_indices *= indices_shape[i];
                }

                size_t outer_sum = (out_idx / outer_sum_inc_indices) * outer_sum_inc_data;
                size_t inner_sum = out_idx % max_inner_sum;

                uint idx = outer_sum + max_inner_sum * indices[out_idx] + inner_sum;
                INPUT0_TYPE val = data[idx];

                output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
            }
        }
    }



}

#undef ORDER
#undef GET_OUTPUT_INDEX
