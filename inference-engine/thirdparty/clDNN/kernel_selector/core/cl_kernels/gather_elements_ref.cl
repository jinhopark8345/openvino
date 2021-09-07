
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
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

    // Calculate indice index
#if INPUT1_DIMS == 4
    #define ORDER b,f,y,x
    const uint x = dim0;
    const uint y = dim1;
#elif INPUT1_DIMS == 5
    #define ORDER b,f,z,y,x
    const uint x = dim0;
    const uint y = dim1 % OUTPUT_SIZE_Y;
    const uint z = dim1 / OUTPUT_SIZE_Y;
#else
    #define ORDER b,f,w,z,y,x
    const uint x = dim0 % OUTPUT_SIZE_X;
    const uint y = dim0 / OUTPUT_SIZE_X;
    const uint z = dim1 % OUTPUT_SIZE_Z;
    const uint w = dim1 / OUTPUT_SIZE_Z;
#endif
    const uint f = dim2 % OUTPUT_FEATURE_NUM;
    const uint b = dim2 / OUTPUT_FEATURE_NUM;

    const int out_idx = GET_OUTPUT_INDEX(INPUT1, ORDER);

#if INPUT1_DIMS == 4
    size_t data_shape[4] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    size_t indices_shape[4] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Y, INPUT1_SIZE_X};
#elif INPUT1_DIMS == 5
    size_t data_shape[5] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    size_t indices_shape[5] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
#else
    size_t data_shape[6] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    size_t indices_shape[6] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_W, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
#endif

    size_t max_inner_sum = 1,
           max_outer_sum = 1,
           outer_sum_inc_data = 1,
           outer_sum_inc_indices = 1;

    // size_t data_shape[6] = {INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_W, INPUT0_SIZE_Z, INPUT0_SIZE_Y, INPUT0_SIZE_X};
    // size_t data_shape[6] = {1, 2, 1, 5, 2, 4};

    // size_t indices_shape[6] = {INPUT1_BATCH_NUM, INPUT1_FEATURE_NUM, INPUT1_SIZE_W, INPUT1_SIZE_Z, INPUT1_SIZE_Y, INPUT1_SIZE_X};
    // size_t indices_shape[6] = {1, 2, 1, 2, 2, 4};

    /*
        step 1: set output target
            {0,1,0,1,1, {0,1,2,3}}

        step 2: where to get data from(update axis part with indices) // look-up : indices
            -> {0,1,0, {1,0,2,4}, 1, {0,1,2,3}}

        step 3: get the each data from data array // look-up : data
            -> {0,1,0,1,1,0} = {75} // 40 * 1 + 8 * 1 + 4 * 1
                        // outer_sum_inc_data(40) *
            -> {0,1,0,0,1,1} = {38} // 40 * 1 + 4 * 1 + 1
            -> {0,1,0,2,1,2} = {17} // 40 * 1 + 8 * 2 + 4 * 1 + 2
            -> {0,1,0,4,1,3} = {43} // 40 * 1 + 8 * 4 + 4 * 1 + 3

    outer_sum_inc_indices == 16
    outer_sum_inc_data == 40 => outer_sum == (0 || 1) * 40
    max_inner_sum == 8

    idx = outer_sum(0 || 40) + max_unner_sum(8) * indices[out_idx] + inner_sum(0~7)
    */


    for (size_t i = AXIS + 1; i < INPUT1_DIMS; i++)
        max_inner_sum *= indices_shape[i]; // max_inner_sum == 2*4

    for (int i = 0; i < AXIS; i++)
        max_outer_sum *= indices_shape[i]; // max_outer_sum == 2

    for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
        outer_sum_inc_data *= data_shape[i]; // outer_sum_inc_data == 5 * 2 * 4 == 40
    }
    max_outer_sum *= outer_sum_inc_data; // max_outer_sum == 2 * 40 == 80

    for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
        outer_sum_inc_indices *= indices_shape[i]; // outer_sum_inc_indices == 2*2*4 == 16
    }

    size_t outer_sum = (out_idx / outer_sum_inc_indices) * outer_sum_inc_data; // calculate the outter part, outer_sum = (0 || 1) * 40
    size_t inner_sum = out_idx % max_inner_sum; // calculate the inner part of the sum,

    uint idx = outer_sum + max_inner_sum * indices[out_idx] + inner_sum;
    INPUT0_TYPE val = data[idx];

#ifdef JINHO_DEBUG
    printf("%d %d %d\n", dim2, dim1, dim0);

    if(dim0 == 2 && dim1 == 1 && dim2 == 0 ){
        printf("\n----------------------------------\n");
        printf("AXIS: %d\n", AXIS);
        printf("INPUT1_DIMS: %d\n", INPUT1_DIMS);

        for (size_t i = AXIS + 1; i < INPUT1_DIMS; i++){
            printf("indices_shape[%d]: %d\n", i,indices_shape[i]);
        }
        printf("max_inner_sum: %d \n", max_inner_sum);

        for (int i = 0; i < AXIS; i++){
            printf("indices_shape[%d]: %d\n", i,indices_shape[i]);
        }

        for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
            printf("data_shape[%d]: %d\n", i, data_shape[i]);
        }
        printf("outer_sum_inc_data: %d \n", outer_sum_inc_data);

        printf("max_outer_sum: %d \n", max_outer_sum);

        for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
            printf("indices_shape[%d]: %d\n", i, indices_shape[i]);
        }
        printf("outer_sum_inc_indices: %d \n", outer_sum_inc_indices);

        printf("out_idx: %d \n", out_idx);

        printf("outer_sum: %zu \n", outer_sum);
        printf("inner_sum: %zu \n", inner_sum);
        printf("idx: %u \n", idx);
        printf("val: %lf \n", val);
    }

    if(dim0 == 4 && dim1 == 1 && dim2 == 0 ){
        printf("\n----------------------------------\n");
        printf("AXIS: %d\n", AXIS);
        printf("INPUT1_DIMS: %d\n", INPUT1_DIMS);

        for (size_t i = AXIS + 1; i < INPUT1_DIMS; i++){
            printf("indices_shape[%d]: %d\n", i,indices_shape[i]);
        }
        printf("max_inner_sum: %d \n", max_inner_sum);

        for (int i = 0; i < AXIS; i++){
            printf("indices_shape[%d]: %d\n", i,indices_shape[i]);
        }

        for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
            printf("data_shape[%d]: %d\n", i, data_shape[i]);
        }
        printf("outer_sum_inc_data: %d \n", outer_sum_inc_data);

        printf("max_outer_sum: %d \n", max_outer_sum);

        for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
            printf("indices_shape[%d]: %d\n", i, indices_shape[i]);
        }
        printf("outer_sum_inc_indices: %d \n", outer_sum_inc_indices);

        printf("out_idx: %d \n", out_idx);

        printf("outer_sum: %zu \n", outer_sum);
        printf("inner_sum: %zu \n", inner_sum);
        printf("idx: %u \n", idx);
        printf("val: %lf \n", val);
    }

    if(dim0 == 0 && dim1 == 1 && dim2 == 1 ){

        printf("\n----------------------------------\n");
        printf("AXIS: %d\n", AXIS);
        printf("INPUT1_DIMS: %d\n", INPUT1_DIMS);

        for (size_t i = AXIS + 1; i < INPUT1_DIMS; i++){
            printf("indices_shape[%d]: %d\n", i,indices_shape[i]);
        }
        printf("max_inner_sum: %d \n", max_inner_sum);

        for (int i = 0; i < AXIS; i++){
            printf("indices_shape[%d]: %d\n", i,indices_shape[i]);
        }

        for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
            printf("data_shape[%d]: %d\n", i, data_shape[i]);
        }
        printf("outer_sum_inc_data: %d \n", outer_sum_inc_data);

        printf("max_outer_sum: %d \n", max_outer_sum);

        for (size_t i = AXIS; i < INPUT1_DIMS; i++) {
            printf("indices_shape[%d]: %d\n", i, indices_shape[i]);
        }
        printf("outer_sum_inc_indices: %d \n", outer_sum_inc_indices);

        printf("out_idx: %d \n", out_idx);

        printf("outer_sum: %zu \n", outer_sum);
        printf("inner_sum: %zu \n", inner_sum);
        printf("idx: %u \n", idx);
        printf("val: %lf \n", val);
    }
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[out_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif


}

#undef ORDER
#undef GET_OUTPUT_INDEX
