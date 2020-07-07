#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../intrinsics_update/vima/vima.hpp"
#include <immintrin.h>

typedef struct base_type {
    __v32f *base;
    __v32u *label;
} base_type;

int training_instances, training_features, n_instances, n_vectors, VSIZE, vector_size, base_size, output_size, hidden_size, o_vectors;
__v32f *bias;
clock_t total_begin, total_end, read_begin, read_end, hidden_begin, hidden_end, output_begin, output_end, norm_begin1, norm_end1, class_begin, class_end;
double total_spent, read_spent, hidden_spent, output_spent, norm_spent1, norm_spent2, class_spent, norm_begin2, norm_end2;
