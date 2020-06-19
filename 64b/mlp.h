#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../intrinsics/vima/vima.hpp"

typedef struct base_type {
    __v64d *base;
    __v32u *label;
} base_type;

int training_instances, training_features, n_instances, n_vectors, VSIZE, vector_size, base_size, output_size, hidden_size, o_vectors;
__v64d *bias;
clock_t total_begin, total_end, read_begin, read_end, hidden_begin, hidden_end, output_begin, output_end, norm_begin, norm_end, class_begin, class_end;
double total_spent, read_spent, hidden_spent, output_spent, norm_spent, class_spent;
