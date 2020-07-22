#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../../intrinsics/vima/vima.hpp"

// typedef struct base_type {
//     __v32f *base;
//     __v32u *label;
// } base_type;

__v32u *label;
__v32f *bias, *base;
int training_instances, training_features, n_instances, n_vectors, VSIZE, vector_size, inst_in_vec, base_size, output_size, hidden_size, o_vectors;
clock_t total_begin, total_end, read_begin, read_end, hidden_begin, hidden_end, output_begin, output_end, class_begin, class_end;
double total_spent, read_spent, hidden_spent, output_spent, class_spent;
