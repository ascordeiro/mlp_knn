#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../intrinsics/vima/vima.hpp"

__v32f *bias, sum;
int instances, features, instance_size, n_vectors, VSIZE, vector_size, alignment, inst_in_vec, output_size, hidden_size, o_vectors;
// clock_t total_begin, total_end, hidden_begin, hidden_end, output_begin, output_end, class_begin, class_end, init_begin, init_end;
// double total_spent, hidden_spent, output_spent, class_spent, init_spent, aux_spent;
