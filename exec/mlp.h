#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../../intrinsics/vima/vima.hpp"

__v32f *bias;
int training_instances, training_features, n_vectors, VSIZE, vector_size, inst_in_vec, output_size, hidden_size, o_vectors;
clock_t total_begin, total_end, hidden_begin, hidden_end, output_begin, output_end, class_begin, class_end;
double total_spent, hidden_spent, output_spent, class_spent;
