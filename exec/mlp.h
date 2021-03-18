#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../intrinsics/vima/vima.hpp"

__v32f *bias, sum;
int instances, features, instance_size, n_vectors, VSIZE, vector_size, alignment, inst_in_vec, output_size, hidden_size, o_vectors;