#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
// #include <time.h>
#include "../../intrinsics/vima/vima.hpp"

__v32f *tr_base;
__v32u *tr_label;
int vector_size, VSIZE, k_neighbors, training_instances, training_features, test_instances, label_instances, tr_base_size, n_vectors, n_instances;
// clock_t read_begin, read_end, ed_begin, ed_end, class_begin, class_end, total_begin, total_end, inst_begin, inst_end, i_begin, i_end;
// double total_spent, read_spent, ed_spent = 0.0, class_spent = 0.0;
