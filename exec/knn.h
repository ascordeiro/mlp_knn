#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../intrinsics/vima/vima.hpp"

__v32f *tr_base;
__v32u *tr_label;
int vector_size, VSIZE, k_neighbors, training_instances, training_features, test_instances, label_instances, tr_base_size, n_vectors, n_instances;