#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vima.hpp"

typedef struct base_type {
    __v64d *base;
    __v32u *label;
} base_type;

int training_instances, training_features, n_instances, n_vectors, VSIZE, vector_size, base_size, output_size, hidden_size, o_vectors;
__v64d *bias;
