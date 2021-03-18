#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#define AVX_SIZE 16

float *tr_base;
u_int32_t *tr_label;

int training_instances, training_features, test_instances, k_neighbors, base_size;

