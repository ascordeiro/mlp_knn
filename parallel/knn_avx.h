#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include <immintrin.h>

#define AVX_SIZE 16

float *tr_base;
u_int32_t *tr_label;

int training_instances, training_features, test_instances, k_neighbors, base_size;
clock_t read_begin, read_end, ed_begin, ed_end, class_begin, class_end, total_begin, total_end, inst_begin, inst_end;
double total_spent, read_spent, ed_spent = 0.0, class_spent;
