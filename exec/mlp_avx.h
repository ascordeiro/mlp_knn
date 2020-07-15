#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

#define AVX_SIZE 16

int training_instances, training_features, base_size, output_size, hidden_size;
float *bias, *base;
__uint32_t *label;

clock_t total_begin, total_end, read_begin, read_end, hidden_begin, hidden_end, output_begin, output_end, norm_begin1, norm_end1, class_begin, class_end;
double total_spent, read_spent, hidden_spent, output_spent, norm_spent1, norm_spent2, class_spent, norm_begin2, norm_end2;
