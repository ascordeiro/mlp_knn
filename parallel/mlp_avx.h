#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include <time.h>
#include <immintrin.h>
#include <omp.h>

#define AVX_SIZE 16

int instances, features, output_size, hidden_size;
float *bias;

// clock_t total_begin, total_end, hidden_begin, hidden_end, output_begin, output_end, class_begin, class_end;
// double total_spent, hidden_spent, output_spent, class_spent;
