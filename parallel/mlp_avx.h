#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#define AVX_SIZE 16

int instances, features, output_size, hidden_size;
float *bias;
