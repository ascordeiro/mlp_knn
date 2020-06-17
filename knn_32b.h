#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include "../intrinsics_update/vima/vima.hpp"

typedef struct base_type {
    __v32f *base;
    __v32u *label;
} base_type;

int train_lines = 0, test_lines = 0, columns = 0, len_ivector, instances, tr_ientries, te_ientries, v_neg = 0, f_neg = 0, v_pos = 0, f_pos = 0, vector_size;
int VSIZE, k;
int training_instances, training_features, v_trsize, ed_idx = 0, label_instances;
clock_t read_begin, read_end, ed_begin, ed_end, class_begin, class_end, total_begin, total_end;
double total_spent, read_spent, ed_spent = 0.0, class_spent;
