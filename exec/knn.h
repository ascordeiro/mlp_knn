#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include "../../intrinsics/vima/vima.hpp"

__v32f *tr_base;
__v32u *tr_label;
int vector_size, VSIZE, k, training_instances, training_features, label_instances;
clock_t read_begin, read_end, ed_begin, ed_end, class_begin, class_end, total_begin, total_end;
double total_spent, read_spent, ed_spent = 0.0, class_spent;
