#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <time.h>
#include "vima.hpp"
// #include "chive.hpp"

typedef struct base_type {
    __v64d *base;
    __v32u *label;
} base_type;

int train_lines = 0, test_lines = 0, columns = 0, len_ivector, instances, tr_ientries, te_ientries, v_neg = 0, f_neg = 0, v_pos = 0, f_pos = 0, vector_size;
int VSIZE, k;
int training_instances, training_features, v_trsize, ed_idx = 0, label_instances;

// void euclidean_distance_256b(__v32d *train_base, __v32d *test_base, __v32d **e_distance, __v32d *partial_sub, __v32d *partial_mul, int n_vectest, int index)
// {
//     __v32d partial_sum = 0.0, sum = 0.0;
//     int i, j;
//     for (i = index, j = 0; i < index + (n_vectest * VSIZE) && j < (n_vectest * VSIZE); i += VSIZE, j += VSIZE)
//     {
//         _vim32_dsubs(&train_base[i], &test_base[j], &partial_sub[j]);
//         _vim32_dmuls(&partial_sub[j], &partial_sub[j], &partial_mul[j]);
//     }
//     if (training_features < VSIZE)
//     {
//         for (i = 0; i < VSIZE / training_features; ++i)
//         {
//             partial_sum = 0.0;
//             for (j = i * training_features; j < (i * training_features) + training_features; ++j)
//             {
//                 partial_sum += partial_mul[j];
//             }
//             e_distance[i][ed_idx] = sqrt(partial_sum);
//         }
//         ed_idx++;
//     }
//     else
//     {
//         for (i = 0; i < VSIZE * n_vectest; i += VSIZE)
//         {
//             _vim32_dcsum(&partial_mul[i], &partial_sum);
//             sum += partial_sum;
//         }
//         e_distance[0][ed_idx++] = sqrt(sum);
//     }
// }

// void euclidean_distance_8Kb(__v32d *train_base, __v32d *test_base, __v32d **e_distance, __v32d *partial_sub, __v32d *partial_mul, int n_vectest, int index)
// {
//     __v32d partial_sum = 0.0, sum = 0.0;
//     int i, j;
//     for (i = index, j = 0; i < index + (n_vectest * VSIZE) && j < (n_vectest * VSIZE); i += VSIZE, j += VSIZE)
//     {
//         _vim1K_dsubs(&train_base[i], &test_base[j], &partial_sub[j]);
//         _vim1K_dmuls(&partial_sub[j], &partial_sub[j], &partial_mul[j]);
//     }
//     if (training_features < VSIZE)
//     {
//         for (i = 0; i < VSIZE / training_features; ++i)
//         {
//             partial_sum = 0.0;
//             for (j = i * training_features; j < (i * training_features) + training_features; ++j)
//             {
//                 partial_sum += partial_mul[j];
//             }
//             e_distance[i][ed_idx] = sqrt(partial_sum);
//         }
//         ed_idx++;
//     }
//     else
//     {
//         for (i = 0; i < VSIZE * n_vectest; i += VSIZE)
//         {
//             _vim1K_dcsum(&partial_mul[i], &partial_sum);
//             sum += partial_sum;
//         }
//         e_distance[0][ed_idx++] = sqrt(sum);
//     }
// }
