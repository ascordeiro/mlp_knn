#include "knn.h"

void read_files(char const *argv[], base_type *train_base) {
    int i, j, k, l;
    read_begin = clock();
    vector_size = atoi(argv[1]);
    if (vector_size == 256) {
        VSIZE = VM32L;
    } else {
        VSIZE = VM1KL;
    }

    FILE *f = fopen(argv[2], "r");
    if (fscanf(f, "%d", &training_instances));
    if (fscanf(f, "%d\n", &training_features));

    int n_vectors = training_features/VSIZE;
    int n_copies = 1;
    if (training_features < VSIZE) {
        n_copies = VSIZE/training_features;
        n_vectors = 1;
    }

    label_instances = 1;
    if (training_instances > VSIZE) {
        label_instances = training_instances/VSIZE;
    }

    train_base->base = (__v64d *)malloc(sizeof(__v64d) * training_instances * n_vectors * VSIZE);
    train_base->label = (__v32u *)malloc(sizeof(__v32u) * label_instances * VSIZE);
    if (!train_base->base || !train_base->label) {
        printf("Cannot allocate training base\n");
        exit(1);
    }

    for (i = 0; i < training_instances; ++i) {
        if (fscanf(f, "%u ", &train_base->label[i]));
        for (j = 0; j < training_features; ++j) {
            if (fscanf(f, "%lf ", &train_base->base[i * VSIZE + j]));
        }
        for (j = 1; j < n_copies; ++j) {
            for (k = j * training_features, l = 0; l < training_features; ++k, ++l) {
                train_base->base[i * VSIZE + k] = train_base->base[i * VSIZE + l];
            }
        }
    }
    fclose(f);
    read_end = clock();
    read_spent = (double)(read_end - read_begin) / CLOCKS_PER_SEC;
}

void votes(__v32u *knn, __v32u test_label, int k) {
    int i, pos = 0, neg = 0;
    for (i = 0; i < k; ++i) {
        if (knn[i] == 1) {
            pos++;
        } else {
            neg++;
        }
    }
    if (pos > neg) {
        printf("%s\n", "pos");
    } else {
        printf("%s\n", "neg");
    }
}

void get_ksmallest(__v64d *array, __v32u *label, __v32u *knn, int k) {
    int idx = 0;
    for (int i = 0; i < k; ++i) {
        knn[i] = -1;
        __v64d min = array[0];
        for (int j = 1; j < training_instances; ++j) {
            if (min > array[j]) {
                idx = j;
            }
        }
        knn[i] = label[idx];
        array[idx] = 9999999.0;
    }
}

__v64d **create_mask(int n_instances) {
    __v64d **mask = (__v64d **)malloc(sizeof(__v64d *) * n_instances);
    for (int i = 0; i < n_instances; ++i) {
        mask[i] = (__v64d *)malloc(sizeof(__v64d) * VSIZE);
        for (int j = i * training_features; j < (i * training_features) + training_features; ++j) {
            mask[i][j] = 1.0;
        }
    }
    return mask;
}

// sqrt(pow((x1 - y1), 2) + pow((x2 - y2), 2) + ... + pow((xn - yn), 2))
void classification(char const *argv[], base_type *train_base) {
    int test_instances, test_features, i, ii, j, jj;
    __v64d partial_sum, sum;
    base_type test_base;

    k = atoi(argv[4]);

    FILE *f = fopen(argv[3], "r");
    if (fscanf(f, "%d ", &test_instances));
    if (fscanf(f, "%d\n", &test_features));

    int v_tesize = test_features/VSIZE;
    int n_instances = 1;
    if (test_features < VSIZE) {
        v_tesize = 1;
        n_instances = VSIZE/test_features;
    }

    test_base.base = (__v64d *)malloc(sizeof(__v64d) * v_tesize * VSIZE);
    test_base.label = (__v32u *)malloc(sizeof(__v32u) * n_instances);
    if (!test_base.base || !test_base.label) {
        printf("Cannot allocate train base\n");
        exit(1);
    }

    __v64d **e_distance = (__v64d **)malloc(sizeof(__v64d *) * n_instances);
    for (i = 0; i < n_instances; ++i) {
        e_distance[i] = (__v64d *)malloc(sizeof(__v64d) * training_instances);
    }
    __v64d *partial_sub = (__v64d *)malloc(sizeof(__v64d) * v_tesize * VSIZE);
    __v64d *partial_mul = (__v64d *)malloc(sizeof(__v64d) * v_tesize * VSIZE);
    __v64d *copy_mul = (__v64d *)malloc(sizeof(__v64d) * v_tesize * VSIZE);
    __v64d *partial_and = (__v64d *)malloc(sizeof(__v64d) * VSIZE);
    __v64d *partial_acc = (__v64d *)malloc(sizeof(__v64d) * VSIZE);
    __v64d *vand = (__v64d *)malloc(sizeof(__v64d) * VSIZE);
    __v64d **mask;
    if (test_features < VSIZE) {
        mask = create_mask(n_instances);
    }

    for (i = 0; i < test_instances; i += n_instances) {
        ed_idx = 0;
        for (j = 0; j < n_instances; ++j) {
            if (fscanf(f, "%u ", &test_base.label[j]));
            for (jj = j * test_features; jj < (j * test_features) + test_features; ++jj) {
                if (fscanf(f, "%lf ", &test_base.base[jj]));
            }
        }
        for (j = 0; j < training_instances * VSIZE * v_tesize; j += v_tesize * VSIZE) {
            sum = 0.0;
            if (vector_size == 256) {
                ed_begin = clock();
                for (jj = j, ii = 0; jj < j + (v_tesize * VSIZE) && ii < (v_tesize * VSIZE); jj += VSIZE, ii += VSIZE) {
                    _vim32_dsubs(&train_base->base[jj], &test_base.base[ii], &partial_sub[ii]);
                    _vim32_dmuls(&partial_sub[ii], &partial_sub[ii], &partial_mul[ii]);
                }
                if (test_features < VSIZE) {
                    for (jj = 0; jj < n_instances; ++jj) {
                        _vim32_dmuls(mask[jj], partial_mul, partial_acc);
                        _vim32_dcums(partial_mul, &partial_sum);
                        e_distance[jj][ed_idx] = sqrt(partial_sum);
                    }
                    ed_idx++;
                } else {
                    for (jj = 0; jj < VSIZE * v_tesize; jj += VSIZE) {
                        _vim32_dcums(&partial_mul[jj], &partial_sum);
                        sum += partial_sum;
                    }
                    e_distance[0][ed_idx++] = sqrt(sum);
                }
                ed_end = clock();
            } else {
                ed_begin = clock();
                for (jj = j, ii = 0; jj < j + (v_tesize * VSIZE) && ii < (v_tesize * VSIZE); jj += VSIZE, ii += VSIZE) {
                    _vim1K_dsubs(&train_base->base[jj], &test_base.base[ii], &partial_sub[ii]);
                    _vim1K_dmuls(&partial_sub[ii], &partial_sub[ii], &partial_mul[ii]);
                }
                if (test_features < VSIZE) {
                    for (jj = 0; jj < n_instances; ++jj) {
                        _vim1K_dmuls(&mask[jj][0], &partial_mul[0], &partial_acc[0]);
                        _vim1K_dcums(&partial_mul[0], &partial_sum);
                        e_distance[jj][ed_idx] = sqrt(partial_sum);
                    }
                    ed_idx++;
                } else {
                    for (jj = 0; jj < VSIZE * v_tesize; jj += VSIZE) {
                        _vim1K_dcums(&partial_mul[jj], &partial_sum);
                        sum += partial_sum;
                    }
                    e_distance[0][ed_idx++] = sqrt(sum);
                }
                ed_end = clock();
                ed_spent += (double)(ed_end - ed_begin) / CLOCKS_PER_SEC;
            }
        }

        __v32u *knn = (__v32u *)malloc(sizeof(__v32u) * k);
        class_begin = clock();
        for (j = 0, jj = i; j < n_instances && jj < i + n_instances; ++j, ++jj) {
            get_ksmallest(e_distance[j], train_base->label, knn, k);
            printf("%u. ", jj);
            votes(knn, test_base.label[0], k);
        }
        class_end = clock();
        class_spent += (double)(class_end - class_begin) / CLOCKS_PER_SEC;

        free(knn);
    }
    fclose(f);
    for (i = 0; i < n_instances; i++) {
        free(e_distance[i]);
    }
    if (test_features < VSIZE) {
        for (i = 0; i < n_instances; i++) {
            free(mask[i]);
        }
        free(mask);
    }
    free(e_distance);
    free(partial_sub);
    free(partial_mul);
    free(partial_and);
    free(copy_mul);
    free(vand);
    free(test_base.base);
    free(test_base.label);
}

int main(int argc, char const *argv[]) {
    total_begin = clock();
    base_type train_base;

    // Initialize train and test matrix
    read_files(argv, &train_base);

    // // Calculates Euclidean Distance
    classification(argv, &train_base);

    total_end = clock();
    total_spent = (double)(total_end - total_begin) / CLOCKS_PER_SEC;
    printf("**************************************\n");
    printf("* Execution time:          %lfs *\n", total_spent);
    printf(" ************************************\n");
    printf("* Read time:               %lfs *\n", read_spent);
    printf("* Euclidean Distance time: %lfs *\n", ed_spent);
    printf("* Classification time:     %lfs *\n", class_spent);
    printf("**************************************\n");
    free(train_base.base);
    free(train_base.label);
    return 0;
}
