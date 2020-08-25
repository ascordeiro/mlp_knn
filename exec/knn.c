#include "knn.h"

void read_files(char const *argv[]) {
    read_begin = clock();
    vector_size = atoi(argv[1]);
    if (vector_size == 256) {
        VSIZE = VM64I;
    } else {
        VSIZE = VM2KI;
    }

    training_instances = atoi(argv[2]);
    test_instances = atoi(argv[3]);
    training_features = atoi(argv[4]);

    n_vectors = training_features/VSIZE;
    if (training_features < VSIZE) {
        n_vectors = 1;
    }

    label_instances = 1;
    if (training_instances > VSIZE) {
        label_instances = training_instances/VSIZE;
    }

    tr_base_size = training_instances * n_vectors * VSIZE;
    // tr_base = (__v32f *)malloc(tr_base_size * sizeof(__v32f));
    // tr_label = (__v32u *)malloc(label_instances * VSIZE * sizeof(__v32u));

    tr_base = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * tr_base_size);
    tr_label = (__v32u *)aligned_alloc(vector_size, sizeof(__v32u)*training_instances);

    read_end = clock();
    read_spent = (double)(read_end - read_begin) / CLOCKS_PER_SEC;
}

void votes(__v32u *knn, int k) {
    int i, pos = 0, neg = 0;
    for (i = 0; i < k; ++i) {
        if (knn[i] == 1) {
            pos++;
        } else {
            neg++;
        }
    }
    if(test_instances - 1 == 255) {
        if (pos > neg) {
            printf("%s\n", "pos");
        } else {
            printf("%s\n", "neg");
        }
    }
}

void get_ksmallest(__v32f *array, __v32u *label, __v32u *knn, int k) {
    int idx = 0;
    for (int i = 0; i < k; ++i) {
        knn[i] = 9999999;
        __v32f min = array[0];
        for (int j = 1; j < training_instances; ++j) {
            if (min > array[j]) {
                idx = j;
            }
        }
        knn[i] = label[idx];
        array[idx] = 9999999.0;
    }
}

void read_test_instance(__v32f *base, int size) {
    if (vector_size == 256) {
        for (int i = 0; i < size; i += VSIZE) {
            _vim64_fmovs(1, &base[i]);
        }
    } else {
        // for (int i = 0; i < size; i += VSIZE) {
        for (int i = 0; i < size; i += 4) {
            base[i] = 3.5;
            base[i + 1] = 3.0;
            base[i + 2] = 2.5;
            base[i + 3] = 2.0;
            // _vim2K_fmovs(1, &base[i]);
        }
    }
}

inline __v32f **initialize_mask(int stride, int n_masks) {
    __v32f **mask = (__v32f **)malloc(n_masks * sizeof(__v32f *));
    for (int i = 0; i < n_masks; i++) {
        mask[i] = (__v32f *)malloc(VSIZE * sizeof(__v32f));
    }
    for (int i = 0; i < n_masks; i++) {
        for (int j = i * stride; j < (i * stride) + stride; j++) {
            mask[i][j] = 1.0;
        }
    }
    return mask;
}

// sqrt(pow((x1 - y1), 2) + pow((x2 - y2), 2) + ... + pow((xn - yn), 2))
void classification(char const *argv[]) {
    uint32_t i, j, jj, k;
    __v32f partial_sum;
    __v32f *te_base;

    k_neighbors = atoi(argv[5]);

    int n_instances = 1;
    if (training_features < VSIZE) {
        n_instances = VSIZE/training_features;
    }

    te_base = (__v32f *)aligned_alloc(vector_size, n_vectors * VSIZE * sizeof(__v32f));

    __v32f **e_distance = (__v32f **)aligned_alloc(vector_size, test_instances * sizeof(__v32f *));
    for (i = 0; i < test_instances; ++i) {
        e_distance[i] = (__v32f *)aligned_alloc(vector_size, training_instances * sizeof(__v32f));
    }
    __v32u *knn = (__v32u *)malloc(k_neighbors * sizeof(__v32u));
    __v32f *partial_sub = (__v32f *)malloc(sizeof(__v32f) * n_vectors * VSIZE);
    __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * n_vectors * VSIZE);
    __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);

    __v32f **mask;
    if (training_features < VSIZE) {
        mask = initialize_mask(training_features, n_instances);
    }

    //initializing training base
    read_test_instance(tr_base, tr_base_size);

    // ed_begin = clock();
    if (vector_size == 256) {
        if (training_features < VSIZE) {
            for (i = 0; i < test_instances; i += n_instances) {
                read_test_instance(te_base, VSIZE);
                for (j = 0; j < training_instances; j++) {
                    _vim64_fsubs(&tr_base[j * VSIZE], te_base, partial_sub);
                    _vim64_fmuls(partial_sub, partial_sub, partial_mul);
                    for (k = 0; k < n_instances; k++) {
                        _vim64_fmuls(partial_mul, mask[k], partial_acc);
                        _vim64_fcums(partial_acc, &e_distance[i + k][j]);
                    }
                }
            }
        } else {
            for (i = 0; i < test_instances; i++) {
                read_test_instance(te_base, n_vectors * VSIZE);
                for (j = 0; j < training_instances; j++) {
                    for (k = 0; k < n_vectors; k++) {
                        _vim64_fsubs(&tr_base[(j * VSIZE * n_vectors) + (k * VSIZE)], &te_base[k * VSIZE], partial_sub);
                        _vim64_fmuls(partial_sub, partial_sub, partial_mul);
                        _vim64_fcums(partial_mul, &partial_sum);
                        e_distance[i][j] += partial_sum;

                    }
                }
            }
        }
    } else {
        int test_vectors = test_instances/n_instances, ed_idx = 0;
        __v32f *p_acc;
        __v32f *tests;
        __v32f *p_sub = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * training_instances * test_vectors * VSIZE);
        __v32f *p_mul = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * training_instances * test_vectors * VSIZE);
        __v32f *e_dst = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * test_instances * training_instances);

        if (training_features < VSIZE) {

            if (training_features == 8) {
                p_acc = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * n_instances * VSIZE);
                tests = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * VSIZE);

                for (i = 0; i < tr_base_size; i += VSIZE) {
                        _vim2K_fmovs(1.0, tests);
                        _vim2K_fsubs(&tr_base[i], tests, &p_sub[i]);
                }
                for (i = 0; i < training_instances * VSIZE; i += VSIZE) {
                    _vim2K_fmuls(&p_sub[i], &p_sub[i], &p_mul[i]);
                }
                for (i = 0; i < training_instances * VSIZE; i += VSIZE) {
                    for (j = 0; j < n_instances; ++j) {
                        _vim2K_fmuls(&p_mul[i], mask[j], &p_acc[j * VSIZE]);
                    }
                    for (j = 0; j < n_instances; ++j) {
                        _vim2K_fcums(&p_acc[j * VSIZE], &e_dst[ed_idx++]);
                    }
                }

            } else {
                p_acc = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * 2 * n_instances * VSIZE);
                tests = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * 2 * VSIZE); 

                for (i = 0; i < tr_base_size; i += VSIZE) {
                    for (j = 0; j < test_instances; j += n_instances * 2) {
                        _vim2K_fmovs(1.0, &tests[0]);
                        _vim2K_fmovs(1.0, &tests[VSIZE]);
                        _vim2K_fsubs(&tr_base[i], &tests[0], &p_sub[i * 2]);
                        _vim2K_fsubs(&tr_base[i], &tests[VSIZE], &p_sub[i * 2 + VSIZE]);
                    }
                }
                for (i = 0; i < training_instances * VSIZE * test_vectors; i += VSIZE * 2) {
                    _vim2K_fmuls(&p_sub[i], &p_sub[i], &p_mul[i]);
                    _vim2K_fmuls(&p_sub[i + VSIZE], &p_sub[i + VSIZE], &p_mul[i + VSIZE]);
                }
                for (i = 0; i < training_instances * VSIZE * test_vectors; i += VSIZE * 2) {
                    for (j = 0; j < n_instances; ++j) {
                        _vim2K_fmuls(&p_mul[i], mask[j], &p_acc[j * 2 * VSIZE]);
                        _vim2K_fmuls(&p_mul[i + VSIZE], mask[j], &p_acc[(j * 2 * VSIZE + VSIZE)]);
                    }
                    for (j = 0; j < n_instances; ++j) {
                        _vim2K_fcums(&p_acc[j * VSIZE * 2], &e_dst[ed_idx]);
                        _vim2K_fcums(&p_acc[j * VSIZE * 2 + VSIZE], &e_dst[ed_idx + n_instances]);
                        ed_idx++;
                    }
                }
            }

            for (i = 0; i < test_instances * training_instances; i += 4) {
                e_dst[i] = sqrt(e_dst[i]);
                e_dst[i + 1] = sqrt(e_dst[i + 1]);
                e_dst[i + 2] = sqrt(e_dst[i + 2]);
                e_dst[i + 3] = sqrt(e_dst[i + 3]);
            }
            free(p_sub);
            free(p_mul);
            free(tests);
            free(p_acc);
            free(e_dst);
            // for (i = 0; i < test_instances; i += n_instances) {
            //     read_test_instance(te_base, VSIZE);
            //     for (j = 0; j < training_instances; j++) {
            //         _vim2K_fsubs(&tr_base[j * VSIZE], te_base, partial_sub);
            //         _vim2K_fmuls(partial_sub, partial_sub, partial_mul);
            //         for (k = 0; k < n_instances; k++) {
            //             _vim2K_fmuls(partial_mul, mask[k], partial_acc);
            //             _vim2K_fcums(partial_acc, &e_distance[i + k][j]);
            //         }
            //     }
            // }
        } else {
            for (i = 0; i < test_instances; i++) {
                read_test_instance(te_base, n_vectors * VSIZE);
                for (j = 0; j < training_instances; j++) {
                    for (k = 0; k < n_vectors; k++) {
                        _vim2K_fsubs(&tr_base[(j * VSIZE * n_vectors) + (k * VSIZE)], &te_base[k * VSIZE], partial_sub);
                        _vim2K_fmuls(partial_sub, partial_sub, partial_mul);
                        _vim2K_fcums(partial_mul, &partial_sum);
                        e_distance[i][j] += partial_sum;
                    }
                }
            }
        }
    }

    for (i = 0; i < test_instances; ++i) {
        for (j = 0; j < training_instances; ++j) {
            e_distance[i][j] = sqrt(e_distance[i][j]);
        }
        class_begin = clock();
        get_ksmallest(e_distance[i], tr_label, knn, k_neighbors);
        votes(knn, k_neighbors);
        class_end = clock();
        class_spent += (double)(class_end - class_begin) / CLOCKS_PER_SEC;
    }
    ed_end = clock();
    ed_spent = (double)(ed_end - ed_begin)/CLOCKS_PER_SEC;

    free(knn);
    for (i = 0; i < n_instances; i++) {
        free(e_distance[i]);
    }
    free(e_distance);
    if (training_features < VSIZE) {
        for (i = 0; i < n_instances; i++) {
            free(mask[i]);
        }
        free(mask);
    }
    free(partial_sub);
    free(partial_mul);
    free(partial_acc);
    free(te_base);
}

int main(int argc, char const *argv[]) {
    total_begin = clock();

    // Initialize train and test matrix
    read_files(argv);

    // Calculates Euclidean Distance
    classification(argv);

    total_end = clock();
    total_spent = (double)(total_end - total_begin) / CLOCKS_PER_SEC;
    printf("**************************************\n");
    printf("* Execution time:          %fs *\n", total_spent);
    printf(" ************************************\n");
    printf("* Read time:               %fs *\n", read_spent);
    printf("* Euclidean Distance time: %fs *\n", ed_spent);
    printf("* Classification time:     %fs *\n", class_spent);
    printf("**************************************\n");
    free(tr_base);
    free(tr_label);
    return 0;
}
