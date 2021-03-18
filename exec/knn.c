#include "knn.h"

void read_files(char const *argv[]) {
    vector_size = atoi(argv[1]);
    if (vector_size == 256) {
        VSIZE = VM64I;
    } else {
        VSIZE = VM2KI;
    }

    training_instances = atoi(argv[2]);
    test_instances = atoi(argv[3]);
    training_features = atoi(argv[4]);
    k_neighbors = atoi(argv[5]);

    n_vectors = training_features/VSIZE;
    n_instances = 1;
    if (training_features < VSIZE) {
        n_vectors = 1;
        n_instances = VSIZE/training_features;
    }

    tr_base_size = (training_instances * training_features) + VSIZE;
    tr_base = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * tr_base_size);
    tr_label = (__v32u *)aligned_alloc(vector_size, sizeof(__v32u) * training_instances);
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
    if (pos > neg) {
        printf("%s\n", "pos");
    } else {
        printf("%s\n", "neg");
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

// sqrt(pow((x1 - y1), 2) + pow((x2 - y2), 2) + ... + pow((xn - yn), 2))
void classification(char const *argv[]) {
    int32_t i, j, k, ed_idx = 0;

    __v32f *te_base = (__v32f *)aligned_alloc(vector_size, (n_vectors * VSIZE * sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));
    __v32f *e_distance = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * test_instances * training_instances);
    __v32u *knn = (__v32u *)aligned_alloc(vector_size, k_neighbors * sizeof(__v32u));
    __v32f *temp_test = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * VSIZE);
    __v32f *temp_train = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * VSIZE);
    __v32f *partial_sum = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * VSIZE);

    if (vector_size == 256) {
        for (int i = 0; i < training_instances * training_features; i += VSIZE) {
            _vim64_fmovs(1.0, &tr_base[i]);
        }
        if (training_features < VSIZE) {
            __v32f *mask = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * VSIZE);
            for (i = 0; i < training_features; ++i) {
                mask[i] = 1.0;
            }
            for (i = 0; i < test_instances; i += n_instances) {
                _vim64_fmovs(0.5, te_base);
                for (j = 0; j < training_instances; ++j) {
                    _vim64_fmuls(&tr_base[j * training_features], mask, temp_train);
                    for (k = 0; k < n_instances; ++k) {
                        _vim64_fmuls(&te_base[k * training_features], mask, temp_test);
                        _vim64_fsubs(temp_train, temp_test, temp_test);
                        _vim64_fmuls(temp_test, temp_test, temp_test);
                        _vim64_fcums(temp_test, &e_distance[ed_idx++]);
                    }
                }
            }
            free(mask);
        } else {
            for (i = 0; i < test_instances; ++i) {
                for (j = 0; j < n_vectors; ++j) {
                    _vim64_fmovs(0.5, &te_base[j * VSIZE]);
                }
                for (j = 0; j < training_instances; ++j) {
                    for (k = 0; k < n_vectors; ++k) {
                        _vim64_fsubs(&tr_base[(j * VSIZE * n_vectors) + (k * VSIZE)], &te_base[k * VSIZE], temp_test);
                        _vim64_fmuls(temp_test, temp_test, temp_test);
                        _vim64_fcums(temp_test, &partial_sum[k]);
                    }
                    _vim64_fcums(partial_sum, &e_distance[ed_idx++]);
                }
            }
        }
    } else {
        if (training_features < VSIZE) {
            __v32f *mask = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * VSIZE);
            for (i = 0; i < training_features; ++i) {
                mask[i] = 1.0;
            }
            for (int i = 0; i < training_instances * training_features; i += VSIZE) {
                _vim2K_fmovs(1.0, &tr_base[i]);
            }    
            for (i = 0; i < test_instances; i += n_instances) {
                _vim2K_fmovs(0.5, te_base);
                for (j = 0; j < training_instances; ++j) {
                    _vim2K_fmuls(&tr_base[j* training_features], mask, temp_train);
                    for (k = 0; k < n_instances; ++k) {
                        _vim2K_fmuls(&te_base[k * training_features], mask, temp_test);
                        _vim2K_fsubs(temp_train, temp_test, temp_test);
                        _vim2K_fmuls(temp_test, temp_test, temp_test);
                        _vim2K_fcums(temp_test, &e_distance[ed_idx++]);
                    }
                }
            }
            free(mask);
        } else {
            for (i = 0; i < test_instances; ++i) {
                for (j = 0; j < n_vectors; ++j) {
                    _vim2K_fmovs(0.5, &te_base[j * VSIZE]);
                }
                for (j = 0; j < training_instances; ++j) {
                    for (k = 0; k < n_vectors; ++k) {
                        _vim2K_fsubs(&tr_base[(j * VSIZE * n_vectors) + (k * VSIZE)], &te_base[k * VSIZE], temp_test);
                        _vim2K_fmuls(temp_test, temp_test, temp_test);
                        _vim2K_fcums(temp_test, &partial_sum[k]);
                    }
                    _vim2K_fcums(partial_sum, &e_distance[ed_idx++]);
                }
            }
        }
    }
    for (i = 0; i < test_instances * training_instances; ++i) {
        e_distance[i] = sqrt(e_distance[i]);
    }
    
    for (i = 0; i < test_instances * training_instances; i += training_instances) {
        get_ksmallest(&e_distance[i], tr_label, knn, k_neighbors);
        printf("i. %d", i);
        votes(knn, k_neighbors);
    }

    free(partial_sum);
    free(knn);
    free(e_distance);
    free(te_base);
}

int main(int argc, char const *argv[]) {
    // Initialize train and test matrix
    read_files(argv);

    // Calculates Euclidean Distance
    classification(argv);

    free(tr_base);
    free(tr_label);
    return 0;
}
