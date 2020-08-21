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
    printf("tr_base 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)tr_base>>8) &31));
    printf("tr_base vector_base: %20llu - ptr:%p\n", (void*)tr_base, &tr_base);
    printf("tr_base position_0: %20llu - ptr:%p\n", (void*)&tr_base[0], &tr_base);
    printf("tr_base position_1: %20llu - ptr:%p\n", (void*)&tr_base[1], &tr_base);
    printf("tr_base position_f: %20llu - ptr:%p\n\n", (void*)&tr_base[tr_base_size], &tr_base);

    tr_label = (__v32u *)aligned_alloc(vector_size, sizeof(__v32u)*training_instances);
    printf("tr_label 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)tr_label>>8) &31));
    printf("tr_label vector_base: %20llu - ptr:%p\n", (void*)tr_label, &tr_label);
    printf("tr_label position_0: %20llu - ptr:%p\n", (void*)&tr_label[0], &tr_label);
    printf("tr_label position_1: %20llu - ptr:%p\n", (void*)&tr_label[1], &tr_label);
    printf("tr_label position_f: %20llu - ptr:%p\n\n", (void*)&tr_label[training_instances], &tr_label); 

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
        for (int i = 0; i < size; i += VSIZE) {
            _vim2K_fmovs(1, &base[i]);
        }
    }
}

inline __v32f **initialize_mask(int stride, int n_masks) {
    __v32f **mask = (__v32f **)malloc(n_masks * sizeof(__v32f *));
    for (int i = 0; i < n_masks; i++) {
        mask[i] = (__v32f *)malloc(VSIZE * sizeof(__v32f));
    }
    printf("mask 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)mask>>8) &31));
    printf("mask vector_base: %20llu - ptr:%p\n", (void*)mask, &mask);
    printf("mask position_0: %20llu - ptr:%p\n", (void*)&mask[0], &mask);
    printf("mask position_1: %20llu - ptr:%p\n", (void*)&mask[1], &mask);
    printf("mask position_f: %20llu - ptr:%p\n\n", (void*)&mask[n_masks], &mask); 
    for (int i = 0; i < n_masks; i++) {
        for (int j = i * stride; j < (i * stride) + stride; j++) {
            mask[i][j] = 1.0;
        }
    }
    return mask;
}

// sqrt(pow((x1 - y1), 2) + pow((x2 - y2), 2) + ... + pow((xn - yn), 2))
void classification(char const *argv[]) {
    int i, j, jj, k;
    __v32f partial_sum;
    __v32f *te_base;

    k_neighbors = atoi(argv[5]);

    int n_instances = 1;
    if (training_features < VSIZE) {
        n_instances = VSIZE/training_features;
    }

    te_base = (__v32f *)aligned_alloc(vector_size, n_vectors * VSIZE * sizeof(__v32f));
    printf("te_base 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)te_base>>8) &31));
    printf("te_base vector_base: %20llu - ptr:%p\n", (void*)te_base, &te_base);
    printf("te_base position_0: %20llu - ptr:%p\n", (void*)&te_base[0], &te_base);
    printf("te_base position_1: %20llu - ptr:%p\n", (void*)&te_base[1], &te_base);
    printf("te_base position_f: %20llu - ptr:%p\n\n", (void*)&te_base[n_vectors * VSIZE], &te_base); 

    __v32f **e_distance = (__v32f **)aligned_alloc(vector_size, test_instances * sizeof(__v32f *));
    for (i = 0; i < test_instances; ++i) {
        e_distance[i] = (__v32f *)aligned_alloc(vector_size, training_instances * sizeof(__v32f));
    }
    printf("e_distance 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)e_distance>>8) &31));
    printf("e_distance vector_base: %20llu - ptr:%p\n", (void*)e_distance, &e_distance);
    printf("e_distance position_0: %20llu - ptr:%p\n", (void*)&e_distance[0], &e_distance);
    printf("e_distance position_1: %20llu - ptr:%p\n", (void*)&e_distance[1], &e_distance);
    printf("e_distance position_f: %20llu - ptr:%p\n\n", (void*)&e_distance[test_instances], &e_distance); 

    __v32u *knn = (__v32u *)malloc(k_neighbors * sizeof(__v32u));
    // printf("knn 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)knn>>8) &31));
    // printf("knn vector_base: %20llu - ptr:%p\n", (void*)knn, &knn);
    // printf("knn position_0: %20llu - ptr:%p\n", (void*)&knn[0], &knn);
    // printf("knn position_1: %20llu - ptr:%p\n", (void*)&knn[1], &knn);
    // printf("knn position_f: %20llu - ptr:%p\n\n", (void*)&knn[k_neighbors], &knn); 

    __v32f *partial_sub = (__v32f *)malloc(sizeof(__v32f) * n_vectors * VSIZE);
    printf("partial_sub 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)partial_sub>>8) &31));
    printf("partial_sub vector_base: %20llu - ptr:%p\n", (void*)partial_sub, &partial_sub);
    printf("partial_sub position_0: %20llu - ptr:%p\n", (void*)&partial_sub[0], &partial_sub);
    printf("partial_sub position_1: %20llu - ptr:%p\n", (void*)&partial_sub[1], &partial_sub);
    printf("partial_sub position_f: %20llu - ptr:%p\n\n", (void*)&partial_sub[n_vectors * VSIZE], &partial_sub); 

    __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * n_vectors * VSIZE);
    printf("partial_mul 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)partial_mul>>8) &31));
    printf("partial_mul vector_base: %20llu - ptr:%p\n", (void*)partial_mul, &partial_mul);
    printf("partial_mul position_0: %20llu - ptr:%p\n", (void*)&partial_mul[0], &partial_mul);
    printf("partial_mul position_1: %20llu - ptr:%p\n", (void*)&partial_mul[1], &partial_mul);
    printf("partial_mul position_f: %20llu - ptr:%p\n\n", (void*)&partial_mul[n_vectors * VSIZE], &partial_mul); 

    __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    printf("partial_acc 1024-byte aligned addr: vault %20llu - ptr:%p\n", (((long int)partial_acc>>8) &31));
    printf("partial_acc vector_base: %20llu - ptr:%p\n", (void*)partial_acc, &partial_acc);
    printf("partial_acc position_0: %20llu - ptr:%p\n", (void*)&partial_acc[0], &partial_acc);
    printf("partial_acc position_1: %20llu - ptr:%p\n", (void*)&partial_acc[1], &partial_acc);
    printf("partial_acc position_f: %20llu - ptr:%p\n\n", (void*)&partial_acc[VSIZE], &partial_acc); 

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
        if (training_features < VSIZE) {
            for (i = 0; i < test_instances; i += n_instances) {
                read_test_instance(te_base, VSIZE);
                for (j = 0; j < training_instances; j++) {
                    _vim2K_fsubs(&tr_base[j * VSIZE], te_base, partial_sub);
                    _vim2K_fmuls(partial_sub, partial_sub, partial_mul);
                    for (k = 0; k < n_instances; k++) {
                        _vim2K_fmuls(partial_mul, mask[k], partial_acc);
                        _vim2K_fcums(partial_acc, &e_distance[i + k][j]);
                    }
                }
            }
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
