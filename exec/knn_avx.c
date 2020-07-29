#include "knn_avx.h"

void read_files(char const *argv[]) {
    read_begin = clock();

    training_instances = atoi(argv[1]);
    test_instances = atoi(argv[2]);
    training_features = atoi(argv[3]);

    base_size = training_instances * training_features;

    tr_base = (float *)aligned_alloc(64, sizeof(float) * base_size);
    tr_label = (__uint32_t *)aligned_alloc(64, sizeof(__uint32_t)*training_instances);
    
    __m512 avx_base;
    for(int i = 0; i < base_size; i += AVX_SIZE) {
        avx_base = _mm512_load_ps(&tr_base[i]);
        avx_base = _mm512_set1_ps((float)1.0);
        _mm512_store_ps(&tr_base[i], avx_base);
    }

    read_end = clock();
    read_spent = (double)(read_end - read_begin) / CLOCKS_PER_SEC;
}

void votes(u_int32_t *knn, int k) {
    int i, pos = 0, neg = 0;
    for (i = 0; i < k; ++i) {
        if (knn[i] == 1) {
            pos++;
        } else {
            neg++;
        }
    }
    // if (pos > neg) {
    //     printf("%s\n", "pos");
    // } else {
    //     printf("%s\n", "neg");
    // }
}

void get_ksmallest(float *array, u_int32_t *label, u_int32_t *knn, int k) {
    int idx = 0;
    for (int i = 0; i < k; ++i) {
        knn[i] = 9999999;
        float min = array[0];
        for (int j = 1; j < training_instances; ++j) {
            if (min > array[j]) {
                idx = j;
            }
        }
        knn[i] = label[idx];
        array[idx] = 9999999.0;
    }
}

void read_test_instance(float *base, int size) {
    __m512 avx_base;
    for(int i = 0; i < size; i += AVX_SIZE) {
        avx_base = _mm512_load_ps(&base[i]);
        avx_base = _mm512_set1_ps((float)1.0);
        _mm512_store_ps(&base[i], avx_base);
    }
}

// sqrt(pow((x1 - y1), 2) + pow((x2 - y2), 2) + ... + pow((xn - yn), 2))
void classification(char const *argv[]) {
    int i, j, k, ed_idx = 0;

    k_neighbors = atoi(argv[4]);

    int te_base_size = training_features;
    int masks;
    if (training_features < AVX_SIZE) {
        te_base_size = AVX_SIZE;
        masks = AVX_SIZE/training_features;
    }

    __m512 avx_tebase, avx_trbase, avx_psub, avx_pmul;
    u_int32_t *knn = (u_int32_t *)calloc(k_neighbors, sizeof(u_int32_t));
    float *e_distance = (float *)aligned_alloc(64, training_instances * sizeof (float));
    float *te_base = (float *)aligned_alloc(64, te_base_size * sizeof(float));

    ed_begin = clock();
    if (training_features < AVX_SIZE) {
        __mmask16 avx_mask[2] = {0xff00, 0xff};
        // for each test instance
        for (i = 0; i < test_instances; i++) {
            inst_begin = clock();
            ed_idx = 0;
            read_test_instance(te_base, te_base_size);
            avx_tebase = _mm512_setr_ps(te_base[0], te_base[1], te_base[2], te_base[3], te_base[4], te_base[5], te_base[6], te_base[7],
                                        te_base[0], te_base[1], te_base[2], te_base[3], te_base[4], te_base[5], te_base[6], te_base[7]);

            for (j = 0; j < base_size; j += AVX_SIZE) {
                avx_trbase = _mm512_load_ps(&tr_base[j]);
                avx_psub = _mm512_sub_ps(avx_trbase, avx_tebase);
                avx_pmul = _mm512_mul_ps(avx_psub, avx_psub);
                for (k = 0; k < masks; k += AVX_SIZE) {
                    e_distance[ed_idx++] = _mm512_mask_reduce_add_ps(avx_mask[k], avx_pmul);
                }
            }
            class_begin = clock();
            get_ksmallest(e_distance, tr_label, knn, k_neighbors);
            printf("%d. ", i);
            votes(knn, k_neighbors);
            class_end = clock();
            class_spent += (double)(class_end - class_begin) / CLOCKS_PER_SEC;
            inst_end = clock();
            printf("tempo: %f\n", (double)(inst_end - inst_begin)/CLOCKS_PER_SEC);
        }
    } else {
        int n_vector = training_features/AVX_SIZE;
        for (i = 0; i < test_instances; i++) {
            inst_begin = clock();
            ed_idx = 0;
            read_test_instance(te_base, te_base_size);
            for (j = 0; j < base_size; j += training_features) {
                for (k = 0; k < n_vector; k++) {
                    avx_trbase = _mm512_load_ps(&tr_base[j + k * AVX_SIZE]);
                    avx_tebase = _mm512_load_ps(&te_base[k * AVX_SIZE]);
                    avx_psub = _mm512_sub_ps(avx_trbase, avx_tebase);
                    avx_pmul = _mm512_mul_ps(avx_psub, avx_psub);
                    e_distance[ed_idx] += _mm512_reduce_add_ps(avx_pmul);
                }
                ed_idx++;
            }
            class_begin = clock();
            get_ksmallest(e_distance, tr_label, knn, k_neighbors);
            printf("%d. ", i);
            votes(knn, k_neighbors);
            class_end = clock();
            class_spent += (double)(class_end - class_begin) / CLOCKS_PER_SEC;
            inst_end = clock();
            printf("tempo: %f\n", (double)(inst_end - inst_begin)/CLOCKS_PER_SEC);
        }
    }
    ed_end = clock();
    ed_spent += (double)(ed_end - ed_begin - class_spent) / CLOCKS_PER_SEC;
    
    free(knn);
    free(e_distance);
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
