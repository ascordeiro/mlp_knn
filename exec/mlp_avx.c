#include "mlp.h"


void read_file(char const *argv[], base_type *train_base) {
    vector_size = atoi(argv[1]);
    if (vector_size == 256) {
        VSIZE = VM64I;
    } else {
        VSIZE = VM2KI;
    }

    training_instances = atoi(argv[2]);
    training_features = atoi(argv[3]);

    n_vectors = training_features/VSIZE;
    n_instances = 1;
    if (training_features < VSIZE) {
        n_vectors = 1;
        n_instances =  VSIZE/training_features;
    }

    if (training_features * (training_features/2) < VSIZE) {
        base_size = (training_instances/(VSIZE/(training_features * (training_features/2)))) * VSIZE;
    } else {
        base_size = training_instances * n_vectors * VSIZE;
    }

    train_base->base = (__v32f *)aligned_alloc(32, sizeof(__v32f)*base_size);
    train_base->label = (__v32u *)aligned_alloc(32, sizeof(__v32u)*training_instances);
    // train_base->base = (__v32f *)calloc(base_size, sizeof(__v32f));
    // train_base->label = (__v32u *)calloc(training_instances, sizeof(__v32u));
}

__v32f *relu_layer(base_type *train_base) {
    int i, j, k, h_idx = 0;
    int fw_size = (training_features / 2) / n_instances;
    if (training_features * (training_features / 2) < VSIZE) {
        fw_size = 1;
    }
    hidden_size = training_features/2 * training_instances;
    if (hidden_size < VSIZE) {
        hidden_size = VSIZE;
    }
    __v32f *f_weights = (__v32f *)aligned_alloc(32, fw_size * VSIZE * sizeof(__v32f));
    __v32f *partial_mul = (__v32f *)aligned_alloc(32, sizeof(__v32f) * fw_size * n_vectors * VSIZE);
    __v32f *partial_sum = (__v32f *)aligned_alloc(32, sizeof(__v32f) * fw_size * n_vectors * VSIZE);
    __v32f *partial_acc = (__v32f *)aligned_alloc(32, sizeof(__v32f) * VSIZE);
    __v32f *hidden_layer = (__v32f *)aligned_alloc(32, hidden_size * sizeof(__v32f));
    // __v32f *f_weights = (__v32f *)calloc(fw_size * VSIZE, sizeof(__v32f));
    // __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * fw_size * n_vectors * VSIZE);
    // __v32f *partial_sum = (__v32f *)malloc(sizeof(__v32f) * fw_size * n_vectors * VSIZE);
    // __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    // __v32f *hidden_layer = (__v32f *)calloc(hidden_size, sizeof(__v32f));
    __v32f sum, p_sum;

    int m_size;
    if ((vector_size == 256 && training_features >= 16) || (vector_size == 8192 && training_features >= 64)) {
        m_size = VSIZE/training_features;
    } 
    else {
        m_size = (VSIZE/(training_features * (training_features/2))) * training_features/2;
    }
    // __v32f **mask = (__v32f **)calloc(m_size, sizeof(__v32f *));
    // for (i = 0; i < m_size; ++i) {
    //     mask[i] = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    // }
    __v32f **mask = (__v32f **)aligned_alloc(32, m_size * sizeof(__v32f *));
    for (i = 0; i < m_size; ++i) {
        mask[i] = (__v32f *)aligned_alloc(32, VSIZE * sizeof(__v32f));
    }
    
    bias = (__v32f *)aligned_alloc(32, VSIZE * sizeof(__v32f));
    // bias = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    __m256 avx_base, avx_weights, avx_bias, avx_pmul, avx_psum;
    // avx_bias = _mm256_broadcastss_ps(1.0);
    if (vector_size == 256) {
        for (i = 0; i < base_size; i += VSIZE * n_vectors) {
            for (j = 0; j < fw_size; ++j) {
                for (k = 0; k < n_vectors; ++k) {
                    _vim64_fmuls(&train_base->base[i + k * VSIZE], &f_weights[j * VSIZE], &partial_mul[j * VSIZE * n_vectors + k * VSIZE]);
                    _vim64_fadds(bias, &partial_mul[j * VSIZE * n_vectors + k * VSIZE], &partial_sum[j * VSIZE * n_vectors + k * VSIZE]);
                }
            }
            if (training_features >= VSIZE) {
                for (j = 0; j < fw_size; ++j) {
                    sum = 0.0;
                    for (k = 0; k < n_vectors; ++k) {
                        _vim64_fcums(&partial_sum[j * VSIZE * n_vectors + k * VSIZE], &p_sum);
                        sum += p_sum;
                    }
                    if (sum > 0.0) {
                        hidden_layer[h_idx] = sum;
                    }
                    ++h_idx;
                }
            } else {
                for (j = 0; j < fw_size; ++j) {
                    for (k = 0; k < m_size; ++k) {
                        _vim64_fmuls(mask[k], &partial_sum[j * VSIZE], partial_acc);
                        _vim64_fcums(partial_acc, &sum);
                        if (sum > 0.0) {
                            hidden_layer[h_idx] = sum;
                        }
                        ++h_idx;
                    }
                }
            }
        }
    } else {
        for (i = 0; i < base_size; i += VSIZE * n_vectors) {
            for (j = 0; j < fw_size; ++j) {
                for (k = 0; k < n_vectors; ++k) {
                    _vim2K_fmuls(&train_base->base[i + k * VSIZE], &f_weights[j * VSIZE], &partial_mul[j * VSIZE * n_vectors + k * VSIZE]);
                    _vim2K_fadds(bias, &partial_mul[j * VSIZE * n_vectors + k * VSIZE], &partial_sum[j * VSIZE * n_vectors + k * VSIZE]);
                }
            }
            if (training_features >= VSIZE) {
                for (j = 0; j < fw_size; ++j) {
                    sum = 0.0;
                    for (k = 0; k < n_vectors; ++k) {
                        _vim2K_fcums(&partial_sum[j * VSIZE + k * VSIZE], &p_sum);
                        sum += p_sum;
                    }
                    if (sum > 0.0) {
                        hidden_layer[h_idx] = sum;
                    }
                    ++h_idx;
                }
            } else {
                for (j = 0; j < fw_size; ++j) {
                    for (k = 0; k < m_size; ++k) {
                        _vim2K_fmuls(mask[k], &partial_sum[j * VSIZE], partial_acc);
                        _vim2K_fcums(partial_acc, &sum);
                        if (sum > 0.0) {
                            hidden_layer[h_idx] = sum;
                        }
                        ++h_idx;
                    }
                }
            }
        }
    }
    for (i = 0; i < m_size; i++) {
        free(mask[i]);
    }
    free(mask);
    free(partial_mul);
    free(partial_sum);
    free(partial_acc);
    free(f_weights);
    return hidden_layer;
}

__v32f *softmax_layer(__v32f *hidden_layer) {
    int i, j, k, o_idx = 0;
    __v32f sum, p_sum;

    int out_inst_size = (training_features/2)/VSIZE;
    if (training_features/2 < VSIZE) {
        out_inst_size = 1;
    }

    int olayer_size = output_size * training_instances;
    if (olayer_size < VSIZE) {
        olayer_size = VSIZE;
    }

    __v32f *output_layer = (__v32f *)aligned_alloc(32, olayer_size * sizeof(__v32f));
    __v32f *f_output = (__v32f *)aligned_alloc(32, VSIZE * output_size * sizeof(__v32f));
    __v32f *partial_mul = (__v32f *)aligned_alloc(32, sizeof(__v32f) * VSIZE * out_inst_size * output_size);
    __v32f *partial_sum = (__v32f *)aligned_alloc(32, sizeof(__v32f) * VSIZE * out_inst_size * output_size);
    __v32f *partial_acc = (__v32f *)aligned_alloc(32, sizeof(__v32f) * VSIZE);
    // __v32f *output_layer = (__v32f *)calloc(olayer_size, sizeof(__v32f));
    // __v32f *f_output = (__v32f *)calloc(VSIZE * output_size, sizeof(__v32f));
    // __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * VSIZE * out_inst_size * output_size);
    // __v32f *partial_sum = (__v32f *)malloc(sizeof(__v32f) * VSIZE * out_inst_size * output_size);
    // __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);

    __v32f **mask;
    if (VSIZE/(training_features/2) > 1) {
        mask = (__v32f **)aligned_alloc(32, VSIZE/(training_features/2) * sizeof(__v32f *));
        for (int i = 0; i < VSIZE/(training_features/2); ++i) {
            mask[i] = (__v32f *)aligned_alloc(32, VSIZE * sizeof(__v32f));
        }
        // mask = (__v32f **)calloc(VSIZE/(training_features/2), sizeof(__v32f *));
        // for (int i = 0; i < VSIZE/(training_features/2); ++i) {
        //     mask[i] = (__v32f *)calloc(VSIZE, sizeof(__v32f));
        // }
    }

    if (vector_size == 256) {
        for (i = 0; i < hidden_size; i += VSIZE * out_inst_size) {
            for (j = 0; j < output_size * VSIZE; j += VSIZE) {
                for (k = 0; k < out_inst_size * VSIZE; k += VSIZE) {
                    _vim64_fmuls(&hidden_layer[i + k], &f_output[j], &partial_mul[(j * out_inst_size) + k]);
                    _vim64_fadds(bias, &partial_mul[(j * out_inst_size) + k], &partial_sum[(j * out_inst_size) + k]);
                }
            }
            if (training_features/2 >= VSIZE) {
                for (j = 0; j < VSIZE * output_size; j += VSIZE) {
                    sum = 0.0;
                    for (k = 0; k < out_inst_size; ++k) {
                        _vim64_fcums(&partial_sum[(j * out_inst_size) + k], &p_sum);
                        sum += p_sum;
                    }
                    if (sum > 0.0) {
                        output_layer[o_idx] = sum;
                    }
                    ++o_idx;
                }
            } else {
                for (j = 0; j < out_inst_size * output_size; ++j) {
                    for (k = 0; k < VSIZE/(training_features/2); ++k) {
                        _vim64_fmuls(mask[k], &partial_sum[j * VSIZE], partial_acc);
                        _vim64_fcums(partial_acc, &sum);
                        if (sum > 0.0) {
                            output_layer[o_idx] = sum;
                        }
                        ++o_idx;
                    }
                }
            }
        }
    } else {
        for (i = 0; i < hidden_size; i += VSIZE * out_inst_size) {
            for (j = 0; j < output_size * VSIZE; j += VSIZE) {
                for (k = 0; k < out_inst_size * VSIZE; k += VSIZE) {
                    _vim2K_fmuls(&hidden_layer[i + k], &f_output[j], &partial_mul[(j * out_inst_size) + k]);
                    _vim2K_fadds(bias, &partial_mul[(j * out_inst_size) + k], &partial_sum[(j * out_inst_size) + k]);
                }
            }
            if (training_features/2 >= VSIZE) {
                for (j = 0; j < VSIZE * output_size; j += VSIZE) {
                    sum = 0.0;
                    for (k = 0; k < out_inst_size; ++k) {
                        _vim2K_fcums(&partial_sum[(j * out_inst_size) + k], &p_sum);
                        sum += p_sum;
                    }
                    if (sum > 0.0) {
                        output_layer[o_idx] = sum;
                    }
                    ++o_idx;
                }
            } else {
                for (k = 0; k < VSIZE/(training_features/2); ++k) {
                    for (j = 0; j < out_inst_size * output_size; ++j) {
                        _vim2K_fmuls(mask[k], &partial_sum[j * VSIZE], partial_acc);
                        _vim2K_fcums(partial_acc, &sum);
                        if (sum > 0.0) {
                            output_layer[o_idx] = sum;
                        }
                        ++o_idx;
                    }
                    if (o_idx == 512) break;
                }
            }
        }
    }

    if (VSIZE/(training_features/2) > 1) {
        for (i = 0; i < (VSIZE/(training_features/2)); ++i) {
            free(mask[i]);
        }
        free(mask);
    }
    free(f_output);
    free(partial_sum);
    free(partial_mul);
    free(partial_acc);
    return output_layer;
}

void classification(__v32f *output_layer) {
    __v32f sum_exp;
    __v32f *result = (__v32f *)aligned_alloc(32, sizeof(__v32f) * output_size);
    // __v32f *result = (__v32f *)malloc(sizeof(__v32f) * output_size);
    for (int i = 0; i < training_instances; ++i) {
        sum_exp = 0.0;
        for (int j = 0; j < output_size; ++j) {
            sum_exp += exp(output_layer[i * output_size + j]);
        }
        for (int j = 0; j < output_size; ++j) {
            result[j] = exp(output_layer[i * output_size + j]) / sum_exp;
        }
        if (result[0] > result[1]) {
            printf("%d. %s\n", i, "neg");
        } else {
            printf("%d. %s\n", i, "pos");
        }
    }
    free(result);
}

__v32f *normalization(__v32f *layer, int size) {
    // __v32f *broad_minmax = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    // __v32f *broad_min = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    // __v32f *partial_sub = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    __v32f *broad_minmax = (__v32f *)aligned_alloc(32, sizeof(__v32f) * VSIZE);
    __v32f *broad_min = (__v32f *)aligned_alloc(32, sizeof(__v32f) * VSIZE);
    __v32f *partial_sub = (__v32f *)aligned_alloc(32, sizeof(__v32f) * VSIZE);
    if (size < VSIZE) {
        size = VSIZE;
    }
    __v32f *norm = (__v32f *)aligned_alloc(32, sizeof(__v32f) * size);
    // __v32f *norm = (__v32f *)malloc(sizeof(__v32f) * size);

    __v32f min = layer[0];
    __v32f max = layer[0];
    for (int i = 1; i < size; ++i) {
        if (layer[i] < min) {
            min = layer[i];
        }
        if (layer[i] > max) {
            max = layer[i];
        }
    }
    __v32f max_min = max - min;

    if (vector_size == 256) {
        _vim64_fmovs(max_min, broad_minmax);
        _vim64_fmovs(min, broad_min);

        for (int i = 0; i < size; i += VSIZE) {
            _vim64_fsubs(&layer[i], broad_min, partial_sub);
            _vim64_fdivs(partial_sub, broad_minmax, &norm[i]);
        }
    } else {
        _vim2K_fmovs(max_min, broad_minmax);
        _vim2K_fmovs(min, broad_min);

        for (int i = 0; i < size; i += VSIZE) {
            _vim2K_fsubs(&layer[i], broad_min, partial_sub);
            _vim2K_fdivs(partial_sub, broad_minmax, &norm[i]);
        }
    }
    free(broad_minmax);
    free(broad_min);
    free(partial_sub);
    return norm;
}

int main(int argc, char const *argv[]) {
    total_begin = clock();
    base_type train_base;

    read_begin = clock();
    read_file(argv, &train_base);
    output_size = atoi(argv[4]);
    read_end = clock();
    read_spent = (double)(read_end - read_begin) / CLOCKS_PER_SEC;

    hidden_begin = clock();
    __v32f *hidden_layer = relu_layer(&train_base);
    hidden_end = clock();
    hidden_spent = (double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC;

    norm_begin1 = clock();
    __v32f *hidden_norm = normalization(hidden_layer, hidden_size);
    norm_end1 = clock();
    norm_spent1 = (double)(norm_end1 - norm_begin1) / CLOCKS_PER_SEC;

    output_begin = clock();
    __v32f *output_layer = softmax_layer(hidden_norm);
    output_end = clock();
    output_spent = (double)(output_end - output_begin) / CLOCKS_PER_SEC;

    norm_begin2 = clock();
    __v32f *output_norm = normalization(output_layer, output_size*training_instances);
    norm_end2 = clock();
    norm_spent2 = (double)(norm_end2 - norm_begin2) / CLOCKS_PER_SEC;

    class_begin = clock();
    classification(output_norm);
    class_end = clock();
    class_spent = (double)(class_end - class_begin) / CLOCKS_PER_SEC;

    total_end = clock();
    total_spent = (double)(total_end - total_begin) / CLOCKS_PER_SEC;
    printf("*************************************\n");
    printf("* Execution time:         %fs *\n", total_spent);
    printf(" ***********************************\n");
    printf("* Read time:              %fs *\n", read_spent);
    printf("* Input x Hidden layer:   %fs *\n", hidden_spent);
    printf("* Hidden x Output layer:  %fs *\n", output_spent);
    printf("* Normalization:          %fs *\n", norm_spent1 + norm_spent2);
    printf("* Classification time:    %fs *\n", class_spent);
    printf("*************************************\n");
    free(train_base.base);
    free(train_base.label);
    free(hidden_norm);
    free(hidden_layer);
    free(output_layer);
    free(output_norm);
    free(bias);
    return 0;
}
