#include "mlp.h"


void read_file(char const *argv[]) {
    vector_size = atoi(argv[1]);
    if (vector_size == 256) {
        VSIZE = VM64I;
    } else {
        VSIZE = VM2KI;
    }

    training_instances = atoi(argv[2]);
    training_features = atoi(argv[3]);

    n_vectors = training_features/VSIZE;

    inst_in_vec = VSIZE/(training_features*(training_features/2));
    if (inst_in_vec > 1) {
        base_size = training_instances/inst_in_vec * VSIZE;
    } else if (VSIZE/training_features > 1) {
        base_size = training_instances * VSIZE; 
    } else {
        base_size = training_instances * n_vectors * VSIZE;
    }

    base = (__v32f *)calloc(base_size, sizeof(__v32f));
    label = (__v32u *)calloc(training_instances, sizeof(__v32u));

    if (vector_size == 256) {
        for (int i = 0; i < base_size; i += VSIZE) {
            _vim64_fmovs(1, &base[i]);
        }
    } else {
        for (int i = 0; i < base_size; i += VSIZE) {
            _vim2K_fmovs(1, &base[i]);
        }
    }

}

inline void *initialize_weights(__v32f *weights, int size) {
    if (vector_size == 256) {
        for (int i = 0; i < size; i += VSIZE) {
            _vim64_fmovs(0.5, &weights[i]);
        }
    } else {
        for (int i = 0; i < size; i += VSIZE) {
            _vim2K_fmovs(0.5, &weights[i]);
        }
    }
}

inline __v32f **initialize_mask(int stride, int n_masks) {
    __v32f **mask = (__v32f **)calloc(n_masks, sizeof(__v32f *));
    for (int i = 0; i < n_masks; i++) {
        mask[i] = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    }
    for (int i = 0; i < n_masks; i++) {
        for (int j = i * stride; j < (i * stride) + stride; j++) {
            mask[i][j] = 1.0;
        }
    }
    return mask;
}

__v32f *relu_layer() {
    int i, j, k, h_idx = 0;

    int weight_size, mask_size;
     __v32f **mask;

    if (inst_in_vec > 1) {
        weight_size = VSIZE;
        mask_size = VSIZE/training_features;
        mask = initialize_mask(training_features, mask_size);
    } else if (VSIZE/training_features > 1) {
        weight_size = training_features*(training_features/2); 
        mask_size = VSIZE/training_features;
        mask = initialize_mask(mask_size, mask_size);
    } else {
        weight_size = n_vectors * training_features/2 * VSIZE;
    }

    __v32f *weights = (__v32f *)calloc(weight_size, sizeof(__v32f));
    initialize_weights(weights, weight_size);

    hidden_size = training_features/2 * training_instances * output_size;
    __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    __v32f *hidden_layer = (__v32f *)malloc(sizeof(__v32f) * hidden_size);
    __v32f *p_hlayer = (__v32f *)calloc(hidden_size, sizeof(__v32f));
    __v32f sum, p_sum;

    bias = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    if (vector_size == 256) {
        _vim64_fmovs(1, &bias[i]);
        if (inst_in_vec > 1) {
            for (i = 0; i < base_size; i += VSIZE) {
                _vim64_fmuls(&base[i], weights, partial_mul);
                for (j = 0; j < mask_size; j++) {
                    _vim64_fmuls(partial_mul, mask[j], partial_acc);
                    _vim64_fcums(partial_acc, &p_hlayer[h_idx]);
                    p_hlayer[h_idx + training_features/2] = p_hlayer[h_idx];
                    h_idx++;
                    if (j == (mask_size/2) -1 && j == mask_size - 1) {
                        h_idx += training_features/2;
                    }      
                }
                initialize_weights(weights, weight_size);
            }
        } else if (VSIZE/training_features > 1) {
            for (i = 0; i < training_instances; i++) {
                for (j = 0; j < weight_size; j += VSIZE) {
                    _vim64_fmuls(&base[i * VSIZE], &weights[j], partial_mul);
                    for (k = 0; k < mask_size; k++) {
                        _vim64_fmuls(partial_mul, mask[k], partial_acc);
                        _vim64_fcums(partial_acc, &p_hlayer[h_idx]);   
                        p_hlayer[h_idx + training_features/2] = p_hlayer[h_idx];
                        h_idx++;
                    }
                }
                h_idx += training_features/2;
                initialize_weights(weights, weight_size);
            }
        } else {
            for (i = 0; i < training_instances; i++) {
                for (j = 0; j < weight_size; j += training_features) {
                    sum = 0.0;
                    for (k = 0; k < n_vectors * VSIZE; k += VSIZE) {
                        _vim64_fmuls(&base[(i * VSIZE) + n_vectors], &weights[j + n_vectors], partial_mul);
                        _vim64_fcums(partial_mul, &p_sum);
                        sum += p_sum;
                    }
                    p_hlayer[h_idx] = sum;
                    p_hlayer[h_idx + training_features/2] = sum;
                    h_idx++;
                }
                h_idx += training_features/2;
                initialize_weights(weights, weight_size);
            }
        }
    } else {
        _vim2K_fmovs(1, &bias[i]);
        if (inst_in_vec > 1) {
            for (i = 0; i < base_size; i += VSIZE) {
                _vim2K_fmuls(&base[i], weights, partial_mul);
                for (j = 0; j < mask_size; j++) {
                    _vim2K_fmuls(partial_mul, mask[j], partial_acc);
                    _vim2K_fcums(partial_acc, &p_hlayer[h_idx]);
                    p_hlayer[h_idx + training_features/2] = p_hlayer[h_idx];
                    h_idx++;
                    if (j == (mask_size/2) -1 && j == mask_size - 1) {
                        h_idx += training_features/2;
                    }
                }
                initialize_weights(weights, weight_size);
            }
        } else if (VSIZE/training_features > 1) {
            for (i = 0; i < training_instances; i++) {
                for (j = 0; j < weight_size; j += VSIZE) {
                    _vim2K_fmuls(&base[i * VSIZE], &weights[j], partial_mul);
                    for (k = 0; k < mask_size; k++) {
                        _vim2K_fmuls(partial_mul, mask[k], partial_acc);
                        _vim2K_fcums(partial_acc, &p_hlayer[h_idx]);
                        p_hlayer[h_idx + training_features/2] = p_hlayer[h_idx];
                        h_idx++;
                    }
                }
                h_idx += training_features/2;
                initialize_weights(weights, weight_size);
            }
        } else {
            for (i = 0; i < training_instances; i++) {
                for (j = 0; j < weight_size; j += training_features) {
                    sum = 0.0;
                    for (k = 0; k < n_vectors * VSIZE; k += VSIZE) {
                        _vim2K_fmuls(&base[(i * VSIZE) + n_vectors], &weights[j + n_vectors], partial_mul);
                        _vim2K_fcums(partial_mul, &p_sum);
                        sum += p_sum;
                    }
                    p_hlayer[h_idx] = sum;
                    p_hlayer[h_idx + training_features/2] = sum;
                    h_idx++;
                }
                h_idx += training_features/2;
                initialize_weights(weights, weight_size);
            }
        }
    }

    if (vector_size == 256) {
        for (i = 0; i < hidden_size; i += VSIZE) {
            _vim64_fadds(&p_hlayer[i], bias, &hidden_layer[i]);
        }
    } else {
        for (i = 0; i < hidden_size; i += VSIZE) {
            _vim2K_fadds(&p_hlayer[i], bias, &hidden_layer[i]);
        }
    }
    if (inst_in_vec > 1 || VSIZE/training_features > 1) {
        for (i = 0; i < (VSIZE/training_features); ++i) {
            free(mask[i]);
        }
        free(mask);    
    }
    free(p_hlayer);
    free(partial_mul);
    free(partial_acc);
    free(weights);
    return hidden_layer;
}

__v32f *softmax_layer(__v32f *hidden_layer) {
    int i, j, k, o_idx = 0;
    __v32f sum, p_sum;

    __v32f *weights = (__v32f *)malloc(sizeof(__v32f) * hidden_size);
    initialize_weights(weights, hidden_size);

    int mask_size = VSIZE/(training_features/2);
    __v32f **mask;
    if (mask_size > 1) {
        mask = initialize_mask(training_features/2, mask_size);
    }

    int o_size = training_instances * output_size;
    if (o_size < VSIZE) {
        o_size = VSIZE;
    }
    __v32f *output_layer = (__v32f *)malloc(sizeof(__v32f) * training_instances * o_size);
    __v32f *p_olayer = (__v32f *)calloc(training_instances * o_size, sizeof(__v32f));
    __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);

    int o_vectors = (training_features/2)/VSIZE;
    if (training_features/2 < VSIZE) {
        o_vectors = 1;
    }

    if(vector_size == 256) {
        if (mask_size > 1) {
            for (i = 0; i < hidden_size; i += VSIZE) {
                _vim64_fmuls(&hidden_layer[i], &weights[i], partial_mul);
                for (j = 0; j < mask_size; j++) {
                    _vim64_fmuls(partial_mul, mask[j], partial_acc);
                    _vim64_fcums(partial_acc, &p_olayer[o_idx++]);
                }
            }
        } else {
            for (i = 0; i < hidden_size; i += VSIZE * o_vectors) {
                sum = 0.0;
                for (j = 0; j < o_vectors; j++) {
                    _vim64_fmuls(&hidden_layer[i + (j * VSIZE)], &weights[i + (j * VSIZE)], partial_mul);
                    _vim64_fcums(partial_mul, &p_sum);
                    sum += p_sum;
                }
                p_olayer[o_idx++] = sum;
            }
        }
    } else {
        if (mask_size > 1) {
            for (i = 0; i < hidden_size; i += VSIZE) {
                _vim2K_fmuls(&hidden_layer[i], &weights[i], partial_mul);
                for (j = 0; j < mask_size; j++) {
                    _vim2K_fmuls(partial_mul, mask[j], partial_acc);
                    _vim2K_fcums(partial_acc, &p_olayer[o_idx++]);
                }
            }
        } else {
            for (i = 0; i < hidden_size; i += VSIZE * o_vectors) {
                sum = 0.0;
                for (j = 0; j < o_vectors; j++) {
                    _vim2K_fmuls(&hidden_layer[i + (j * VSIZE)], &weights[i + (j * VSIZE)], partial_mul);
                    _vim2K_fcums(partial_mul, &p_sum);
                    sum += p_sum;
                }
                p_olayer[o_idx++] = sum;
            }
        }
    }

    if (vector_size == 256) {
        for (i = 0; i < o_size; i += VSIZE) {
            _vim64_fadds(&p_olayer[i], bias, &output_layer[i]);
        }
    } else {
        for (i = 0; i < o_size; i += VSIZE) {
            _vim2K_fadds(&p_olayer[i], bias, &output_layer[i]);
        }

    }

    if (VSIZE/(training_features/2) > 1) {
        for (i = 0; i < (VSIZE/(training_features/2)); ++i) {
            free(mask[i]);
        }
        free(mask);
    }
    free(p_olayer);
    free(partial_mul);
    free(partial_acc);
    return output_layer;
    return 0;
}

void classification(__v32f *output_layer) {
    __v32f sum_exp;
    __v32f *result = (__v32f *)malloc(sizeof(__v32f) * output_size);
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

int main(int argc, char const *argv[]) {
    total_begin = clock();

    read_begin = clock();
    read_file(argv);
    output_size = atoi(argv[4]);
    read_end = clock();
    read_spent = (double)(read_end - read_begin) / CLOCKS_PER_SEC;

    hidden_begin = clock();
    __v32f *hidden_layer = relu_layer();
    hidden_end = clock();
    hidden_spent = (double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC;

    output_begin = clock();
    __v32f *output_layer = softmax_layer(hidden_layer);
    output_end = clock();
    output_spent = (double)(output_end - output_begin) / CLOCKS_PER_SEC;

    class_begin = clock();
    classification(output_layer);
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
    printf("* Classification time:    %fs *\n", class_spent);
    printf("*************************************\n");
    free(base);
    free(label);
    free(hidden_layer);
    free(output_layer);
    free(bias);
    return 0;
}