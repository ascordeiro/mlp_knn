#include "mlp.h"

// inline void initialize_vector_2K(__v32f *vector, int size) {
//     init_begin = clock();
//     for (int i = 0; i < size; i += VSIZE) {
//         _vim2K_fmovs(1.0, &vector[i]);
//     }
//     init_end = clock();
//     init_spent += (double)(init_end - init_begin) / CLOCKS_PER_SEC;
// }

// inline void initialize_vector_64(__v32f *vector, int size) {
//     init_begin = clock();
//     for (int i = 0; i < size; i += VSIZE) {
//         _vim64_fmovs(1.0, &vector[i]);
//     }
//     init_end = clock();
//     init_spent += (double)(init_end - init_begin) / CLOCKS_PER_SEC;
// }

__v32f *relu_layer() {
    int i, j, jj, k, l, h_idx = 0;
    hidden_size = features/2 * instances;
    int weight_size = features * (features/2);
    if (features * (features/2) < VSIZE) {
        weight_size = VSIZE;
    }

    __v32f *mask;
    __v32f *instance_vector = (__v32f *)aligned_alloc(vector_size, (instance_size * sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));
    __v32f *temp_instance = (__v32f *)aligned_alloc(vector_size, (VSIZE * sizeof(__v32f)));
    __v32f *weights = (__v32f *)aligned_alloc(vector_size, (weight_size * sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));
    __v32f *temp_weights = (__v32f *)aligned_alloc(vector_size, (VSIZE * sizeof(__v32f)));
    __v32f *p_sum = (__v32f *)aligned_alloc(vector_size, (VSIZE * sizeof(__v32f)));
    __v32f *hidden_layer = (__v32f *)aligned_alloc(vector_size, (hidden_size *  sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));

    bias = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * VSIZE);
    if (vector_size == 256) {
        for (i = 0; i < weight_size; i += VSIZE) {
            _vim64_fmovs(1.0, weights);
        }
        if (features == 8) {
            __v32f *temp_instance2 = (__v32f *)aligned_alloc(vector_size, (VSIZE * sizeof(__v32f)));
            __v32f *temp_weights2 = (__v32f *)aligned_alloc(vector_size, (VSIZE * sizeof(__v32f)));
            mask = (__v32f *)aligned_alloc(vector_size, VSIZE * sizeof(__v32f));
            for (i = 0; i < features; ++i) {
                mask[i] = 1.0;
            }
            for (i = 0; i < instances; i += instance_size) {
                _vim64_fmovs(1.0, instance_vector);
                // hidden_begin = clock();
                for (j = 0; j < VSIZE; j += features * 2) {
                    _vim64_fmuls(&instance_vector[j], mask, temp_instance);
                    _vim64_fmuls(&instance_vector[j + features], mask, temp_instance2);
                    for (k = 0; k < features/2; ++k) {
                        _vim64_fmuls(&weights[k * features], mask, temp_weights);
                        _vim64_fmuls(temp_instance, temp_weights, temp_weights2);
                        _vim64_fmuls(temp_instance2, temp_weights, temp_weights);
                        _vim64_fcums(temp_weights2, &hidden_layer[h_idx++]);
                        _vim64_fcums(temp_weights, &hidden_layer[h_idx++]);
                    }
                }
                // hidden_end = clock();
                // printf("vector %d: %f\n", i, (double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC);
            }
            free(mask);
            free(temp_instance2);
            free(temp_weights2);
        } else if (features == 16 || features == 32) {
            mask = (__v32f *)aligned_alloc(vector_size, VSIZE * sizeof(__v32f));
            for (i = 0; i < features; ++i) {
                mask[i] = 1.0;
            }
            for (i = 0; i < instances; i += instance_size) {
                // hidden_begin = clock();
                _vim64_fmovs(1.0, instance_vector);
                for (j = 0; j < VSIZE; j += features) {
                    _vim64_fmuls(&instance_vector[j], mask, temp_instance);
                    for (k = 0; k < features/2; ++k) {
                        _vim64_fmuls(&weights[k * features], mask, temp_weights);
                        _vim64_fmuls(temp_instance, temp_weights, temp_weights);
                        _vim64_fcums(temp_weights, &hidden_layer[h_idx++]);
                    }
                }
                // hidden_end = clock();
                // printf("vector %d: %f\n", i, (double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC);
            }
            free(mask);
        } else {
            for (i = 0; i < instances; ++i) {
                // hidden_begin = clock();
                for (j = 0; j < n_vectors; ++j) {
                    _vim64_fmovs(1.0, &instance_vector[j * VSIZE]);
                }
                for (j = 0; j < features/2; ++j) {
                    for (k = 0; k < n_vectors; ++k) {
                        _vim64_fmuls(&instance_vector[k * VSIZE], &weights[(j * features) + (k * VSIZE)], temp_weights);
                        _vim64_fcums(temp_weights, &p_sum[k]);
                    }
                    _vim64_fcums(p_sum, &hidden_layer[h_idx++]);
                }
                // hidden_end = clock();
                // printf("instance %d: %f\n", i, (double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC);
            }
        }
        _vim64_fmovs(1.0, bias);
        for (i = 0; i < hidden_size; i += VSIZE) {
            _vim64_fadds(&hidden_layer[i], bias, &hidden_layer[i]);
        }
    } else {
        for (i = 0; i < weight_size; i += VSIZE) {
            _vim2K_fmovs(1.0, weights);
        }
        mask = (__v32f *)aligned_alloc(vector_size, VSIZE * sizeof(__v32f));
        for (i = 0; i < features; ++i) {
            mask[i] = 1.0;
        }
        for (i = 0; i < instances; i += instance_size) {
            // hidden_begin = clock();
            _vim2K_fmovs(1.0, instance_vector);
            for (j = 0; j < VSIZE; j += features) {
                _vim2K_fmuls(&instance_vector[j], mask, temp_instance);
                for (k = 0; k < features/2; ++k) {
                    _vim2K_fmuls(&weights[(k * features)], mask, temp_weights);
                    _vim2K_fmuls(temp_instance, temp_weights, temp_weights);
                    _vim2K_fcums(temp_weights, &hidden_layer[h_idx++]);
                }
            }
            // hidden_end = clock();
            // printf("vector %d: %f\n", i, (double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC);
        }
        free(mask);

        _vim2K_fmovs(1.0, bias);
        for (i = 0; i < hidden_size; i += VSIZE) {
            _vim2K_fadds(&hidden_layer[i], bias, &hidden_layer[i]);
        }
    }
        
    for (i = 0; i < hidden_size; ++i) {
        if (hidden_layer[i] < 0.0) 
        hidden_layer[i] = 0.0;
    }

    free(p_sum);
    free(instance_vector);
    free(temp_instance);
    free(weights);
    free(temp_weights);
    return hidden_layer;
}

__v32f *softmax_layer(__v32f *hidden_layer) {
    int i, ii, j, k, o_idx = 0;
    int o_size = instances * output_size;
    if (o_size < VSIZE) {
        o_size = VSIZE;
    }

    __v32f *weights = (__v32f *)aligned_alloc(vector_size, (VSIZE * n_vectors/2 * sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));
    __v32f *temp_hidden = (__v32f *)aligned_alloc(vector_size, VSIZE * sizeof(__v32f));
    __v32f *temp_weights = (__v32f *)aligned_alloc(vector_size, VSIZE * sizeof(__v32f));
    __v32f *p_sum = (__v32f *)aligned_alloc(vector_size, VSIZE * sizeof(__v32f));
    __v32f *output_layer = (__v32f *)aligned_alloc(vector_size, o_size * sizeof(__v32f));

    if(vector_size == 256) {
        for (i = 0; i < n_vectors/2; ++i) {
            _vim64_fmovs(1.0, weights);
        }
        if (features < 128) {
        __v32f *mask = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * n_vectors * VSIZE);
        for (i = 0; i < features/2; ++i) {
            mask[i] = 1.0;
        }
            for (i = 0, ii = 0; i < hidden_size; i += features/2, ii += features/2) {
                _vim64_fmuls(&hidden_layer[i], mask, temp_hidden);
                if (i % (VSIZE/2) == 0) {
                    ii = 0;
                }
                for (j = 0; j < output_size; ++j) {
                    _vim64_fmuls(&weights[(ii * 2) + (j * (features/2))], mask, temp_weights);
                    _vim64_fmuls(temp_hidden, temp_weights, temp_weights);
                    _vim64_fcums(temp_weights, &output_layer[o_idx++]);
                }
            }
            free(mask);
        } else {
            for (i = 0; i < hidden_size; i += VSIZE * n_vectors/2) {
                for (j = 0; j < output_size; ++j) {
                    for (k = 0; k < n_vectors/2; ++k) {
                        _vim64_fmuls(&hidden_layer[i + (k * VSIZE)], weights, temp_weights);
                        _vim64_fcums(temp_weights, &p_sum[k]);
                    }
                    _vim64_fcums(p_sum, &output_layer[o_idx]);
                }
            }
        }
        for (i = 0; i < o_size; i += VSIZE) {
            _vim64_fadds(&output_layer[i], bias, &output_layer[i]);
        }
    } else {
        __v32f *mask = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * n_vectors * VSIZE);
        for (i = 0; i < features/2; ++i) {
            mask[i] = 1.0;
        }
        for (i = 0; i < n_vectors/2; ++i) {
            _vim2K_fmovs(1.0, weights);
        } 
        for (i = 0, ii = 0; i < hidden_size; i += features/2, ii += features/2) {
            _vim2K_fmuls(&hidden_layer[i], mask, temp_hidden);
            if (i % (VSIZE/2) == 0) {
                ii = 0;
            }
            for (j = 0; j < output_size; ++j) {
                _vim2K_fmuls(&weights[(ii * 2) + (j * (features/2))], mask, temp_weights);
                _vim2K_fmuls(temp_hidden, temp_weights, temp_weights);
                _vim2K_fcums(temp_weights, &output_layer[o_idx++]);
            }
        }
        free(mask);
        for (i = 0; i < o_size; i += VSIZE) {
            _vim2K_fadds(&output_layer[i], bias, &output_layer[i]);
        }
    }

    free(p_sum);
    free(weights);
    free(temp_hidden);
    free(temp_weights);
    return output_layer;
}

void classification(__v32f *output_layer) {
    __v32f sum_exp;
    __v32f *result = (__v32f *)aligned_alloc(vector_size, sizeof(__v32f) * output_size);

    for (int i = 0; i < instances; ++i) {
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
    // total_begin = clock();

    vector_size = atoi(argv[1]);
    instances = atoi(argv[2]);
    features = atoi(argv[3]);
    output_size = atoi(argv[4]);

    if (vector_size == 256) {
        VSIZE = VM64I;
    } else {
        VSIZE = VM2KI;
    }

    n_vectors = features/VSIZE;
    instance_size = n_vectors * VSIZE;
    if (features < VSIZE) {
        n_vectors = 1;
        instance_size = VSIZE/features;
    }

    // hidden_begin = clock();
    __v32f *hidden_layer = relu_layer();
    // hidden_end = clock();
    // aux_spent = init_spent;
    // hidden_spent = ((double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC);


    // output_begin = clock();
    __v32f *output_layer = softmax_layer(hidden_layer);
    // output_end = clock();
    // aux_spent = init_spent - aux_spent;
    // output_spent = ((double)(output_end - output_begin) / CLOCKS_PER_SEC) - aux_spent;

    // class_begin = clock();
    classification(output_layer);
    // class_end = clock();
    // class_spent = (double)(class_end - class_begin) / CLOCKS_PER_SEC;

    // total_end = clock();
    // total_spent = (double)(total_end - total_begin) / CLOCKS_PER_SEC;
    // printf("*************************************\n");
    // printf("* Execution time:         %fs *\n", total_spent);
    // printf(" ***********************************\n");
    // printf("* Initialization time:    %fs *\n", init_spent);
    // printf("* Input x Hidden layer:   %fs *\n", hidden_spent);
    // printf("* Hidden x Output layer:  %fs *\n", output_spent);
    // printf("* Classification time:    %fs *\n", class_spent);
    // printf("*************************************\n");

    free(hidden_layer);
    free(output_layer);
    free(bias);
    return 0;
}