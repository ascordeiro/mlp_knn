#include "mlp.h"

__v32f *relu_layer() {
    int i, j, jj, k, l, h_idx = 0;
    hidden_size = features/2 * instances;
    int weight_size = features * (features/2);
    if (features * (features/2) < VSIZE) {
        weight_size = VSIZE;
    }

    __v32f *mask;
    __v32f *instance_vector = (__v32f *)aligned_alloc(alignment, (instance_size * features * sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));
    __v32f *temp_instance = (__v32f *)aligned_alloc(alignment, (VSIZE * sizeof(__v32f)));
    __v32f *weights = (__v32f *)aligned_alloc(alignment, (weight_size * sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));
    __v32f *temp_weights = (__v32f *)aligned_alloc(alignment, (VSIZE * sizeof(__v32f)));
    __v32f *p_sum = (__v32f *)aligned_alloc(alignment, (VSIZE * sizeof(__v32f)));
    __v32f *hidden_layer = (__v32f *)aligned_alloc(alignment, (hidden_size *  sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));
    __v32f *relu_aux = (__v32f *)aligned_alloc(alignment, (VSIZE * sizeof(__v32f)));

    bias = (__v32f *)aligned_alloc(alignment, sizeof(__v32f) * VSIZE);
    if (vector_size == 256) {
        for (i = 0; i < weight_size; i += VSIZE) {
            _vim64_fmovs(1.0, weights);
        }
        if (features == 8) {
            __v32f *temp_instance2 = (__v32f *)aligned_alloc(alignment, (VSIZE * sizeof(__v32f)));
            __v32f *temp_weights2 = (__v32f *)aligned_alloc(alignment, (VSIZE * sizeof(__v32f)));
            mask = (__v32f *)aligned_alloc(alignment, VSIZE * sizeof(__v32f));
            for (i = 0; i < features; ++i) {
                mask[i] = 1.0;
            }
            for (i = 0; i < instances; i += instance_size) {
                _vim64_fmovs(1.0, instance_vector);
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
            }
            free(mask);
            free(temp_instance2);
            free(temp_weights2);
        } else if (features == 16 || features == 32) {
            mask = (__v32f *)aligned_alloc(alignment, VSIZE * sizeof(__v32f));
            for (i = 0; i < features; ++i) {
                mask[i] = 1.0;
            }
            for (i = 0; i < instances; i += instance_size) {
                _vim64_fmovs(1.0, instance_vector);
                for (j = 0; j < VSIZE; j += features) {
                    _vim64_fmuls(&instance_vector[j], mask, temp_instance);
                    for (k = 0; k < features/2; ++k) {
                        _vim64_fmuls(&weights[k * features], mask, temp_weights);
                        _vim64_fmuls(temp_instance, temp_weights, temp_weights);
                        _vim64_fcums(temp_weights, &hidden_layer[h_idx++]);
                    }
                }
            }
            free(mask);
        } else {
            for (i = 0; i < instances; ++i) {
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
            }
        }
        _vim64_fmovs(1.0, bias);
        _vim64_fmovs(0.0, relu_aux);
        for (i = 0; i < hidden_size; i += VSIZE) {
            _vim64_fadds(&hidden_layer[i], bias, &hidden_layer[i]);
        }
        for (i = 0; i < hidden_size; i += VSIZE) {
            _vim64_fmaxs(&hidden_layer[i], relu_aux, &hidden_layer[i]);
        }
    } else {
        for (i = 0; i < weight_size; i += VSIZE) {
            _vim2K_fmovs(1.0, weights);
        }
        if (features < VSIZE) {
            mask = (__v32f *)aligned_alloc(alignment, VSIZE * sizeof(__v32f));
            for (i = 0; i < features; ++i) {
                mask[i] = 1.0;
            }
            for (i = 0; i < instances; i += instance_size) {
                _vim2K_fmovs(1.0, instance_vector);
                for (j = 0; j < VSIZE; j += features) {
                    _vim2K_fmuls(&instance_vector[j], mask, temp_instance);
                    for (k = 0; k < features/2; ++k) {
                        _vim2K_fmuls(&weights[(k * features)], mask, temp_weights);
                        _vim2K_fmuls(temp_instance, temp_weights, temp_weights);
                        _vim2K_fcums(temp_weights, &hidden_layer[h_idx++]);
                    }
                }
            }
            free(mask);
        } else {
            for (i = 0; i < instances; ++i) {
                for (j = 0; j < n_vectors; ++j) {
                    _vim2K_fmovs(1.0, &instance_vector[j * VSIZE]);
                }
                for (j = 0; j < features/2; ++j) {
                    for (k = 0; k < n_vectors; ++k) {
                        _vim2K_fmuls(&instance_vector[k * VSIZE], &weights[(j * features) + (k * VSIZE)], temp_weights);
                        _vim2K_fcums(temp_weights, &p_sum[k]);
                    }
                    _vim2K_fcums(p_sum, &hidden_layer[h_idx++]);
                }
            }
        }

        _vim2K_fmovs(1.0, bias);
        _vim2K_fmovs(0.0, relu_aux);
        for (i = 0; i < hidden_size; i += VSIZE) {
            _vim2K_fadds(&hidden_layer[i], bias, &hidden_layer[i]);
        }
        for (i = 0; i < hidden_size; i += VSIZE) {
            _vim2K_fmaxs(&hidden_layer[i], relu_aux, &hidden_layer[i]);
        }
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

    __v32f *weights = (__v32f *)aligned_alloc(alignment, (VSIZE * n_vectors/2 * sizeof(__v32f)) + (VSIZE * sizeof(__v32f)));
    __v32f *temp_hidden = (__v32f *)aligned_alloc(alignment, VSIZE * sizeof(__v32f));
    __v32f *temp_weights = (__v32f *)aligned_alloc(alignment, VSIZE * sizeof(__v32f));
    __v32f *p_sum = (__v32f *)aligned_alloc(alignment, VSIZE * sizeof(__v32f));
    __v32f *output_layer = (__v32f *)aligned_alloc(alignment, o_size * sizeof(__v32f));

    if(vector_size == 256) {
        for (i = 0; i < n_vectors/2; ++i) {
            _vim64_fmovs(1.0, weights);
        }
        if (features < 128) {
        __v32f *mask = (__v32f *)aligned_alloc(alignment, sizeof(__v32f) * n_vectors * VSIZE);
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
        if (features < VSIZE) {
            __v32f *mask = (__v32f *)aligned_alloc(alignment, sizeof(__v32f) * n_vectors * VSIZE);
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
        } else {
            for (i = 0; i < hidden_size; i += VSIZE * n_vectors/2) {
                for (j = 0; j < output_size; ++j) {
                    for (k = 0; k < n_vectors/2; ++k) {
                        _vim2K_fmuls(&hidden_layer[i + (k * VSIZE)], weights, temp_weights);
                        _vim2K_fcums(temp_weights, &p_sum[k]);
                    }
                    _vim2K_fcums(p_sum, &output_layer[o_idx]);
                }
            }
        }
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
    __v32f *result = (__v32f *)aligned_alloc(alignment, sizeof(__v32f) * output_size);

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

    vector_size = atoi(argv[1]);
    instances = atoi(argv[2]);
    features = atoi(argv[3]);
    output_size = atoi(argv[4]);

    if (vector_size == 256) {
        alignment = 256;
        VSIZE = VM64I;
    } else {
        alignment = 8192;
        VSIZE = VM2KI;
    }

    n_vectors = features/VSIZE;
    instance_size = 1;
    if (features < VSIZE) {
        n_vectors = 1;
        instance_size = VSIZE/features;
    }

    __v32f *hidden_layer = relu_layer();

    __v32f *output_layer = softmax_layer(hidden_layer);

    classification(output_layer);

    free(hidden_layer);
    free(output_layer);
    free(bias);
    return 0;
}