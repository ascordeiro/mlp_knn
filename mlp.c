#include "mlp.h"


void read_file(char const *argv[], base_type *train_base) {
    int i, j, k;
    vector_size = atoi(argv[1]);
    if (vector_size == 256) {
        VSIZE = VM32L;
    } else {
        VSIZE = VM1KL;
    }

    FILE *f = fopen(argv[2], "r");
    if (fscanf(f, "%d", &training_instances));
    if (fscanf(f, "%d\n", &training_features));

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

    train_base->base = (__v64d *)malloc(sizeof(__v64d) *  base_size);
    train_base->label = (__v32u *)malloc(sizeof(__v32u) * training_instances);
    if (!train_base->base || !train_base->label) {
        printf("Cannot allocate training base\n");
        exit(1);
    }
    __v64d aux;
    if (training_features * (training_features/2) <= VSIZE) {
        for (i = 0; i < training_instances; ++i) {
            if (fscanf(f, "%u", &train_base->label[i]));
            for (j = 0; j < training_features; ++j) {
                if (fscanf(f, "%lf", &aux));
                for (k = j * (training_features/2); k < (j * (training_features/2)) + (training_features/2); ++k) {
                    train_base->base[i * (training_features * (training_features/2)) + k] = aux;
                }
            }
        }
    } else {
        for (i = 0; i < training_instances; ++i) {
            if (fscanf(f, "%u ", &train_base->label[i]));
            for (j = 0; j < VSIZE * n_vectors; j += n_instances) {
                if (fscanf(f, "%lf ", &train_base->base[i * VSIZE * n_vectors + j]));
                for (k = j + 1; k < j + n_instances; ++k) {
                    train_base->base[i * VSIZE * n_vectors + k] = train_base->base[i * VSIZE + j];
                }
            }
        }
    }
    fclose(f);
}

__v64d *generate_weights(int n_weights) {
    int i;
    __v64d *w = (__v64d *)malloc(sizeof(__v64d) * n_weights);
    srand((unsigned int)time(NULL));
    __v64d a = 10.0;
    for (i = 0; i < n_weights; ++i) {
        w[i] = (((__v64d)rand() / (__v64d)(RAND_MAX)) * a) - ((__v64d)rand()/(__v64d)(RAND_MAX) * a);
    }
    return w;
}

__v64d *formated_hidden_weights(__v64d *w, int fw_size, int instances) {
    int i, j, k = 0;
    __v64d *f_w = (__v64d *)malloc(sizeof(__v64d) * fw_size * VSIZE);

    for (i = 0; i < training_features/2; ++i) {
        if (k == instances) {
            k = 0;
        }
        for (j = k; j < VSIZE; j += instances) {
            f_w[(i/instances) * VSIZE + j] =  w[i];
        }
        k++;
    }
    return f_w;
}

void formated_hidden(__v64d *h_layer, __v64d *hidden_layer, int index) {
    int n_iter = 1;
    int t_features = training_features/2;
    if (training_features/2 < VSIZE) {
        t_features = training_features;
        n_iter = output_size;
    }
    for (int i = 0; i < n_iter; ++i) {
        for (int j = 0; j < training_features/2; ++j) {
            if (h_layer[j] > 0.0)
            hidden_layer[index * t_features + i * training_features/2 + j] = h_layer[j];
        }
    }
}

__v64d *relu_layer(base_type *train_base, __v64d *weights) {
    int i, j, k, l = 0, h_idx = 0;
    int fw_size = (training_features / 2) / n_instances;
    int instances = n_instances;
    if (training_features * (training_features / 2) < VSIZE) {
        fw_size = 1;
        instances = training_features/2;
    }
    hidden_size = training_features * training_instances;
    if (training_features > VSIZE) {
        hidden_size = (training_features/2) * training_instances; 
    } 
    __v64d *f_weights = formated_hidden_weights(weights, fw_size, instances);
    int count = 0;

    __v64d *partial_mul = (__v64d *)malloc(sizeof(__v64d) * fw_size * n_vectors * VSIZE);
    __v64d *partial_sum = (__v64d *)malloc(sizeof(__v64d) * fw_size * n_vectors * VSIZE);
    __v64d *h_layer = (__v64d *)malloc(sizeof(__v64d) * training_features/2);
    __v64d *hidden_layer = (__v64d *)malloc(sizeof(__v64d) * hidden_size);
    __v64d sum, p_sum;

    bias = (__v64d *)malloc(sizeof(__v64d) * VSIZE);
    if (vector_size == 256) {
        _vim32_dmovs(1.0, bias);
        for (i = 0; i < training_instances; ++i) {
            h_idx = 0;
            for (j = 0; j < fw_size; ++j) {
                for (k = 0; k < n_vectors; ++k) {
                    _vim32_dmuls(&train_base->base[i * VSIZE * n_vectors + k * VSIZE], &f_weights[j * VSIZE], &partial_mul[j * VSIZE * n_vectors + k * VSIZE]);
                    _vim32_dadds(bias, &partial_mul[j * VSIZE * n_vectors + k * VSIZE], &partial_sum[j * VSIZE * n_vectors + k * VSIZE]);
                }
            }
            if (training_features >= VSIZE) {
                for (j = 0; j < fw_size; ++j) {
                    sum = 0.0;
                    for (k = 0; k < n_vectors; ++k) {
                        _vim32_dcums(&partial_sum[j * VSIZE * n_vectors + k * VSIZE], &p_sum);
                        sum += p_sum;
                    }
                    h_layer[h_idx++] = sum;
                }
            } else {
                for (j = 0; j < training_features/2; ++j) {
                    sum = 0.0;
                    if (l == instances) {
                        l = 0;
                    }
                    for (k = l; k < VSIZE; k += instances) {
                        sum += partial_sum[j / instances * VSIZE + k];
                    }
                    l++;
                    h_layer[h_idx++] = sum;
                }
            }
            formated_hidden(h_layer, hidden_layer, i);
        }
    } else {
        _vim1K_dmovs(1.0, bias);
        for (i = 0; i < base_size; i += VSIZE * n_vectors) {
            h_idx = 0;
            for (j = 0; j < fw_size * VSIZE; j += VSIZE) {
                for (k = 0; k < n_vectors; ++k) {
                    // printf("idx_base: %d - idx_w: %d - idx_mul: %d - idx_sum: %d\n", i, j, j, j);
                    _vim1K_dmuls(&train_base->base[i + k * VSIZE], &f_weights[j], &partial_mul[j * n_vectors + k * VSIZE]);
                    _vim1K_dadds(bias, &partial_mul[j * n_vectors + k * VSIZE], &partial_sum[j * n_vectors + k * VSIZE]);
                }
            }
            if (training_features >= VSIZE) {
                for (j = 0; j < fw_size; ++j) {
                    sum = 0.0;
                    for (k = 0; k < n_vectors; ++k) {
                        _vim1K_dcums(&partial_sum[j * VSIZE + k * VSIZE], &p_sum);
                        sum += p_sum;
                    }
                    h_layer[h_idx++] = sum;
                }
                formated_hidden(h_layer, hidden_layer, count);
                count++;
            } else {
                
                for (j = 0; j < VSIZE; j += training_features * (training_features / 2)) {
                    h_idx = 0;
                    for (k = j; k < j + (training_features/2); ++k) {
                        sum = 0.0;
                        for (l = k; l < j + training_features * (training_features / 2); l += instances) {
                            sum += partial_sum[l];
                        }
                        h_layer[h_idx++] = sum;
                    }
                    formated_hidden(h_layer, hidden_layer, count);
                    count++;
                }
            }
        }
    }
    free(partial_mul);
    free(partial_sum);
    free(h_layer);
    free(f_weights);
    return hidden_layer;
}

__v64d *formated_output_weights(__v64d *w) {
    int features = training_features;
    if (training_features/2 > VSIZE) {
        features = VSIZE * 2;
    }
    __v64d *f_w = (__v64d *)malloc(sizeof(__v64d) * VSIZE * o_vectors);
    for (int i = 0; i < output_size; ++i) {
        for (int j = i * features/2; j < VSIZE * o_vectors; j += features) {
            for (int k = j; k < j + features/2; ++k) {
                f_w[k] = w[i];
            } 
        }
    }
    return f_w;
}

__v64d *softmax_layer(__v64d *hidden_layer, __v64d *output_weights) {
    int i, j, k, o_idx = 0;
    __v64d sum, p_sum;
    o_vectors = output_size;
    int instance_size = (training_features/2)/VSIZE;
    if (training_features/2 < VSIZE) {
        instance_size = 1;
        o_vectors = 1;
    }
    __v64d *output_layer = (__v64d *)malloc(sizeof(__v64d) * output_size * training_instances);
    __v64d *f_output = formated_output_weights(output_weights);

    __v64d *partial_mul = (__v64d *)malloc(sizeof(__v64d) * hidden_size * o_vectors);
    __v64d *partial_sum = (__v64d *)malloc(sizeof(__v64d) * hidden_size * o_vectors);
    if (vector_size == 256) {
        for (i = 0; i < hidden_size; i += VSIZE * instance_size) {
            for (j = 0; j < o_vectors; ++j) {
                for (k = 0; k < instance_size; ++k) {
                    _vim32_dmuls(&hidden_layer[i + k * VSIZE], &f_output[j * VSIZE], &partial_mul[i * o_vectors + j * VSIZE * instance_size + k * VSIZE]);
                    _vim32_dadds(bias, &partial_mul[i * o_vectors + j * VSIZE * instance_size + k * VSIZE], &partial_sum[i * o_vectors + j * VSIZE * instance_size + k * VSIZE]);
                }
            }
        }
        if (training_features/2 >= VSIZE) {
            for (i = 0; i < hidden_size * o_vectors; i += VSIZE * instance_size) {
                sum = 0.0;
                for (j = 0; j < instance_size; ++j) {
                    _vim32_dcums(&partial_sum[i + j * VSIZE], &p_sum);
                    sum += p_sum;
                }
                output_layer[o_idx++] = sum;
            }
        }
    } else {
        for (i = 0; i < hidden_size; i += VSIZE * instance_size) {
            for (j = 0; j < o_vectors; ++j) {
                for (k = 0; k < instance_size; ++k) {
                    _vim1K_dmuls(&hidden_layer[i + k * VSIZE], &f_output[j * VSIZE], &partial_mul[i * o_vectors + j * VSIZE * instance_size + k * VSIZE]);
                    _vim1K_dadds(bias, &partial_mul[i * o_vectors + j * VSIZE * instance_size + k * VSIZE], &partial_sum[i * o_vectors + j * VSIZE * instance_size + k * VSIZE]);
                }
            }
        }
        if (training_features/2 >= VSIZE) {
            for (i = 0; i < hidden_size * o_vectors; i += VSIZE * instance_size) {
                sum = 0.0;
                for (j = 0; j < instance_size; ++j) {
                    _vim1K_dcums(&partial_sum[i + j * VSIZE], &p_sum);
                    sum += p_sum;
                }
                output_layer[o_idx++] = sum;
            }
        }
    }
    if (training_features/2 < VSIZE) {
        for (i = 0; i < hidden_size * o_vectors; i += training_features / 2) {
            sum = 0.0;
            for (j = i; j < i + training_features/2; ++j) {
                sum += partial_sum[j];
            }
            output_layer[o_idx++] = sum;
        }
    }
    free(f_output);
    free(partial_sum);
    free(partial_mul);
    return output_layer;
}

void classification(__v64d *output_layer) {
    __v64d sum_exp;
    __v64d *result = (__v64d *)malloc(sizeof(__v64d) * output_size);
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
}

void normalization(__v64d *output_layer) {
    __v64d min = output_layer[0];
    __v64d max = output_layer[0];
    for (int i = 1; i < output_size * training_instances; ++i) {
        if (output_layer[i] < min) {
            min = output_layer[i];
        }
        if (output_layer[i] > max) {
            max = output_layer[i];
        }
    }
    __v64d max_min = max - min;
    for(int i = 0; i < output_size * training_instances; ++i) {
        output_layer[i] = (output_layer[i] - min)/max_min;
    }
}

int main(int argc, char const *argv[]) {
    clock_t begin = clock();
    base_type train_base;
    read_file(argv, &train_base);
    output_size = atoi(argv[3]);

    __v64d *hidden_weights = generate_weights(training_features/2);
    __v64d *hidden_layer = relu_layer(&train_base, hidden_weights);
    __v64d *output_weights = generate_weights(output_size);
    __v64d *output_layer = softmax_layer(hidden_layer, output_weights);
    normalization(output_layer);
    classification(output_layer);

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Execution time: %lf\n\n\n\n", time_spent);
    free(train_base.base);
    free(train_base.label);
    free(hidden_layer);
    free(output_layer);
    free(hidden_weights);
    free(output_weights);
    free(bias);
    return 0;
}