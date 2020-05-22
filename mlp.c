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

__v64d *formated_hidden_weights(int fw_size, int instances) {
    int i, j, k = 0;
    __v64d *f_w = (__v64d *)malloc(sizeof(__v64d) * fw_size * VSIZE);
    srand((unsigned int)time(NULL));
    __v64d a = 10.0, weight;
    for (i = 0; i < training_features/2; ++i) {
        weight = (((__v64d)rand() / (__v64d)(RAND_MAX)) * a) - ((__v64d)rand()/(__v64d)(RAND_MAX) * a);
        if (k == instances) {
            k = 0;
        }
        for (j = k; j < VSIZE; j += instances) {
            f_w[(i/instances) * VSIZE + j] =  weight;
        }
        k++;
    }
    return f_w;
}

__v64d **create_mask_hidden(int size, int stride) {
    int inst_size = ((training_features/2) * training_features), k = 0;
    __v64d **mask = (__v64d **)malloc(sizeof(__v64d *) * size);
    for (int i = 0; i < size; ++i) {
        mask[i] = (__v64d *)malloc(sizeof(__v64d) * VSIZE);
    }

    for (int i = 0; i < size; ++i) {
        if (k == stride) {
            k = 0;
        }
        if (VSIZE == 256 || (VSIZE == 8192 && training_features >= 64)) {
            for (int j = i; j < VSIZE; j += stride) {
                mask[i][j] = 1.0;
            }
        } else {
            for (int j = ((i/stride) * inst_size) + k; j < ((i/stride) * inst_size) + inst_size; j += stride) {
                mask[i][j] = 1.0;
            }
            ++k;
        }
    }
    return mask;
}

__v64d *relu_layer(base_type *train_base) {
    int i, j, k, l = 0, h_idx = 0;
    int fw_size = (training_features / 2) / n_instances;
    int instances = n_instances;
    if (training_features * (training_features / 2) < VSIZE) {
        fw_size = 1;
        instances = training_features/2;
    }
    hidden_size = training_features/2 * training_instances;

    __v64d *f_weights = formated_hidden_weights(fw_size, instances);

    __v64d *partial_mul = (__v64d *)malloc(sizeof(__v64d) * fw_size * n_vectors * VSIZE);
    __v64d *partial_sum = (__v64d *)malloc(sizeof(__v64d) * fw_size * n_vectors * VSIZE);
    __v64d *partial_acc = (__v64d *)malloc(sizeof(__v64d) * VSIZE);
    __v64d *hidden_layer = (__v64d *)malloc(sizeof(__v64d) * hidden_size);
    __v64d sum, p_sum, **mask;

    int m_size, stride;
    if (VSIZE == 256 || (VSIZE == 8192 && training_features >= 64)) {
        stride = VSIZE/training_features;
        m_size = stride;

    } else {
        stride = training_features/2;
        m_size = (VSIZE/(training_features * (training_features/2))) * stride;
    }
    mask = create_mask_hidden(m_size, stride);

    bias = (__v64d *)malloc(sizeof(__v64d) * VSIZE);
    if (vector_size == 256) {
        _vim32_dmovs(1.0, bias);
        for (i = 0; i < training_instances; ++i) {
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
                    if (sum > 0.0) {
                        hidden_layer[h_idx] = sum;
                    }
                    ++h_idx;
                }
            } else {
                for (j = 0; j < m_size; ++j) {
                    _vim32_dmuls(mask[k], partial_mul, partial_acc);
                    _vim32_dcums(partial_acc, &sum);
                    if (sum > 0.0) {
                        hidden_layer[h_idx] = sum;
                    }
                    ++h_idx;
                }
            }
        }
    } else {
        _vim1K_dmovs(1.0, bias);
        for (i = 0; i < base_size; i += VSIZE * n_vectors) {
            for (j = 0; j < fw_size * VSIZE; j += VSIZE) {
                for (k = 0; k < n_vectors; ++k) {
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
                    if (sum > 0.0) {
                        hidden_layer[h_idx] = sum;
                    }
                    ++h_idx;
                }
            } else {
                for (j = 0; j < m_size; ++j) {
                    _vim1K_dmuls(mask[k], partial_mul, partial_acc);
                    _vim1K_dcums(partial_acc, &sum);
                    if (sum > 0.0) {
                        hidden_layer[h_idx] = sum;
                    }
                     ++h_idx;
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

__v64d *formated_output_weights() {

    __v64d a = 10.0, weight;
    __v64d *f_w = (__v64d *)malloc(sizeof(__v64d) * VSIZE * output_size);

    for (int i = 0; i < output_size; ++i) {
        weight = (((__v64d)rand() / (__v64d)(RAND_MAX)) * a) - ((__v64d)rand()/(__v64d)(RAND_MAX) * a);
        for (int j = i * VSIZE; j < (i * VSIZE) + VSIZE; ++j) {
            f_w[j] = weight;
        }
    }
    return f_w;
}

__v64d **create_mask_output(int n_instances) {
    int w_size = training_features/2;
    __v64d **mask = (__v64d **)malloc(sizeof(__v64d *) * n_instances);
    for (int i = 0; i < n_instances; ++i) {
        mask[i] = (__v64d *)calloc(VSIZE, sizeof(__v64d));
        for (int j = i * w_size; j < (i * w_size) + w_size; ++j) {
            mask[i][j] = 1.0;
        }
    }
    return mask;
}

__v64d *softmax_layer(__v64d *hidden_layer) {
    int i, j, k, o_idx = 0;
    __v64d sum, p_sum;

    int out_inst_size = (training_features/2)/VSIZE;
    if (training_features/2 < VSIZE) {
        out_inst_size = 1;
    }
    __v64d *output_layer = (__v64d *)malloc(sizeof(__v64d) * output_size * training_instances);
    __v64d *f_output = formated_output_weights();

    __v64d *partial_mul = (__v64d *)malloc(sizeof(__v64d) * VSIZE * out_inst_size * output_size);
    __v64d *partial_sum = (__v64d *)malloc(sizeof(__v64d) * VSIZE * out_inst_size * output_size);
    __v64d *partial_acc = (__v64d *)malloc(sizeof(__v64d) * VSIZE);

    __v64d **mask;
    if (VSIZE/(training_features/2) > 1) {
        mask = create_mask_output(VSIZE/(training_features/2));
    }

    if (vector_size == 256) {
        for (i = 0; i < hidden_size; i += VSIZE * out_inst_size) {
            for (j = 0; j < output_size * VSIZE; j += VSIZE) {
                for (k = 0; k < out_inst_size * VSIZE; k += VSIZE) {
                    _vim32_dmuls(&hidden_layer[i + k], &f_output[j], &partial_mul[(j * out_inst_size) + k]);
                    _vim32_dadds(bias, &partial_mul[(j * out_inst_size) + k], &partial_sum[(j * out_inst_size) + k]);
                }
            }
            if (training_features/2 >= VSIZE) {
                for (j = 0; j < VSIZE * output_size; j += VSIZE) {
                    sum = 0.0;
                    for (k = 0; k < out_inst_size; ++k) {
                        _vim32_dcums(&partial_sum[(j * out_inst_size) + k], &p_sum);
                        sum += p_sum;
                    }
                    if (sum > 0.0) {
                        output_layer[o_idx] = sum;
                    }
                    ++o_idx;
                }
            } else {
                for (j = 0; j < VSIZE/(training_features/2); ++j) {
                    _vim32_dmuls(mask[j], partial_mul, partial_acc);
                    _vim32_dcums(partial_acc, &sum);
                    if (sum > 0.0) {
                        output_layer[o_idx] = sum;
                    }
                    ++o_idx;
                }
            }
        }
    } else {
        for (i = 0; i < hidden_size; i += VSIZE * out_inst_size) {
            for (j = 0; j < output_size * VSIZE; j += VSIZE) {
                for (k = 0; k < out_inst_size * VSIZE; k += VSIZE) {
                    _vim1K_dmuls(&hidden_layer[i + k], &f_output[j], &partial_mul[(j * out_inst_size) + k]);
                    _vim1K_dadds(bias, &partial_mul[(j * out_inst_size) + k], &partial_sum[(j * out_inst_size) + k]);
                }
            }
            if (training_features/2 >= VSIZE) {
                for (j = 0; j < VSIZE * output_size; j += VSIZE) {
                    sum = 0.0;
                    for (k = 0; k < out_inst_size; ++k) {
                        _vim1K_dcums(&partial_sum[(j * out_inst_size) + k], &p_sum);
                        sum += p_sum;
                    }
                    if (sum > 0.0) {
                        output_layer[o_idx] = sum;
                    }
                    ++o_idx;
                }
            } else {
                for (j = 0; j < VSIZE/(training_features/2); ++j) {
                    _vim1K_dmuls(mask[j], partial_mul, partial_acc);
                    _vim1K_dcums(partial_acc, &sum);
                    if (sum > 0.0) {
                        output_layer[o_idx] = sum;
                    }
                    ++o_idx;
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
    total_begin = clock();
    base_type train_base;

    read_begin = clock();
    read_file(argv, &train_base);
    output_size = atoi(argv[3]);
    read_end = clock();
    read_spent = (double)(read_end - read_begin) / CLOCKS_PER_SEC;

    hidden_begin = clock();
    __v64d *hidden_layer = relu_layer(&train_base);
    hidden_end = clock();
    hidden_spent = (double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC;

    output_begin = clock();
    __v64d *output_layer = softmax_layer(hidden_layer);
    output_end = clock();
    output_spent = (double)(output_end - output_begin) / CLOCKS_PER_SEC;

    norm_begin = clock();
    normalization(output_layer);
    norm_end = clock();
    norm_spent = (double)(norm_end - norm_begin) / CLOCKS_PER_SEC;

    class_begin = clock();
    classification(output_layer);
    class_end = clock();
    class_spent = (double)(class_end - class_begin) / CLOCKS_PER_SEC;

    total_end = clock();
    total_spent = (double)(total_end - total_begin) / CLOCKS_PER_SEC;
    printf("*************************************\n");
    printf("* Execution time:         %lfs *\n", total_spent);
    printf(" ***********************************\n");
    printf("* Read time:              %lfs *\n", read_spent);
    printf("* Input x Hidden layer:   %lfs *\n", hidden_spent);
    printf("* Hidden x Output layer:  %lfs *\n", output_spent);
    printf("* Normalization:          %lfs *\n", norm_spent);
    printf("* Classification time:    %lfs *\n", class_spent);
    printf("*************************************\n");
    free(train_base.base);
    free(train_base.label);
    free(hidden_layer);
    free(output_layer);
    free(bias);
    return 0;
}
