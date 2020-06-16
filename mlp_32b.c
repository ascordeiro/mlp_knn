#include "mlp_32b.h"


void read_file(char const *argv[], base_type *train_base) {
    int i, j, k;
    vector_size = atoi(argv[1]);
    if (vector_size == 256) {
        VSIZE = VM64I;
    } else {
        VSIZE = VM2KI;
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

    train_base->base = (__v32f *)calloc(base_size, sizeof(__v32f));
    train_base->label = (__v32u *)calloc(training_instances, sizeof(__v32u));
    if (!train_base->base || !train_base->label) {
        printf("Cannot allocate training base\n");
        exit(1);
    }
    __v32f aux;
    if (training_features * (training_features/2) <= VSIZE) {
        for (i = 0; i < training_instances; ++i) {
            if (fscanf(f, "%u", &train_base->label[i]));
            for (j = 0; j < training_features; ++j) {
                if (fscanf(f, "%f", &aux));
                for (k = j * (training_features/2); k < (j * (training_features/2)) + (training_features/2); ++k) {
                    train_base->base[i * (training_features * (training_features/2)) + k] = aux;
                }
            }
        }
    } else {
        for (i = 0; i < training_instances; ++i) {
            if (fscanf(f, "%u ", &train_base->label[i]));
            for (j = 0; j < VSIZE * n_vectors; j += n_instances) {
                if (fscanf(f, "%f ", &train_base->base[i * VSIZE * n_vectors + j]));
                for (k = j + 1; k < j + n_instances; ++k) {
                    train_base->base[i * VSIZE * n_vectors + k] = train_base->base[i * VSIZE + j];
                }
            }
        }
    }
    fclose(f);
}

__v32f *formated_hidden_weights(int fw_size, int instances) {
    int i, j, k = 0;
    __v32f *f_w = (__v32f *)calloc(fw_size * VSIZE, sizeof(__v32f));
    srand((unsigned int)time(NULL));
    __v32f a = 1.0, weight;
    for (i = 0; i < training_features/2; ++i) {
        weight = (((__v32f)rand() / (__v32f)(RAND_MAX)) * a) - ((__v32f)rand()/(__v32f)(RAND_MAX) * a);
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

__v32f **create_mask_hidden(int size, int stride) {
    int i, j, inst_size = ((training_features/2) * training_features), k = 0;
    __v32f **mask = (__v32f **)calloc(size, sizeof(__v32f *));
    for (i = 0; i < size; ++i) {
        mask[i] = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    }

    for (i = 0; i < size; ++i) {
        if (k == stride) {
            k = 0;
        }
        if ((vector_size == 256 && training_features >= 16) || (vector_size == 8192 && training_features >= 64)) {
            for (j = i; j < VSIZE; j += stride) {
                mask[i][j] = 1.0;
            }
        } else {
            for (j = ((i/stride) * inst_size) + k; j < ((i/stride) * inst_size) + inst_size; j += stride) {
                mask[i][j] = 1.0;
            }
            ++k;
        }
    }
    return mask;
}

__v32f *relu_layer(base_type *train_base) {
    int i, j, k, h_idx = 0;
    int fw_size = (training_features / 2) / n_instances;
    int instances = n_instances;
    if (training_features * (training_features / 2) < VSIZE) {
        fw_size = 1;
        instances = training_features/2;
    }
    hidden_size = training_features/2 * training_instances;
    if (hidden_size < VSIZE) {
        hidden_size = VSIZE;
    }
    __v32f *f_weights = formated_hidden_weights(fw_size, instances);

    __v32f *partial_mul = (__v32f *)calloc(fw_size * n_vectors * VSIZE, sizeof(__v32f));
    __v32f *partial_sum = (__v32f *)calloc(fw_size * n_vectors * VSIZE, sizeof(__v32f));
    __v32f *partial_acc = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    __v32f *hidden_layer = (__v32f *)calloc(hidden_size, sizeof(__v32f));
    __v32f sum, p_sum, **mask;

    int m_size, stride;
    if ((vector_size == 256 && training_features >= 16) || (vector_size == 8192 && training_features >= 64)) {
        stride = VSIZE/training_features;
        m_size = stride;
    } 
    else {
        stride = training_features/2;
        m_size = (VSIZE/(training_features * (training_features/2))) * stride;
    }
    mask = create_mask_hidden(m_size, stride);

    bias = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    if (vector_size == 256) {
        _vim64_fmovs(1.0, bias);
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
        _vim2K_fmovs(1.0, bias);
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

__v32f *formated_output_weights() {

    __v32f a = 1.0, weight;
    __v32f *f_w = (__v32f *)calloc(VSIZE * output_size, sizeof(__v32f));

    for (int i = 0; i < output_size; ++i) {
        weight = (((__v32f)rand() / (__v32f)(RAND_MAX)) * a) - ((__v32f)rand()/(__v32f)(RAND_MAX) * a);
        for (int j = i * VSIZE; j < (i * VSIZE) + VSIZE; ++j) {
            f_w[j] = weight;
        }
    }
    return f_w;
}

__v32f **create_mask_output(int n_instances) {
    int w_size = training_features/2;
    __v32f **mask = (__v32f **)calloc(n_instances, sizeof(__v32f *));
    for (int i = 0; i < n_instances; ++i) {
        mask[i] = (__v32f *)calloc(VSIZE, sizeof(__v32f));
        for (int j = i * w_size; j < (i * w_size) + w_size; ++j) {
            mask[i][j] = 1.0;
        }
    }
    return mask;
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


    __v32f *output_layer = (__v32f *)calloc(olayer_size, sizeof(__v32f));
    __v32f *f_output = formated_output_weights();

    __v32f *partial_mul = (__v32f *)calloc(VSIZE * out_inst_size * output_size, sizeof(__v32f));
    __v32f *partial_sum = (__v32f *)calloc(VSIZE * out_inst_size * output_size, sizeof(__v32f));
    __v32f *partial_acc = (__v32f *)calloc(VSIZE, sizeof(__v32f));

    __v32f **mask;
    if (VSIZE/(training_features/2) > 1) {
        mask = create_mask_output(VSIZE/(training_features/2));
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
    __v32f *result = (__v32f *)calloc(output_size, sizeof(__v32f));
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
    __v32f *broad_minmax = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    __v32f *broad_min = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    __v32f *partial_sub = (__v32f *)calloc(VSIZE, sizeof(__v32f));
    if (size < VSIZE) {
        size = VSIZE;
    }
    __v32f *norm = (__v32f *)calloc(size, sizeof(__v32f));

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
    output_size = atoi(argv[3]);
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
