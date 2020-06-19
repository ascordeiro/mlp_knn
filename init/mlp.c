#include "mlp.h"

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

    train_base->base = (__v32f *)malloc(sizeof(__v32f) * base_size);
    train_base->label = (__v32u *)malloc(sizeof(__v32u) * training_instances);
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
    __v32f *f_w = (__v32f *)malloc(sizeof(__v32f) * fw_size * VSIZE);
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
    int i;
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
    __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * fw_size * n_vectors * VSIZE);
    __v32f *partial_sum = (__v32f *)malloc(sizeof(__v32f) * fw_size * n_vectors * VSIZE);
    __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    __v32f *hidden_layer = (__v32f *)calloc(hidden_size, sizeof(__v32f));

    int m_size, stride;
    if ((vector_size == 256 && training_features >= 16) || (vector_size == 8192 && training_features >= 64)) {
        stride = VSIZE/training_features;
        m_size = stride;
    } 
    else {
        stride = training_features/2;
        m_size = (VSIZE/(training_features * (training_features/2))) * stride;
    }
    __v32f **mask = create_mask_hidden(m_size, stride);

    bias = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    if (vector_size == 256) {
        _vim64_fmovs(1.0, bias);
    } else {
        _vim2K_fmovs(1.0, bias);
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
    __v32f *f_w = (__v32f *)malloc(sizeof(__v32f) * VSIZE * output_size);

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

__v32f *softmax_layer() {
    int i;

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
    __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * VSIZE * out_inst_size * output_size);
    __v32f *partial_sum = (__v32f *)malloc(sizeof(__v32f) * VSIZE * out_inst_size * output_size);
    __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);

    __v32f **mask;
    if (VSIZE/(training_features/2) > 1) {
        mask = create_mask_output(VSIZE/(training_features/2));
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

    output_begin = clock();
    __v32f *output_layer = softmax_layer();
    output_end = clock();
    output_spent = (double)(output_end - output_begin) / CLOCKS_PER_SEC;

    total_end = clock();
    total_spent = (double)(total_end - total_begin) / CLOCKS_PER_SEC;
    printf("*************************************\n");
    printf("* Execution time:         %fs *\n", total_spent);
    printf(" ***********************************\n");
    printf("* Read time:              %fs *\n", read_spent);
    printf("* Input x Hidden layer:   %fs *\n", hidden_spent);
    printf("* Hidden x Output layer:  %fs *\n", output_spent);
    printf("*************************************\n");
    free(train_base.base);
    free(train_base.label);
    free(hidden_layer);
    free(output_layer);
    free(bias);
    return 0;
}
