#include "knn.h"

void read_files(char const *argv[], base_type *train_base) {
    int i, j, k, l;
    read_begin = clock();
    vector_size = atoi(argv[1]);
    if (vector_size == 256) {
        VSIZE = VM64I;
    } else {
        VSIZE = VM2KI;
    }

    FILE *f = fopen(argv[2], "r");
    if (fscanf(f, "%d", &training_instances));
    if (fscanf(f, "%d\n", &training_features));

    int n_vectors = training_features/VSIZE;
    int n_copies = 1;
    if (training_features < VSIZE) {
        n_copies = VSIZE/training_features;
        n_vectors = 1;
    }

    label_instances = 1;
    if (training_instances > VSIZE) {
        label_instances = training_instances/VSIZE;
    }

    train_base->base = (__v32f *)calloc(training_instances * n_vectors * VSIZE, sizeof(__v32f));
    train_base->label = (__v32u *)calloc(label_instances * VSIZE, sizeof(__v32u));
    if (!train_base->base || !train_base->label) {
        printf("Cannot allocate training base\n");
        exit(1);
    }

    for (i = 0; i < training_instances; ++i) {
        if (fscanf(f, "%u ", &train_base->label[i]));
        for (j = 0; j < training_features; ++j) {
            if (fscanf(f, "%f ", &train_base->base[i * VSIZE + j]));
        }
        for (j = 1; j < n_copies; ++j) {
            for (k = j * training_features, l = 0; l < training_features; ++k, ++l) {
                train_base->base[i * VSIZE + k] = train_base->base[i * VSIZE + l];
            }
        }
    }
    fclose(f);
    read_end = clock();
    read_spent = (double)(read_end - read_begin) / CLOCKS_PER_SEC;
}


__v32f **create_mask(int n_instances) {
    __v32f **mask = (__v32f **)calloc(n_instances, sizeof(__v32f *));
    for (int i = 0; i < n_instances; ++i) {
        mask[i] = (__v32f *)calloc(VSIZE, sizeof(__v32f));
        for (int j = i * training_features; j < (i * training_features) + training_features; ++j) {
            mask[i][j] = 1.0;
        }
    }
    return mask;
}

// sqrt(pow((x1 - y1), 2) + pow((x2 - y2), 2) + ... + pow((xn - yn), 2))
void classification(char const *argv[], base_type *train_base) {
    int test_instances, test_features, i, j, jj;
    base_type test_base;

    k = atoi(argv[4]);

    FILE *f = fopen(argv[3], "r");
    if (fscanf(f, "%d ", &test_instances));
    if (fscanf(f, "%d\n", &test_features));

    int v_tesize = test_features/VSIZE;
    int n_instances = 1;
    if (test_features < VSIZE) {
        v_tesize = 1;
        n_instances = VSIZE/test_features;
    }

    test_base.base = (__v32f *)calloc(v_tesize * VSIZE, sizeof(__v32f));
    test_base.label = (__v32u *)calloc(n_instances, sizeof(__v32u));
    if (!test_base.base || !test_base.label) {
        printf("Cannot allocate train base\n");
        exit(1);
    }

    __v32f **e_distance = (__v32f **)calloc(n_instances, sizeof(__v32f *));
    for (i = 0; i < n_instances; ++i) {
        e_distance[i] = (__v32f *)calloc(training_instances, sizeof(__v32f));
    }
    __v32f *partial_sub = (__v32f *)malloc(sizeof(__v32f) * v_tesize * VSIZE);
    __v32f *partial_mul = (__v32f *)malloc(sizeof(__v32f) * v_tesize * VSIZE);
    __v32f *partial_acc = (__v32f *)malloc(sizeof(__v32f) * VSIZE);
    __v32f **mask;
    if (test_features < VSIZE) {
        mask = create_mask(n_instances);
    }

    for (i = 0; i < test_instances; i += n_instances) {
        for (j = 0; j < n_instances; ++j) {
            if (fscanf(f, "%u ", &test_base.label[j]));
            for (jj = j * test_features; jj < (j * test_features) + test_features; ++jj) {
                if (fscanf(f, "%f ", &test_base.base[jj]));
            }
        }
        __v32u *knn = (__v32u *)calloc(k, sizeof(__v32u));
        free(knn);
    }
    fclose(f);
    for (i = 0; i < n_instances; i++) {
        free(e_distance[i]);
    }
    if (test_features < VSIZE) {
        for (i = 0; i < n_instances; i++) {
            free(mask[i]);
        }
        free(mask);
    }
    free(e_distance);
    free(partial_sub);
    free(partial_mul);
    free(partial_acc);
    free(test_base.base);
    free(test_base.label);
}

int main(int argc, char const *argv[]) {
    total_begin = clock();
    base_type train_base;

    // Initialize train and test matrix
    read_files(argv, &train_base);

    // Calculates Euclidean Distance
    classification(argv, &train_base);

    total_end = clock();
    total_spent = (double)(total_end - total_begin) / CLOCKS_PER_SEC;
    printf("**************************************\n");
    printf("* Execution time:          %fs *\n", total_spent);
    printf(" ************************************\n");
    printf("* Read time:               %fs *\n", read_spent);
    printf("* Euclidean Distance time: %fs *\n", ed_spent);
    printf("**************************************\n");
    free(train_base.base);
    free(train_base.label);
    return 0;
}
