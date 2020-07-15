#include "mlp_avx.h"

void init_vec(char const *argv[]) {
    training_instances = atoi(argv[1]);
    training_features = atoi(argv[2]);

    base_size = training_features * training_instances;

    base = (float *)aligned_alloc(32, sizeof(float)*base_size);
    label = (__uint32_t *)aligned_alloc(32, sizeof(__uint32_t)*training_instances);
}

float *relu_layer() {
    int i, j, k, h_idx = 0;
    int hlayer_size = training_features/2;
    int w_size = hlayer_size * (training_features * training_instances);
    int n_vectors = training_features/AVX_SIZE;

    hidden_size = hlayer_size * training_instances;

    __m512 avx_base, avx_weights, avx_bias, avx_pmul, avx_psum, avx_phidden;
    __mmask16 avx_mask[2] = {0xff00, 0xff};
    float *h_weights = (float *)aligned_alloc(32, w_size * sizeof(float));
    float *hidden_layer = (float *)aligned_alloc(32, hidden_size * sizeof(float));
    float sum;

    if (training_features < AVX_SIZE) {
        for (i = 0; i < base_size; i += training_features) {
            avx_base = _mm512_setr_ps(base[i], base[i+1], base[i+2], base[i+3], base[i+4], base[i+5], base[i+6], base[i+7], 
                                      base[i], base[i+1], base[i+2], base[i+3], base[i+4], base[i+5], base[i+6], base[i+7]);
            for (j = i * hlayer_size; j < i * hlayer_size + (training_features * hlayer_size); j += AVX_SIZE) {
                avx_weights = _mm512_load_ps(&h_weights[j]);
                avx_pmul = _mm512_mul_ps(avx_base, avx_weights);
                for (k = 0; k < 2; k++) {
                    hidden_layer[h_idx++] = _mm512_mask_reduce_add_ps(&avx_mask[k], avx_pmul);
                }
            }
        }    
    } else {
        for (i = 0; i < base_size; i += training_features) {
            for (j = i * hlayer_size; j < i * hlayer_size + (training_features * hlayer_size); j += training_features) {
                sum = 0.0;
                for (k = 0; k < n_vectors; k++) {
                    avx_base = _mm512_load_ps(&base[i + k * AVX_SIZE]);
                    avx_weights = _mm512_load_ps(&h_weights[j + k * AVX_SIZE]);
                    avx_pmul = _mm512_mul_ps(avx_base, avx_weights);
                    sum += mm512_reduce_add_ps(avx_pmul);
                }
                hidden_layer[h_idx++] = sum;
            }
        }
    }
    h_idx = 0;
    for (i = 0; i < hidden_size; i += AVX_SIZE) {
        avx_phidden = _mm512_load_ps(&hidden_layer[i]);
        avx_psum = _mm512_add_ps(avx_phidden, avx_bias);
        _mm512_store_ps(&hidden_layer[h_idx], avx_psum);
        for (j = h_idx; j < h_idx + AVX_SIZE; j++) {
            if (hidden_layer[j] < 0.0) {
                hidden_layer[j] = 0.0;
            }
            h_idx += AVX_SIZE;
        }
    }

    free(h_weights);
    return hidden_layer;
}

float *softmax_layer(float *hidden_layer) {
    int i, j, k, o_idx = 0;
    float sum, p_sum;
    int olayer_size = output_size * training_instances, hlayer_size = training_features/2;
    int oweights_size = training_features * training_instances;
    int n_vectors = hlayer_size/AVX_SIZE;
    float *output_layer = (float *)aligned_alloc(32, olayer_size * sizeof(float));
    float *o_weights = (float *)aligned_alloc(32, oweights_size * sizeof(float));

    __m512 avx_pmul, avx_psum, avx_hidden, avx_oweights, avx_output, bias;
    __mmask16 avx_mask8[4] = {0xf0000, 0xf00, 0xf0, 0xf};
    __mmask16 avx_mask16[2] = {0xff00, 0xff};

    if (training_features == 8) {
        for (i = 0; i < hidden_size; i += training_features) {
            avx_hidden = _mm512_setr_ps(hidden_layer[i], hidden_layer[i+1], hidden_layer[i+2], hidden_layer[i+3], 
                                        hidden_layer[i], hidden_layer[i+1], hidden_layer[i+2], hidden_layer[i+3],
                                        hidden_layer[i+4], hidden_layer[i+5], hidden_layer[i+6], hidden_layer[i+7],
                                        hidden_layer[i+4], hidden_layer[i+5], hidden_layer[i+6], hidden_layer[i+7]);
            for (j = 0; j < oweights_size; j += AVX_SIZE) {
                avx_oweights = _mm512_load_ps(&o_weights[j]);
                avx_pmul = _mm512_mul_ps(avx_hidden, avx_oweights);
                for (k = 0; k < 4; k++) {
                    output_layer[o_idx++] = _mm512_mask_reduce_add_ps(&avx_mask8[k], avx_pmul);
                }
            }
        }

    } else if (training_features == 16) {
        for (i = 0; i < hidden_size; i += hlayer_size) {
            avx_hidden = _mm512_setr_ps(hidden_layer[i], hidden_layer[i+1], hidden_layer[i+2], hidden_layer[i+3], 
                                        hidden_layer[i+4], hidden_layer[i+5], hidden_layer[i+6], hidden_layer[i+7], 
                                        hidden_layer[i], hidden_layer[i+1], hidden_layer[i+2], hidden_layer[i+3], 
                                        hidden_layer[i+4], hidden_layer[i+5], hidden_layer[i+6], hidden_layer[i+7]);
        }
        for (j = 0; j < oweights_size; j += AVX_SIZE) {
            avx_oweights = _mm512_load_ps(&o_weights[j]);
            avx_pmul = _mm512_mul_ps(avx_hidden, avx_oweights);
            for (k = 0; k < 2; k++) {
                output_layer[o_idx++] = _mm512_mask_reduce_add_ps(&avx_mask16[k], avx_pmul);
            }
        }
    } else {
        for (i = 0; i < hidden_size; i += hlayer_size) {
            for (j = i * output_size; j < (i * output_size) + training_features; j += hlayer_size) {
                sum = 0.0;
                for (k = 0; k < n_vectors; k++) {
                    avx_hidden = _mm512_load_ps(&hidden_layer[i + k * AVX_SIZE]);
                    avx_oweights = _mm512_load_ps(&o_weights[j + k * AVX_SIZE]);
                    avx_pmul = _mm512_mul_ps(avx_hidden, avx_oweights);
                    sum += mm512_reduce_add_ps(avx_pmul);
                }
                output_layer[o_idx++] = sum;
            }
        }
    }
    o_idx = 0;
    for (i = 0; i < olayer_size; i += AVX_SIZE) {
        avx_output = _mm512_load_ps(&output_layer[i]);
        avx_psum = _mm512_add_ps(avx_output, bias);
        _mm512_store_ps(&output_layer[o_idx], avx_psum);
        for (j = o_idx; j < o_idx + AVX_SIZE; j++) {
            if (output_layer[j] < 0.0) {
                output_layer[j] = 0.0;
            }
            o_idx += AVX_SIZE;
        }
    }

    free(o_weights);
    return output_layer;
}

void classification(float *output_layer) {
    float *sum_exp = (float *)aligned_alloc(32, sizeof(float) * training_instances);
    float *result = (float *)aligned_alloc(32, sizeof(float) * output_size);

    for (int i = 0; i < training_instances; ++i) {
        sum_exp[i] = 0.0;
        for (int j = 0; j < output_size; ++j) {
            sum_exp[i] += exp(output_layer[i * output_size + j]);
        }
    }

    __m512 avx_sumexp, avx_output, avx_div;
    for (int i = 0; i < training_instances; i += AVX_SIZE/2) {
        avx_sumexp = _mm512_setr_ps(sum_exp[i], sum_exp[i], sum_exp[i+1], sum_exp[i+1], sum_exp[i+2], sum_exp[i+2], 
                                    sum_exp[i+3], sum_exp[i+3], sum_exp[i+4], sum_exp[i+4], sum_exp[i+5], sum_exp[i+5], 
                                    sum_exp[+6], sum_exp[i+6], sum_exp[i+7], sum_exp[i+7]);
        avx_output = _mm512_load_ps(&output_layer[i * output_size]);
        avx_div = _mm512_div_ps(avx_output, avx_sumexp);
        for (int j = 0; j < AVX_SIZE; j += output_size) {
            if (result[j] > result[j+1]) {
                printf("%d. %s\n", j, "neg");
            } else {
                printf("%d. %s\n", i, "pos");
            }
        }
    }

    free(result);
}

float *normalization(float *layer, int size) {
    __m512 avx_aux, avx_psub, avx_layer, avx_norm;
    float *norm = (float *)aligned_alloc(32, sizeof(float) * size);

    float min = layer[0];
    float max = layer[0];
    for (int i = 1; i < size; ++i) {
        if (layer[i] < min) {
            min = layer[i];
        }
        if (layer[i] > max) {
            max = layer[i];
        }
    }
    float max_min = max - min;

    __m512 avx_minmax = _mm512_set1_ps(max_min);
    __m512 avx_min = _mm512_set1_ps(min);
    __m512 avx_neg = _mm512_set1_ps((float)-1.0);

    for (int i = 0; i < size; i += AVX_SIZE) {
        avx_layer = _mm512_load_ps(&layer[i]);
        avx_aux = _mm512_mul_ps(avx_layer, avx_neg);
        avx_psub = _mm512_add_ps(avx_aux, avx_min);
        avx_norm = _mm512_div_ps(avx_psub, avx_minmax);
        _mm512_store_ps(&norm[i], avx_norm);  
    }
    return norm;
}

int main(int argc, char const *argv[]) {
    total_begin = clock();

    read_begin = clock();
    init_vec(argv);
    output_size = atoi(argv[3]);
    read_end = clock();
    read_spent = (double)(read_end - read_begin) / CLOCKS_PER_SEC;

    hidden_begin = clock();
    float *hidden_layer = relu_layer();
    hidden_end = clock();
    hidden_spent = (double)(hidden_end - hidden_begin) / CLOCKS_PER_SEC;

    norm_begin1 = clock();
    float *hidden_norm = normalization(hidden_layer, hidden_size);
    norm_end1 = clock();
    norm_spent1 = (double)(norm_end1 - norm_begin1) / CLOCKS_PER_SEC;

    output_begin = clock();
    float *output_layer = softmax_layer(hidden_norm);
    output_end = clock();
    output_spent = (double)(output_end - output_begin) / CLOCKS_PER_SEC;

    norm_begin2 = clock();
    float *output_norm = normalization(output_layer, output_size*training_instances);
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
    free(base);
    free(label);
    free(hidden_norm);
    free(hidden_layer);
    free(output_layer);
    free(output_norm);
    free(bias);
    return 0;
}
