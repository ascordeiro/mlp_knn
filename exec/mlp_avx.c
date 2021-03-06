#include "mlp_avx.h"

void initialize_weights(float *weights, int size) {
    __m512 avx_weights;
    for (int i = 0; i < size; i += AVX_SIZE) {
        avx_weights = _mm512_load_ps(&weights[i]);
        avx_weights = _mm512_set1_ps((float)0.5);
	    _mm512_store_ps(&weights[i], avx_weights);
    }
}

void read_instance(float *instance, int size) {
    __m512 avx_instance;
    for (int i = 0; i < size; i += AVX_SIZE) {
        avx_instance = _mm512_load_ps(&instance[i]);
        avx_instance = _mm512_set1_ps((float)1.0);
	    _mm512_store_ps(&instance[i], avx_instance);
    }
}

float *relu_layer() {
    int i, j, k, h_idx = 0;
    int hlayer_size = features/2;
    int w_size = hlayer_size * features;
    int n_vectors = features/AVX_SIZE;

    hidden_size = hlayer_size * instances;

    __m512 avx_base, avx_weights, avx_bias, avx_phidden;
    __mmask16 avx_mask[2] = {0xff00, 0xff};
    avx_bias = _mm512_set1_ps((float)1.0);

    float *instance = (float *)aligned_alloc(64, features * sizeof(float));
    float *h_weights = (float *)aligned_alloc(64, w_size * sizeof(float));
    float *hidden_layer = (float *)aligned_alloc(64, hidden_size * sizeof(float));

    initialize_weights(h_weights, w_size);

    if (features < AVX_SIZE) {
        for (i = 0; i < instances; ++i) {
            read_instance(instance, features);
            avx_base = _mm512_setr_ps(instance[0], instance[1], instance[2], instance[3], instance[4], instance[5], instance[6], instance[7], 
                                      instance[0], instance[1], instance[2], instance[3], instance[4], instance[5], instance[6], instance[7]);
            for (j = 0; j < w_size; j += AVX_SIZE) {
                avx_weights = _mm512_load_ps(&h_weights[j]);
                avx_weights = _mm512_mul_ps(avx_base, avx_weights);
                for (k = 0; k < 2; ++k) {
                    hidden_layer[h_idx++] = _mm512_mask_reduce_add_ps(avx_mask[k], avx_weights);
                }
            }
        }
    } else {
        for (i = 0; i < instances; ++i) {
            read_instance(instance, features);
            for (j = 0; j < w_size; j += features) {
                for (k = 0; k < n_vectors; ++k) {
                    avx_base = _mm512_load_ps(&instance[k * AVX_SIZE]);
                    avx_weights = _mm512_load_ps(&h_weights[j + k * AVX_SIZE]);
                    avx_weights = _mm512_mul_ps(avx_base, avx_weights);
                    hidden_layer[h_idx] += _mm512_reduce_add_ps(avx_weights);
                }
                h_idx++;
            }
        }
    }

    for (i = 0; i < hidden_size; i += AVX_SIZE) {
        avx_phidden = _mm512_load_ps(&hidden_layer[i]);
        avx_phidden = _mm512_add_ps(avx_phidden, avx_bias);
	    _mm512_store_ps(&hidden_layer[i], avx_phidden);
        for (j = i; j < i + AVX_SIZE; ++j) {
            if (hidden_layer[j] < 0.0) {
            hidden_layer[j] = 0.0;
            }
        }
    }

    free(instance);
    free(h_weights);
    return hidden_layer;
}

float *softmax_layer(float *hidden_layer) {
    int i, j, k, o_idx = 0;
    int olayer_size = output_size * instances;
    int hlayer_size = features/2;
    int oweights_size = hlayer_size * output_size;
    if (oweights_size < AVX_SIZE) {
        oweights_size = AVX_SIZE;
    }
    int n_vectors = hlayer_size/AVX_SIZE;
    float *output_layer = (float *)aligned_alloc(64, olayer_size * sizeof(float));
    float *o_weights = (float *)aligned_alloc(64, oweights_size * sizeof(float));

    __m512 avx_hidden, avx_oweights, avx_output, avx_bias, avx_mul;
    __mmask16 avx_mask8[4] = {0xf000, 0xf00, 0xf0, 0xf};
    __mmask16 avx_mask16[2] = {0xff00, 0xff};
    avx_bias = _mm512_set1_ps((float)1.0);

    initialize_weights(o_weights, oweights_size);

    if (features == 8) {
        avx_oweights = _mm512_setr_ps(o_weights[0], o_weights[1], o_weights[2], o_weights[3], o_weights[0], o_weights[1], 
                                      o_weights[2], o_weights[3], o_weights[0], o_weights[1], o_weights[2], o_weights[3],
                                      o_weights[0], o_weights[1], o_weights[2], o_weights[3]);
        for (i = 0; i < hidden_size; i += features) {
            avx_hidden = _mm512_setr_ps(hidden_layer[i], hidden_layer[i+1], hidden_layer[i+2], hidden_layer[i+3], 
                                        hidden_layer[i], hidden_layer[i+1], hidden_layer[i+2], hidden_layer[i+3],
                                        hidden_layer[i+4], hidden_layer[i+5], hidden_layer[i+6], hidden_layer[i+7],
                                        hidden_layer[i+4], hidden_layer[i+5], hidden_layer[i+6], hidden_layer[i+7]);
            avx_mul = _mm512_mul_ps(avx_hidden, avx_oweights);
            for (k = 0; k < 4; ++k) {
                output_layer[o_idx++] = _mm512_mask_reduce_add_ps(avx_mask8[k], avx_mul);
            }
        }
    } else if (features == 16) {
        avx_oweights = _mm512_load_ps(o_weights);
        for (i = 0; i < hidden_size; i += hlayer_size) {
            avx_hidden = _mm512_setr_ps(hidden_layer[i], hidden_layer[i+1], hidden_layer[i+2], hidden_layer[i+3], 
                                        hidden_layer[i+4], hidden_layer[i+5], hidden_layer[i+6], hidden_layer[i+7], 
                                        hidden_layer[i], hidden_layer[i+1], hidden_layer[i+2], hidden_layer[i+3], 
                                        hidden_layer[i+4], hidden_layer[i+5], hidden_layer[i+6], hidden_layer[i+7]);
            avx_mul = _mm512_mul_ps(avx_hidden, avx_oweights);
            for (k = 0; k < 2; ++k) {
                output_layer[o_idx++] = _mm512_mask_reduce_add_ps(avx_mask16[k], avx_mul);
            }
        }
    } else {
        for (i = 0; i < hidden_size; i += hlayer_size) {
                for (k = 0; k < n_vectors; ++k) {
                    avx_hidden = _mm512_load_ps(&hidden_layer[i + k * AVX_SIZE]);
                    avx_oweights = _mm512_load_ps(&o_weights[k * AVX_SIZE]);
                    avx_oweights = _mm512_mul_ps(avx_hidden, avx_oweights);
                    output_layer[o_idx] += _mm512_reduce_add_ps(avx_oweights);
                }
		    o_idx++;
		    output_layer[o_idx++] = output_layer[o_idx - 1];
            }
    }

    for (i = 0; i < olayer_size; i += AVX_SIZE) {
        avx_output = _mm512_load_ps(&output_layer[i]);
        avx_output = _mm512_add_ps(avx_output, avx_bias);
        _mm512_store_ps(&output_layer[i], avx_output);
    }

    free(o_weights);
    return output_layer;
}

void classification(float *output_layer) {
    float *sum_exp = (float *)aligned_alloc(64, sizeof(float) * instances);
    float *result = (float *)aligned_alloc(64, sizeof(float) * instances * output_size);

    __m512 avx_sumexp, avx_output, avx_div;
    for (int i = 0; i < instances; i += AVX_SIZE) {
	    avx_sumexp = _mm512_load_ps(&sum_exp[i]);
        avx_sumexp = _mm512_setzero_ps();
	    _mm512_store_ps(&sum_exp[i], avx_sumexp);
    }

    for (int i = 0; i < instances; ++i) {
        for (int j = 0; j < output_size; ++j) {
            sum_exp[i] += expf(output_layer[i * output_size + j]);
        }
    }

    for (int i = 0, j = 0; i < instances; i += AVX_SIZE/2, j += AVX_SIZE) {
        avx_sumexp = _mm512_setr_ps(sum_exp[i], sum_exp[i], sum_exp[i+1], sum_exp[i+1], sum_exp[i+2], sum_exp[i+2], 
                                    sum_exp[i+3], sum_exp[i+3], sum_exp[i+4], sum_exp[i+4], sum_exp[i+5], sum_exp[i+5], 
                                    sum_exp[i+6], sum_exp[i+6], sum_exp[i+7], sum_exp[i+7]);
        avx_output = _mm512_load_ps(&output_layer[j]);
        avx_div = _mm512_div_ps(avx_output, avx_sumexp);
	    _mm512_store_ps(&result[j], avx_div);
    }

    for (int i = 0, j = 0; j < instances * output_size; ++i, j += output_size) {
        if (result[j] > result[j+1]) {
            printf("%d. %s\n", i, "neg");
        } else {
            printf("%d. %s\n", i, "pos");
        }
    }

    free(result);
}

int main(int argc, char const *argv[]) {
    instances = atoi(argv[1]);
    features = atoi(argv[2]);
    output_size = atoi(argv[3]);

    float *hidden_layer = relu_layer();

    float *output_layer = softmax_layer(hidden_layer);

    classification(output_layer);

    free(hidden_layer);
    free(output_layer);
    free(bias);
    return 0;
}
