#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <time.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Error checking macro for CUDA API calls
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Error checking macro for cuBLAS API calls
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error %d in %s at line %d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


//-----------------------------------------------------------------------------
// Utility Functions (cuda_utils namespace)
//-----------------------------------------------------------------------------

namespace cuda_utils {

// Create a causal mask kernel
__global__ void create_causal_mask_kernel(int seq_len, bool* mask) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq_len && j < seq_len) {
        mask[i * seq_len + j] = (i >= j);
    }
}

void create_causal_mask_cuda(int seq_len, bool* mask) {
    dim3 block_dim(32, 32);
    dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x, (seq_len + block_dim.y - 1) / block_dim.y);
    create_causal_mask_kernel<<<grid_dim, block_dim>>>(seq_len, mask);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void index_causal_mask_kernel(const int* input_pos, int batch_size, int max_seq_len, bool* result_mask) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && j < max_seq_len) {
        result_mask[b * max_seq_len + j] = (j <= input_pos[b]);
    }
}

void index_causal_mask_cuda(const int* input_pos, int batch_size, int seq_len, int max_seq_len, bool* result_mask) {
    dim3 block_dim(32, 32);
    dim3 grid_dim((max_seq_len + block_dim.x - 1) / block_dim.x, (batch_size + block_dim.y - 1) / block_dim.y);
    index_causal_mask_kernel<<<grid_dim, block_dim>>>(input_pos, batch_size, max_seq_len, result_mask);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void sample_topk_kernel(float* logits, int topk, float temperature, int vocab_size, int batch_size, int* sample_token, unsigned long long seed) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        curandState state;
        curand_init(seed, b, 0, &state);
        float* batch_logits = logits + b * vocab_size;

        // Apply temperature scaling
        for (int i = 0; i < vocab_size; ++i) {
            batch_logits[i] /= temperature;
        }

        // NOTE: The following uses Thrust on device. Using Thrust functions inside a __global__
        // kernel is generally not supported. You should consider refactoring this code to perform
        // top-k filtering on the host or implement a device-side solution.
        thrust::device_ptr<float> logits_ptr(batch_logits);
        thrust::device_ptr<int> indices_ptr = thrust::device_pointer_cast(sample_token + b * vocab_size);
        thrust::sequence(thrust::device, indices_ptr, indices_ptr + vocab_size);
        thrust::sort_by_key(thrust::device, logits_ptr, logits_ptr + vocab_size, indices_ptr, thrust::greater<float>());
        
        // Softmax over top-k
        float max_logit = batch_logits[indices_ptr[0]];
        float sum_exp = 0.0f;
        for (int i = 0; i < topk; ++i) {
            sum_exp += expf(batch_logits[indices_ptr[i]] - max_logit);
        }
        float u = curand_uniform(&state);
        float cumsum = 0.0f;
        int k = 0;
        for (; k < topk; ++k) {
            float prob = expf(batch_logits[indices_ptr[k]] - max_logit) / sum_exp;
            cumsum += prob;
            if (cumsum > u) break;
        }
        sample_token[b] = indices_ptr[k < topk ? k : topk - 1];
    }
}

void sample_topk_cuda(float* logits, int topk, float temperature, int vocab_size, int batch_size, int* sample_token) {
    unsigned long long seed = static_cast<unsigned long long>(time(NULL));
    int threads_per_block = 256;
    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
    sample_topk_kernel<<<blocks_per_grid, threads_per_block>>>(logits, topk, temperature, vocab_size, batch_size, sample_token, seed);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda_utils


//-----------------------------------------------------------------------------
// Neural Network Operations (cuda_nn namespace)
//-----------------------------------------------------------------------------

namespace cuda_nn {

__global__ void embedding_forward_kernel(const int* input, int batch_size, int seq_len, int embedding_dim, const float* weight, float* output) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && s < seq_len) {
        int token = input[b * seq_len + s];
        for (int d = 0; d < embedding_dim; ++d) {
            output[(b * seq_len + s) * embedding_dim + d] = weight[token * embedding_dim + d];
        }
    }
}

void embedding_forward_cuda(const int* input, int batch_size, int seq_len, int vocab_size, int embedding_dim, const float* weight, float* output) {
    dim3 block_dim(32, 8);
    dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x, (batch_size + block_dim.y - 1) / block_dim.y);
    embedding_forward_kernel<<<grid_dim, block_dim>>>(input, batch_size, seq_len, embedding_dim, weight, output);
    CUDA_CHECK(cudaGetLastError());
}

// The add_bias kernel is defined at file scope (moved outside of any function)
__global__ void add_bias(float* out, const float* bias, int out_features, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        out[idx] += bias[idx % out_features];
}

void linear_forward_cuda(const float* input, int batch_size, int seq_len, int in_features, int out_features, const float* weight, const float* bias, float* output) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;
    int m = out_features;
    int n = batch_size * seq_len;
    int k = in_features;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, weight, m, input, k, &beta, output, m));
    CUBLAS_CHECK(cublasDestroy(handle));

    int total = batch_size * seq_len * out_features;
    int threads_per_block = 256;
    int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;
    add_bias<<<blocks_per_grid, threads_per_block>>>(output, bias, out_features, total);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void attention_forward_kernel(
    const float* Q, const float* K, const float* V, const bool* mask,
    int batch_size, int seq_len, int max_seq_len, int num_heads, int head_dim,
    float* output
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && h < num_heads && i < seq_len) {
        float scale = 1.0f / sqrtf((float)head_dim);
        float scores[2048]; // Assumes max_seq_len <= 2048
        float max_score = -INFINITY;
        
        int q_offset = (b * num_heads + h) * seq_len * head_dim + i * head_dim;
        for (int j = 0; j < max_seq_len; ++j) {
            if (mask[b * seq_len * max_seq_len + i * max_seq_len + j]) {
                int k_offset = (b * num_heads + h) * max_seq_len * head_dim + j * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += Q[q_offset + d] * K[k_offset + d];
                }
                score *= scale;
                scores[j] = score;
                max_score = fmaxf(max_score, score);
            } else {
                scores[j] = -INFINITY;
            }
        }
        
        float sum_exp = 0.0f;
        for (int j = 0; j < max_seq_len; ++j) {
            if (scores[j] != -INFINITY) {
                scores[j] = expf(scores[j] - max_score);
                sum_exp += scores[j];
            }
        }
        
        int out_offset = (b * num_heads + h) * seq_len * head_dim + i * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            float value = 0.0f;
            for (int j = 0; j < max_seq_len; ++j) {
                if (scores[j] != -INFINITY) {
                    int v_offset = (b * num_heads + h) * max_seq_len * head_dim + j * head_dim + d;
                    value += (scores[j] / sum_exp) * V[v_offset];
                }
            }
            output[out_offset + d] = value;
        }
    }
}

void attention_forward_cuda(
    const float* Q, const float* K, const float* V, const bool* mask,
    int batch_size, int seq_len, int max_seq_len, int num_heads, int head_dim,
    float* output
) {
    dim3 block_dim(32, 1, 1);
    dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x, num_heads, batch_size);
    attention_forward_kernel<<<grid_dim, block_dim>>>(Q, K, V, mask, batch_size, seq_len, max_seq_len, num_heads, head_dim, output);
    CUDA_CHECK(cudaGetLastError());
}

void transformer_decoder_forward_cuda(
    const float* input, int batch_size, int seq_len, int num_layers, int num_heads, int head_dim,
    const float** qkv_weights, const float** qkv_biases, const float** out_weights, const float** out_biases,
    const bool* mask, float* output
) {
    float* current = nullptr;
    size_t current_size = batch_size * seq_len * num_heads * head_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&current, current_size));
    CUDA_CHECK(cudaMemcpy(current, input, current_size, cudaMemcpyDeviceToDevice));
    
    for (int l = 0; l < num_layers; ++l) {
        float* qkv_out = nullptr;
        size_t qkv_size = batch_size * seq_len * num_heads * head_dim * 3 * sizeof(float);
        CUDA_CHECK(cudaMalloc(&qkv_out, qkv_size));
        linear_forward_cuda(current, batch_size, seq_len, num_heads * head_dim, num_heads * head_dim * 3,
                           qkv_weights[l], qkv_biases[l], qkv_out);
        
        float* Q = qkv_out;
        float* K = qkv_out + batch_size * seq_len * num_heads * head_dim;
        float* V = qkv_out + 2 * batch_size * seq_len * num_heads * head_dim;
        
        float* attn_out = nullptr;
        size_t attn_size = batch_size * seq_len * num_heads * head_dim * sizeof(float);
        CUDA_CHECK(cudaMalloc(&attn_out, attn_size));
        attention_forward_cuda(Q, K, K, mask, batch_size, seq_len, seq_len, num_heads, head_dim, attn_out);
        
        float* layer_out = nullptr;
        CUDA_CHECK(cudaMalloc(&layer_out, attn_size));
        linear_forward_cuda(attn_out, batch_size, seq_len, num_heads * head_dim, num_heads * head_dim,
                           out_weights[l], out_biases[l], layer_out);
        
        CUDA_CHECK(cudaFree(qkv_out));
        CUDA_CHECK(cudaFree(attn_out));
        CUDA_CHECK(cudaFree(current));
        current = layer_out;
    }
    
    size_t output_size = batch_size * seq_len * num_heads * head_dim * sizeof(float);
    CUDA_CHECK(cudaMemcpy(output, current, output_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(current));
}

} // namespace cuda_nn


//-----------------------------------------------------------------------------
// External C Interface
//-----------------------------------------------------------------------------

extern "C" {

    void create_causal_mask_cuda(int seq_len, bool* mask) {
        cuda_utils::create_causal_mask_cuda(seq_len, mask);
    }
    
    void index_causal_mask_cuda(const int* input_pos, int batch_size, int seq_len, int max_seq_len, bool* result_mask) {
        cuda_utils::index_causal_mask_cuda(input_pos, batch_size, seq_len, max_seq_len, result_mask);
    }
    
    void sample_topk_cuda(float* logits, int topk, float temperature, int vocab_size, int batch_size, int* sample_token) {
        cuda_utils::sample_topk_cuda(logits, topk, temperature, vocab_size, batch_size, sample_token);
    }
    
    void embedding_forward_cuda(const int* input, int batch_size, int seq_len, int vocab_size, int embedding_dim, const float* weight, float* output) {
        cuda_nn::embedding_forward_cuda(input, batch_size, seq_len, vocab_size, embedding_dim, weight, output);
    }
    
    void linear_forward_cuda(const float* input, int batch_size, int seq_len, int in_features, int out_features, const float* weight, const float* bias, float* output) {
        cuda_nn::linear_forward_cuda(input, batch_size, seq_len, in_features, out_features, weight, bias, output);
    }
    
    void transformer_decoder_forward_cuda(
        const float* input, int batch_size, int seq_len, int num_layers, int num_heads, int head_dim,
        const float** qkv_weights, const float** qkv_biases, const float** out_weights, const float** out_biases,
        const bool* mask, float* output
    ) {
        cuda_nn::transformer_decoder_forward_cuda(input, batch_size, seq_len, num_layers, num_heads, head_dim,
                                                    qkv_weights, qkv_biases, out_weights, out_biases, mask, output);
    }
}

