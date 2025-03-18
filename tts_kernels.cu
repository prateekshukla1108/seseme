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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS error %d in %s at line %d\n", status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace cuda_utils {

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

__global__ void sample_kernel(float* logits, int* indices, float* probs, int topk, int vocab_size, int batch_size, int* sample_token, unsigned long long seed) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    curandState state;
    curand_init(seed, b, 0, &state);
    float* batch_probs = probs + b * topk;
    int* batch_indices = indices + b * vocab_size;
    float cumsum = 0.0f;
    float u = curand_uniform(&state);
    int k;
    for (k = 0; k < topk; k++) {
        cumsum += batch_probs[k];
        if (cumsum > u) break;
    }
    sample_token[b] = batch_indices[k < topk ? k : topk - 1];
}

void sample_topk_cuda(float* logits, int topk, float temperature, int vocab_size, int batch_size, int* sample_token) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int* d_indices;
    float* d_probs;
    CUDA_CHECK(cudaMalloc(&d_indices, batch_size * vocab_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * topk * sizeof(float)));

    thrust::device_ptr<int> indices_ptr(d_indices);
    thrust::sequence(thrust::device, indices_ptr, indices_ptr + batch_size * vocab_size);

    float alpha = 1.0f / temperature;
    CUBLAS_CHECK(cublasSscal(handle, batch_size * vocab_size, &alpha, logits, 1));

    for (int b = 0; b < batch_size; ++b) {
        thrust::device_ptr<float> logits_ptr(logits + b * vocab_size);
        thrust::sort_by_key(thrust::device, logits_ptr, logits_ptr + vocab_size, indices_ptr + b * vocab_size, thrust::greater<float>());
    }

    unsigned long long seed = static_cast<unsigned long long>(time(NULL));
    int threads_per_block = 256;
    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
    sample_kernel<<<blocks_per_grid, threads_per_block>>>(logits, d_indices, d_probs, topk, vocab_size, batch_size, sample_token, seed);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_probs));
    CUBLAS_CHECK(cublasDestroy(handle));
}

}  // namespace cuda_utils

namespace cuda_nn {

__global__ void embedding_forward_kernel(const int* input, int batch_size, int seq_len, int embedding_dim, const float* weight, float* output) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && s < seq_len) {
        int token = input[b * seq_len + s];
        float* out_ptr = output + (b * seq_len + s) * embedding_dim;
        const float* weight_ptr = weight + token * embedding_dim;
        for (int d = 0; d < embedding_dim; d += 4) {
            if (d + 4 <= embedding_dim) {
                float4 emb = *reinterpret_cast<const float4*>(weight_ptr + d);
                *reinterpret_cast<float4*>(out_ptr + d) = emb;
            } else {
                for (int i = d; i < embedding_dim; ++i) {
                    out_ptr[i] = weight_ptr[i];
                }
            }
        }
    }
}

void embedding_forward_cuda(const int* input, int batch_size, int seq_len, int vocab_size, int embedding_dim, const float* weight, float* output) {
    dim3 block_dim(32, 8);
    dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x, (batch_size + block_dim.y - 1) / block_dim.y);
    embedding_forward_kernel<<<grid_dim, block_dim>>>(input, batch_size, seq_len, embedding_dim, weight, output);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void add_bias(float* out, const float* bias, int out_features, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total)
        out[idx] += bias[idx % out_features];
}

void linear_forward_cuda(const float* input, int batch_size, int seq_len, int in_features, int out_features, const float* weight, const float* bias, float* output) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    int m = out_features;
    int n = batch_size * seq_len;
    int k = in_features;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, weight, m, input, k, &beta, output, m));

    int total = batch_size * seq_len * out_features;
    int threads_per_block = 256;
    int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;
    add_bias<<<blocks_per_grid, threads_per_block>>>(output, bias, out_features, total);
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasDestroy(handle));
}

__global__ void softmax_kernel(float* scores, const bool* mask, int batch_size, int num_heads, int seq_len, int max_seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len;
    if (idx >= total) return;
    int b = idx / (num_heads * seq_len);
    int h = (idx % (num_heads * seq_len)) / seq_len;
    int i = idx % seq_len;
    float* score_row = scores + (b * num_heads + h) * seq_len * seq_len + i * seq_len;

    float max_val = -INFINITY;
    for (int j = 0; j <= i; ++j) {  // Causal mask: j <= i
        if (mask[b * max_seq_len + j]) {
            max_val = fmaxf(max_val, score_row[j]);
        }
    }

    float sum_exp = 0.0f;
    for (int j = 0; j <= i; ++j) {
        if (mask[b * max_seq_len + j]) {
            score_row[j] = expf(score_row[j] - max_val);
            sum_exp += score_row[j];
        } else {
            score_row[j] = 0.0f;
        }
    }

    for (int j = 0; j < seq_len; ++j) {
        score_row[j] /= sum_exp;
    }
}

void attention_forward_cuda(
    const float* Q, const float* K, const float* V, const bool* mask,
    int batch_size, int seq_len, int max_seq_len, int num_heads, int head_dim,
    float* output
) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int total_batch = batch_size * num_heads;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    float alpha = scale, beta = 0.0f;

    float* scores;
    CUDA_CHECK(cudaMalloc(&scores, total_batch * seq_len * seq_len * sizeof(float)));

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        seq_len, seq_len, head_dim,
        &alpha, K, head_dim, seq_len * head_dim,
        Q, head_dim, seq_len * head_dim,
        &beta, scores, seq_len, seq_len * seq_len,
        total_batch
    ));

    int threads_per_block = 256;
    int blocks_per_grid = (total_batch * seq_len + threads_per_block - 1) / threads_per_block;
    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(scores, mask, batch_size, num_heads, seq_len, max_seq_len);
    CUDA_CHECK(cudaGetLastError());

    CUBLAS_CHECK(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, seq_len, seq_len,
        &alpha, V, head_dim, seq_len * head_dim,
        scores, seq_len, seq_len * seq_len,
        &beta, output, head_dim, seq_len * head_dim,
        total_batch
    ));

    CUDA_CHECK(cudaFree(scores));
    CUBLAS_CHECK(cublasDestroy(handle));
}

__global__ void elementwise_add(const float* a, const float* b, int size, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = a[i] + b[i];
    }
}

__global__ void gelu_kernel(float* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = x[i];
        x[i] = 0.5 * val * (1.0 + tanhf(sqrtf(2.0 / M_PI) * (val + 0.044715 * val * val * val)));
    }
}

void gelu_cuda(float* x, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    gelu_kernel<<<blocks_per_grid, threads_per_block>>>(x, size);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void layer_norm_kernel(float* input, const float* gamma, const float* beta, int rows, int cols, float* output) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    float* row_data = input + row * cols;
    float* out_row = output + row * cols;

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += row_data[i];
    }
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float mean = shared[0] / cols;

    sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_data[i] - mean;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    float variance = shared[0] / cols;

    for (int i = tid; i < cols; i += blockDim.x) {
        out_row[i] = gamma[i] * (row_data[i] - mean) / sqrtf(variance + 1e-5f) + beta[i];
    }
}

void layer_norm_cuda(float* input, const float* gamma, const float* beta, int rows, int cols, float* output) {
    int threads_per_block = 256;
    dim3 grid_dim(rows);
    size_t shared_mem_size = threads_per_block * sizeof(float);
    layer_norm_kernel<<<grid_dim, threads_per_block, shared_mem_size>>>(input, gamma, beta, rows, cols, output);
    CUDA_CHECK(cudaGetLastError());
}

void transformer_decoder_forward_cuda(
    const float* input, int batch_size, int seq_len, int num_layers, int num_heads, int head_dim,
    const float** q_weights, const float** k_weights, const float** v_weights, const float** out_weights,
    const float** q_biases, const float** k_biases, const float** v_biases, const float** out_biases,
    const float** mlp_0_weights, const float** mlp_2_weights, const float** mlp_0_biases, const float** mlp_2_biases,
    const float** sa_norm_weights, const float** mlp_norm_weights, const float** sa_norm_biases, const float** mlp_norm_biases,
    const float* norm_weights, const float* norm_biases,
    const bool* mask, float* output
) {
    size_t hidden_size = batch_size * seq_len * num_heads * head_dim * sizeof(float);
    size_t qkv_size = hidden_size * 3;

    float *current, *qkv_out, *attn_out, *mlp_out, *temp;
    CUDA_CHECK(cudaMalloc(&current, hidden_size));
    CUDA_CHECK(cudaMalloc(&qkv_out, qkv_size));
    CUDA_CHECK(cudaMalloc(&attn_out, hidden_size));
    CUDA_CHECK(cudaMalloc(&mlp_out, hidden_size));
    CUDA_CHECK(cudaMalloc(&temp, hidden_size));

    CUDA_CHECK(cudaMemcpy(current, input, hidden_size, cudaMemcpyDeviceToDevice));

    for (int l = 0; l < num_layers; ++l) {
        linear_forward_cuda(current, batch_size, seq_len, num_heads * head_dim, num_heads * head_dim * 3,
                            q_weights[l], q_biases[l], qkv_out);

        float* Q = qkv_out;
        float* K = qkv_out + batch_size * seq_len * num_heads * head_dim;
        float* V = qkv_out + 2 * batch_size * seq_len * num_heads * head_dim;

        attention_forward_cuda(Q, K, V, mask, batch_size, seq_len, seq_len, num_heads, head_dim, attn_out);

        linear_forward_cuda(attn_out, batch_size, seq_len, num_heads * head_dim, num_heads * head_dim,
                            out_weights[l], out_biases[l], temp);

        int total_elements = batch_size * seq_len * num_heads * head_dim;
        int threads_per_block = 256;
        int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
        elementwise_add<<<blocks_per_grid, threads_per_block>>>(current, temp, total_elements, temp);

        layer_norm_cuda(temp, sa_norm_weights[l], sa_norm_biases[l], batch_size * seq_len, num_heads * head_dim, current);

        linear_forward_cuda(current, batch_size, seq_len, num_heads * head_dim, num_heads * head_dim * 4,
                            mlp_0_weights[l], mlp_0_biases[l], temp);
        gelu_cuda(temp, batch_size * seq_len * num_heads * head_dim * 4);

        linear_forward_cuda(temp, batch_size, seq_len, num_heads * head_dim * 4, num_heads * head_dim,
                            mlp_2_weights[l], mlp_2_biases[l], mlp_out);

        elementwise_add<<<blocks_per_grid, threads_per_block>>>(current, mlp_out, total_elements, temp);
        layer_norm_cuda(temp, mlp_norm_weights[l], mlp_norm_biases[l], batch_size * seq_len, num_heads * head_dim, current);
    }

    layer_norm_cuda(current, norm_weights, norm_biases, batch_size * seq_len, num_heads * head_dim, output);

    CUDA_CHECK(cudaFree(current));
    CUDA_CHECK(cudaFree(qkv_out));
    CUDA_CHECK(cudaFree(attn_out));
    CUDA_CHECK(cudaFree(mlp_out));
    CUDA_CHECK(cudaFree(temp));
}

}  // namespace cuda_nn

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
        const float** q_weights, const float** k_weights, const float** v_weights, const float** out_weights,
        const float** q_biases, const float** k_biases, const float** v_biases, const float** out_biases,
        const float** mlp_0_weights, const float** mlp_2_weights, const float** mlp_0_biases, const float** mlp_2_biases,
        const float** sa_norm_weights, const float** mlp_norm_weights, const float** sa_norm_biases, const float** mlp_norm_biases,
        const float* norm_weights, const float* norm_biases,
        const bool* mask, float* output
    ) {
        cuda_nn::transformer_decoder_forward_cuda(input, batch_size, seq_len, num_layers, num_heads, head_dim,
                                                  q_weights, k_weights, v_weights, out_weights,
                                                  q_biases, k_biases, v_biases, out_biases,
                                                  mlp_0_weights, mlp_2_weights, mlp_0_biases, mlp_2_biases,
                                                  sa_norm_weights, mlp_norm_weights, sa_norm_biases, mlp_norm_biases,
                                                  norm_weights, norm_biases, mask, output);
    }
} // extern "C"
