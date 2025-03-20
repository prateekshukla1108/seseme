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

#define CUDA_CHECK(call) do { cudaError_t err = call; if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(EXIT_FAILURE); } } while (0)

#define CUBLAS_CHECK(call) do { cublasStatus_t status = call; if (status != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "CUBLAS error %d in %s at line %d\n", status, __FILE__, __LINE__); exit(EXIT_FAILURE); } } while (0)

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
    int* batch_indices = indices + b * vocab_size;
    float topk_sum = 0.0f;
    for (int k = 0; k < topk; k++) {
        int idx = batch_indices[k];
        topk_sum += probs[b * vocab_size + idx];
    }
    float u = curand_uniform(&state) * topk_sum;
    float cumsum = 0.0f;
    for (int k = 0; k < topk; k++) {
        int idx = batch_indices[k];
        float prob = probs[b * vocab_size + idx];
        cumsum += prob;
        if (cumsum > u) {
            sample_token[b] = idx;
            return;
        }
    }
    sample_token[b] = batch_indices[topk - 1];
}

__global__ void compute_topk_probs_kernel(float* logits, int topk, int vocab_size, int batch_size, float* probs) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    float* batch_logits = logits + b * vocab_size;
    float max_val = -INFINITY;
    for (int k = 0; k < vocab_size; ++k) {
        float val = batch_logits[k];
        if (val > max_val) max_val = val;
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < vocab_size; ++k) {
        float exp_val = expf(batch_logits[k] - max_val);
        probs[b * vocab_size + k] = exp_val;
        sum_exp += exp_val;
    }
    for (int k = 0; k < vocab_size; ++k) {
        probs[b * vocab_size + k] = probs[b * vocab_size + k] / sum_exp;
    }
}

void sample_topk_cuda(float* logits, int batch_size, float temperature, int topk, int vocab_size, int* sample_token) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    int* d_indices;
    float* d_probs;
    CUDA_CHECK(cudaMalloc(&d_indices, batch_size * vocab_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_probs, batch_size * vocab_size * sizeof(float)));
    thrust::device_ptr<int> indices_ptr(d_indices);
    thrust::sequence(thrust::device, indices_ptr, indices_ptr + batch_size * vocab_size);
    float alpha = 1.0f / temperature;
    float beta = 0.0f;
    // Scale the logits in place using float GEMM scaling (if needed, you can use cublasSscal)
    CUBLAS_CHECK(cublasSscal(handle, batch_size * vocab_size, &alpha, logits, 1));
    for (int b = 0; b < batch_size; ++b) {
        thrust::device_ptr<float> logits_ptr(logits + b * vocab_size);
        thrust::sort_by_key(thrust::device, logits_ptr, logits_ptr + vocab_size, indices_ptr + b * vocab_size, thrust::greater<float>());
    }
    int threads_per_block = 256;
    int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;
    compute_topk_probs_kernel<<<blocks_per_grid, threads_per_block>>>(logits, topk, vocab_size, batch_size, d_probs);
    CUDA_CHECK(cudaGetLastError());
    sample_kernel<<<blocks_per_grid, threads_per_block>>>(logits, d_indices, d_probs, topk, vocab_size, batch_size, sample_token, static_cast<unsigned long long>(time(NULL)));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_probs));
    CUBLAS_CHECK(cublasDestroy(handle));
}

} // namespace cuda_utils

namespace cuda_nn {

__global__ void embedding_forward_kernel(const int* input, int batch_size, int seq_len, int embedding_dim, const float* weight, float* output) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size && s < seq_len) {
        int token = input[b * seq_len + s];
        float* out_ptr = output + (b * seq_len + s) * embedding_dim;
        const float* weight_ptr = weight + token * embedding_dim;
        for (int d = 0; d < embedding_dim; ++d) {
            out_ptr[d] = weight_ptr[d];
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
    if (idx < total) {
        out[idx] = out[idx] + bias[idx % out_features];
    }
}

void linear_forward_cuda(const float* input, int batch_size, int seq_len, int in_features, int out_features, const float* weight, const float* bias, float* output) {
    if (batch_size <= 0 || seq_len <= 0 || in_features <= 0 || out_features <= 0) {
        fprintf(stderr, "Invalid dimensions: batch_size=%d, seq_len=%d, in_features=%d, out_features=%d\n", batch_size, seq_len, in_features, out_features);
        exit(EXIT_FAILURE);
    }
    if (!input || !weight || !bias || !output) {
        fprintf(stderr, "Null pointer detected in linear_forward_cuda\n");
        exit(EXIT_FAILURE);
    }
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    float alpha = 1.0f;
    float beta = 0.0f;
    int m = out_features;           // Rows of output
    int n = batch_size * seq_len;   // Columns of output
    int k = in_features;            // Shared dimension

    // Revised GEMM call: use CUDA_R_32F.
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                              m, n, k,
                              &alpha, weight, CUDA_R_32F, m,
                              input, CUDA_R_32F, k,
                              &beta, output, CUDA_R_32F, m,
                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

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
    for (int j = 0; j <= i; ++j) {
        if (mask[b * max_seq_len + j]) {
            float val = score_row[j];
            if(val > max_val) max_val = val;
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
        score_row[j] = score_row[j] / sum_exp;
    }
}

void attention_forward_cuda(
    const float* Q, const float* K, const float* V, const bool* mask,
    int batch_size, int seq_len, int max_seq_len, int num_heads, int head_dim,
    float* output
) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    int total_batch = batch_size * num_heads;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    float alpha = scale, beta = 0.0f;
    float* scores;
    CUDA_CHECK(cudaMalloc(&scores, total_batch * seq_len * seq_len * sizeof(float)));
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_T,
        seq_len, seq_len, head_dim,
        &alpha, Q, CUDA_R_32F, head_dim, seq_len * head_dim,
        K, CUDA_R_32F, head_dim, seq_len * head_dim,
        &beta, scores, CUDA_R_32F, seq_len, seq_len * seq_len,
        total_batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    int threads_per_block = 256;
    int blocks_per_grid = (total_batch * seq_len + threads_per_block - 1) / threads_per_block;
    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(scores, mask, batch_size, num_heads, seq_len, max_seq_len);
    CUDA_CHECK(cudaGetLastError());
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        head_dim, seq_len, seq_len,
        &alpha, V, CUDA_R_32F, head_dim, seq_len * head_dim,
        scores, CUDA_R_32F, seq_len, seq_len * seq_len,
        &beta, output, CUDA_R_32F, head_dim, seq_len * head_dim,
        total_batch, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
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
        float gelu = 0.5f * val * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
        x[i] = gelu;
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
    float eps = 1e-5f;
    float inv_std = 1.0f / sqrtf(variance + eps);
    for (int i = tid; i < cols; i += blockDim.x) {
        out_row[i] = gamma[i] * ((row_data[i] - mean) * inv_std) + beta[i];
    }
}

void layer_norm_cuda(float* input, const float* gamma, const float* beta, int rows, int cols, float* output) {
    int threads_per_block = 256;
    dim3 grid_dim(rows);
    size_t shared_mem_size = threads_per_block * sizeof(float);
    layer_norm_kernel<<<grid_dim, threads_per_block, shared_mem_size>>>(input, gamma, beta, rows, cols, output);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void reshape_kernel(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * head_dim;
    if (idx >= total) return;
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int nh = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);
    int input_idx = (b * seq_len + s) * (num_heads * head_dim) + nh * head_dim + d;
    output[idx] = input[input_idx];
}

void reshape(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int total = batch_size * num_heads * seq_len * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_kernel<<<blocks, threads>>>(input, output, batch_size, seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void transformer_decoder_forward_cuda(
    float* input, int batch_size, int seq_len, int embedding_dim, int num_heads, int intermediate_dim,
    const float* q_weight, const float* q_bias,
    const float* k_weight, const float* k_bias,
    const float* v_weight, const float* v_bias,
    const float* out_weight, const float* out_bias,
    const float* mlp_w1_weight, const float* mlp_w1_bias,
    const float* mlp_w2_weight, const float* mlp_w2_bias,
    const float* sa_norm_weight, const float* sa_norm_bias,
    const float* mlp_norm_weight, const float* mlp_norm_bias,
    const bool* mask, float* output
) {
    int head_dim = embedding_dim / num_heads;
    size_t hidden_size = batch_size * seq_len * embedding_dim * sizeof(float);
    float *current, *attn_out, *mlp_out, *temp;
    CUDA_CHECK(cudaMalloc(&current, hidden_size));
    CUDA_CHECK(cudaMalloc(&attn_out, hidden_size));
    CUDA_CHECK(cudaMalloc(&mlp_out, hidden_size));
    CUDA_CHECK(cudaMalloc(&temp, hidden_size));
    CUDA_CHECK(cudaMemcpy(current, input, hidden_size, cudaMemcpyDeviceToDevice));
    float *Q, *K, *V;
    CUDA_CHECK(cudaMalloc(&Q, hidden_size));
    CUDA_CHECK(cudaMalloc(&K, hidden_size));
    CUDA_CHECK(cudaMalloc(&V, hidden_size));
    linear_forward_cuda(current, batch_size, seq_len, embedding_dim, embedding_dim, q_weight, q_bias, Q);
    linear_forward_cuda(current, batch_size, seq_len, embedding_dim, embedding_dim, k_weight, k_bias, K);
    linear_forward_cuda(current, batch_size, seq_len, embedding_dim, embedding_dim, v_weight, v_bias, V);
    float *Q_reshaped, *K_reshaped, *V_reshaped;
    size_t reshape_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
    CUDA_CHECK(cudaMalloc(&Q_reshaped, reshape_size));
    CUDA_CHECK(cudaMalloc(&K_reshaped, reshape_size));
    CUDA_CHECK(cudaMalloc(&V_reshaped, reshape_size));
    reshape(Q, Q_reshaped, batch_size, seq_len, num_heads, head_dim);
    reshape(K, K_reshaped, batch_size, seq_len, num_heads, head_dim);
    reshape(V, V_reshaped, batch_size, seq_len, num_heads, head_dim);
    attention_forward_cuda(Q_reshaped, K_reshaped, V_reshaped, mask, batch_size, seq_len, seq_len, num_heads, head_dim, attn_out);
    CUDA_CHECK(cudaFree(Q));
    CUDA_CHECK(cudaFree(K));
    CUDA_CHECK(cudaFree(V));
    CUDA_CHECK(cudaFree(Q_reshaped));
    CUDA_CHECK(cudaFree(K_reshaped));
    CUDA_CHECK(cudaFree(V_reshaped));
    linear_forward_cuda(attn_out, batch_size, seq_len, embedding_dim, embedding_dim, out_weight, out_bias, temp);
    int total_elements = batch_size * seq_len * embedding_dim;
    int threads_per_block = 256;
    int blocks_per_grid = (total_elements + threads_per_block - 1) / threads_per_block;
    elementwise_add<<<blocks_per_grid, threads_per_block>>>(current, temp, total_elements, temp);
    layer_norm_cuda(temp, sa_norm_weight, sa_norm_bias, batch_size * seq_len, embedding_dim, current);
    linear_forward_cuda(current, batch_size, seq_len, embedding_dim, intermediate_dim, mlp_w1_weight, mlp_w1_bias, temp);
    gelu_cuda(temp, batch_size * seq_len * intermediate_dim);
    linear_forward_cuda(temp, batch_size, seq_len, intermediate_dim, embedding_dim, mlp_w2_weight, mlp_w2_bias, mlp_out);
    elementwise_add<<<blocks_per_grid, threads_per_block>>>(current, mlp_out, total_elements, temp);
    layer_norm_cuda(temp, mlp_norm_weight, mlp_norm_bias, batch_size * seq_len, embedding_dim, current);
    CUDA_CHECK(cudaMemcpy(output, current, hidden_size, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(current));
    CUDA_CHECK(cudaFree(attn_out));
    CUDA_CHECK(cudaFree(mlp_out));
    CUDA_CHECK(cudaFree(temp));
}

} // namespace cuda_nn

extern "C" {
    void create_causal_mask_cuda(int seq_len, bool* mask) {
        cuda_utils::create_causal_mask_cuda(seq_len, mask);
    }
    void index_causal_mask_cuda(const int* input_pos, int batch_size, int seq_len, int max_seq_len, bool* result_mask) {
        cuda_utils::index_causal_mask_cuda(input_pos, batch_size, seq_len, max_seq_len, result_mask);
    }
    void sample_topk_cuda(float* logits, int batch_size, float temperature, int topk, int vocab_size, int* sample_token) {
        cuda_utils::sample_topk_cuda(logits, batch_size, temperature, topk, vocab_size, sample_token);
    }
    void embedding_forward_cuda(const int* input, int batch_size, int seq_len, int vocab_size, int embedding_dim, const float* weight, float* output) {
        cuda_nn::embedding_forward_cuda(input, batch_size, seq_len, vocab_size, embedding_dim, weight, output);
    }
    void linear_forward_cuda(const float* input, int batch_size, int seq_len, int in_features, int out_features, const float* weight, const float* bias, float* output) {
        cuda_nn::linear_forward_cuda(input, batch_size, seq_len, in_features, out_features, weight, bias, output);
    }
    void transformer_decoder_forward_cuda(
        float* input, int batch_size, int seq_len, int embedding_dim, int num_heads, int intermediate_dim,
        const float* q_weights, const float* q_biases,
        const float* k_weights, const float* k_biases,
        const float* v_weights, const float* v_biases,
        const float* out_weights, const float* out_biases,
        const float* mlp_w1_weights, const float* mlp_w1_biases,
        const float* mlp_w2_weights, const float* mlp_w2_biases,
        const float* sa_norm_weights, const float* sa_norm_biases,
        const float* mlp_norm_weights, const float* mlp_norm_biases,
        const bool* mask, float* output
    ) {
        cuda_nn::transformer_decoder_forward_cuda(input, batch_size, seq_len, embedding_dim, num_heads, intermediate_dim,
            q_weights, q_biases, k_weights, k_biases, v_weights, v_biases,
            out_weights, out_biases, mlp_w1_weights, mlp_w1_biases,
            mlp_w2_weights, mlp_w2_biases, sa_norm_weights, sa_norm_biases,
            mlp_norm_weights, mlp_norm_biases, mask, output);
    }
}

