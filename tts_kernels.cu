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
#include <iostream>

// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << status \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

namespace cuda_utils {

    // --- Utility Functions ---

    __global__ void create_causal_mask_kernel(int seq_len, bool* mask) {
        int i = blockIdx.y * blockDim.y + threadIdx.y;
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < seq_len && j < seq_len) {
            mask[i * seq_len + j] = (i >= j);
        }
    }

    void create_causal_mask_cuda(int seq_len, bool* mask) {
        dim3 block_dim(32, 32);
        dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x,
                      (seq_len + block_dim.y - 1) / block_dim.y);
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
        dim3 grid_dim((max_seq_len + block_dim.x - 1) / block_dim.x,
                      (batch_size + block_dim.y - 1) / block_dim.y);
        index_causal_mask_kernel<<<grid_dim, block_dim>>>(input_pos, batch_size, max_seq_len, result_mask);
        CUDA_CHECK(cudaGetLastError());
    }

    __global__ void compute_topk_probs_kernel(float* logits, int* indices, int topk, int vocab_size, int batch_size, float* probs) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= batch_size) return;
        int* batch_indices = indices + b * vocab_size;
        float max_val = -INFINITY;
        for (int k = 0; k < topk; k++) {
            int idx = batch_indices[k];
            float val = logits[b * vocab_size + idx];
            if (val > max_val) max_val = val;
        }
        float sum_exp = 0.0f;
        for (int k = 0; k < topk; k++) {
            int idx = batch_indices[k];
            float exp_val = expf(logits[b * vocab_size + idx] - max_val);
            probs[b * topk + k] = exp_val;
            sum_exp += exp_val;
        }
        for (int k = 0; k < topk; k++) {
            probs[b * topk + k] /= sum_exp;
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

    __global__ void add_kernel(const float* a, const float* b, float* c, int num_elements) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_elements) {
            c[idx] = a[idx] + b[idx];
        }
    }
    
} // namespace cuda_utils

namespace cuda_nn {

    // --- Neural Network Functions ---

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
        dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x,
                      (batch_size + block_dim.y - 1) / block_dim.y);
        embedding_forward_kernel<<<grid_dim, block_dim>>>(input, batch_size, seq_len, embedding_dim, weight, output);
        CUDA_CHECK(cudaGetLastError());
    }

    __global__ void add_bias_kernel(float* output, const float* bias, int m, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < m * n) {
            int col = idx % n;
            output[idx] += bias[col];
        }
    }

    void linear_forward_cuda(const float* input, int batch_size, int seq_len, int in_features, int out_features,
                             const float* weight, const float* bias, float* output) {
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        float alpha = 1.0f;
        float beta = 0.0f;
        int m = batch_size * seq_len;
        int n = out_features;
        int k = in_features;
        // output = input @ weight^T (row-major)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, input, m, weight, k, &beta, output, m));
        int total = m * n;
        int threads_per_block = 256;
        int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;
        add_bias_kernel<<<blocks_per_grid, threads_per_block>>>(output, bias, m, n);
        CUDA_CHECK(cudaGetLastError());
        CUBLAS_CHECK(cublasDestroy(handle));
    }

    // 3. Corrected layer_norm_kernel: Change row to const float*
    __global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int batch_size, int seq_len, int embedding_dim, float eps) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        int s = blockIdx.y * blockDim.y + threadIdx.y;

        if (b >= batch_size || s >= seq_len) return;

        const float* row = input + (b * seq_len + s) * embedding_dim;
        float sum = 0.0f;
        float sq_sum = 0.0f;

        for (int i = 0; i < embedding_dim; i++) {
            float val = row[i];
            sum += val;
            sq_sum += val * val;
        }

        float mean = sum / embedding_dim;
        float var = sq_sum / embedding_dim - mean * mean;
        float inv_std = 1.0f / sqrtf(var + eps);

        float* out_row = output + (b * seq_len + s) * embedding_dim;
        for (int i = 0; i < embedding_dim; i++) {
            out_row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
        }
    }

    void layer_norm_cuda(const float* input, float* output, int batch_size, int seq_len, int embedding_dim,
                         const float* gamma, const float* beta, float eps) {
        dim3 block_dim(16, 16);
        dim3 grid_dim((batch_size + block_dim.x - 1) / block_dim.x,
                      (seq_len + block_dim.y - 1) / block_dim.y);
        layer_norm_kernel<<<grid_dim, block_dim>>>(input, gamma, beta, output, batch_size, seq_len, embedding_dim, eps);
        CUDA_CHECK(cudaGetLastError());
    }

    __global__ void reshape_to_heads_kernel(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch_size * num_heads * seq_len * head_dim;
        if (idx >= total) return;
        int d = idx % head_dim;
        int s = (idx / head_dim) % seq_len;
        int nh = (idx / (head_dim * seq_len)) % num_heads;
        int b = idx / (head_dim * seq_len * num_heads);
        int input_idx = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + nh * head_dim + d;
        output[idx] = input[input_idx];
    }

    void reshape_to_heads(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
        int total = batch_size * num_heads * seq_len * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        reshape_to_heads_kernel<<<blocks, threads>>>(input, output, batch_size, seq_len, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    __global__ void reshape_from_heads_kernel(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = batch_size * seq_len * num_heads * head_dim;
        if (idx >= total) return;
        int b = idx / (seq_len * num_heads * head_dim);
        int s = (idx / (num_heads * head_dim)) % seq_len;
        int nh = (idx / head_dim) % num_heads;
        int d = idx % head_dim;
        int input_idx = (b * num_heads + nh) * seq_len * head_dim + s * head_dim + d;
        output[idx] = input[input_idx];
    }

    void reshape_from_heads(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
        int total = batch_size * seq_len * num_heads * head_dim;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        reshape_from_heads_kernel<<<blocks, threads>>>(input, output, batch_size, seq_len, num_heads, head_dim);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. Corrected masked_softmax_kernel: Added int num_heads parameter and fixed type mismatch.
    __global__ void masked_softmax_kernel(const float* scores, float* weights, int batch_heads, int seq_len, int num_heads, const bool* mask, int max_seq_len) {
        int bh = blockIdx.x * blockDim.x + threadIdx.x;
        int i = blockIdx.y * blockDim.y + threadIdx.y;

        if (bh >= batch_heads || i >= seq_len) return;

        const float* scores_row = scores + bh * seq_len * seq_len + i * seq_len;
        float* weights_row = weights + bh * seq_len * seq_len + i * seq_len;

        float max_val = -INFINITY;
        for (int j = 0; j < seq_len; j++) {
            bool attend = (mask == nullptr) ? (j <= i) : mask[(bh / num_heads) * max_seq_len + j];
            float val = (j <= i && attend) ? scores_row[j] : -INFINITY;
            max_val = fmaxf(max_val, val);
        }

        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            bool attend = (mask == nullptr) ? (j <= i) : mask[(bh / num_heads) * max_seq_len + j];
            float val = (j <= i && attend) ? expf(scores_row[j] - max_val) : 0.0f;
            weights_row[j] = val;
            sum += val;
        }

        if (sum > 0.0f) {
            for (int j = 0; j < seq_len; j++) {
                weights_row[j] /= sum;
            }
        }
    }

    // 5. Corrected sample_topk_cuda: Now defined inside cuda_nn.
    __global__ void sample_kernel(int* indices, float* probs, int topk, int batch_size, int vocab_size, int* sample_token, unsigned long long seed) {
        int b = blockIdx.x * blockDim.x + threadIdx.x;
        if (b >= batch_size) return;

        curandState state;
        curand_init(seed, b, 0, &state);

        float u = curand_uniform(&state);
        float cumsum = 0.0f;

        for (int k = 0; k < topk; k++) {
            cumsum += probs[b * topk + k];
            if (cumsum > u) {
                sample_token[b] = indices[b * vocab_size + k];
                return;
            }
        }
        sample_token[b] = indices[b * vocab_size + topk - 1];
    }

    void sample_topk_cuda(int* indices, float* probs, int topk, int batch_size, int vocab_size, int* sample_token) {
        int threads_per_block = 256;
        int blocks_per_grid = (batch_size + threads_per_block - 1) / threads_per_block;

        int *d_indices;
        float *d_probs;
        CUDA_CHECK(cudaMalloc(&d_indices, batch_size * vocab_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_probs, batch_size * topk * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_indices, indices, batch_size * vocab_size * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_probs, probs, batch_size * topk * sizeof(float), cudaMemcpyHostToDevice));

        sample_kernel<<<blocks_per_grid, threads_per_block>>>(d_indices, d_probs, topk, batch_size, vocab_size, sample_token, static_cast<unsigned long long>(time(NULL)));

        CUDA_CHECK(cudaFree(d_indices));
        CUDA_CHECK(cudaFree(d_probs));
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
        const bool* mask, int max_seq_len, float* output) {

        int head_dim = embedding_dim / num_heads;
        size_t hidden_size = batch_size * seq_len * embedding_dim * sizeof(float);
        size_t intermediate_size = batch_size * seq_len * intermediate_dim * sizeof(float);
        size_t attention_size = batch_size * num_heads * seq_len * head_dim * sizeof(float);
        size_t scores_size = batch_size * num_heads * seq_len * seq_len * sizeof(float);

        // Allocate temporary buffers
        float *ln1, *Q, *K, *V, *Q_reshaped, *K_reshaped, *V_reshaped, *scores, *weights_buf,
              *attention_output, *attn_out_reshaped, *attn_out, *x, *ln2, *mlp_temp, *mlp_out;
        CUDA_CHECK(cudaMalloc(&ln1, hidden_size));
        CUDA_CHECK(cudaMalloc(&Q, hidden_size));
        CUDA_CHECK(cudaMalloc(&K, hidden_size));
        CUDA_CHECK(cudaMalloc(&V, hidden_size));
        CUDA_CHECK(cudaMalloc(&Q_reshaped, attention_size));
        CUDA_CHECK(cudaMalloc(&K_reshaped, attention_size));
        CUDA_CHECK(cudaMalloc(&V_reshaped, attention_size));
        CUDA_CHECK(cudaMalloc(&scores, scores_size));
        CUDA_CHECK(cudaMalloc(&weights_buf, scores_size));
        CUDA_CHECK(cudaMalloc(&attention_output, attention_size));
        CUDA_CHECK(cudaMalloc(&attn_out_reshaped, hidden_size));
        CUDA_CHECK(cudaMalloc(&attn_out, hidden_size));
        CUDA_CHECK(cudaMalloc(&x, hidden_size));
        CUDA_CHECK(cudaMalloc(&ln2, hidden_size));
        CUDA_CHECK(cudaMalloc(&mlp_temp, intermediate_size));
        CUDA_CHECK(cudaMalloc(&mlp_out, hidden_size));

        // Self-attention block
        layer_norm_cuda(input, ln1, batch_size, seq_len, embedding_dim, sa_norm_weights, sa_norm_biases, 1e-5f);

        // Compute Q, K, V
        linear_forward_cuda(ln1, batch_size, seq_len, embedding_dim, embedding_dim, q_weights, q_biases, Q);
        linear_forward_cuda(ln1, batch_size, seq_len, embedding_dim, embedding_dim, k_weights, k_biases, K);
        linear_forward_cuda(ln1, batch_size, seq_len, embedding_dim, embedding_dim, v_weights, v_biases, V);

        // Reshape for multi-head attention
        reshape_to_heads(Q, Q_reshaped, batch_size, seq_len, num_heads, head_dim);
        reshape_to_heads(K, K_reshaped, batch_size, seq_len, num_heads, head_dim);
        reshape_to_heads(V, V_reshaped, batch_size, seq_len, num_heads, head_dim);

        // Attention computation using adjusted SGEMM parameters.
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        float alpha = 1.0f / sqrtf(static_cast<float>(head_dim));
        float beta = 0.0f;
        // First SGEMM: compute scores = Q_reshaped * (K_reshaped)^T using op(A)=N, op(B)=T.
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                                               seq_len, seq_len, head_dim,
                                               &alpha,
                                               Q_reshaped, seq_len, seq_len * head_dim,
                                               K_reshaped, head_dim, seq_len * head_dim,
                                               &beta,
                                               scores, seq_len, seq_len * seq_len,
                                               batch_size * num_heads));

        dim3 threads_per_block(16, 16);
        dim3 blocks_per_grid(
            (batch_size * num_heads + threads_per_block.x - 1) / threads_per_block.x,
            (seq_len + threads_per_block.y - 1) / threads_per_block.y
        );
        // Launch masked softmax kernel.
        masked_softmax_kernel<<<blocks_per_grid, threads_per_block>>>(scores, weights_buf, batch_size * num_heads, seq_len, num_heads, mask, max_seq_len);
        CUDA_CHECK(cudaGetLastError());

        alpha = 1.0f;
        // Second SGEMM: compute attention_output = weights_buf * V_reshaped.
        CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                                               seq_len, head_dim, seq_len,
                                               &alpha,
                                               weights_buf, seq_len, seq_len * seq_len,
                                               V_reshaped, seq_len, seq_len * head_dim,
                                               &beta,
                                               attention_output, seq_len, seq_len * head_dim,
                                               batch_size * num_heads));

        // Reshape attention output and project.
        reshape_from_heads(attention_output, attn_out_reshaped, batch_size, seq_len, num_heads, head_dim);
        linear_forward_cuda(attn_out_reshaped, batch_size, seq_len, embedding_dim, embedding_dim, out_weights, out_biases, attn_out);

        // Residual connection.
        int num_elements = batch_size * seq_len * embedding_dim;
        int threads_per_add = 256;
        int add_blocks = (num_elements + threads_per_add - 1) / threads_per_add;
        cuda_utils::add_kernel<<<add_blocks, threads_per_add>>>(input, attn_out, x, num_elements);

        // MLP block.
        layer_norm_cuda(x, ln2, batch_size, seq_len, embedding_dim, mlp_norm_weights, mlp_norm_biases, 1e-5f);
        linear_forward_cuda(ln2, batch_size, seq_len, embedding_dim, intermediate_dim, mlp_w1_weights, mlp_w1_biases, mlp_temp);
        cuda_utils::gelu_cuda(mlp_temp, batch_size * seq_len * intermediate_dim);
        linear_forward_cuda(mlp_temp, batch_size, seq_len, intermediate_dim, embedding_dim, mlp_w2_weights, mlp_w2_biases, mlp_out);
        cuda_utils::add_kernel<<<add_blocks, threads_per_add>>>(x, mlp_out, output, num_elements);

        // Cleanup.
        CUBLAS_CHECK(cublasDestroy(handle));
        CUDA_CHECK(cudaFree(ln1));
        CUDA_CHECK(cudaFree(Q));
        CUDA_CHECK(cudaFree(K));
        CUDA_CHECK(cudaFree(V));
        CUDA_CHECK(cudaFree(Q_reshaped));
        CUDA_CHECK(cudaFree(K_reshaped));
        CUDA_CHECK(cudaFree(V_reshaped));
        CUDA_CHECK(cudaFree(scores));
        CUDA_CHECK(cudaFree(weights_buf));
        CUDA_CHECK(cudaFree(attention_output));
        CUDA_CHECK(cudaFree(attn_out_reshaped));
        CUDA_CHECK(cudaFree(attn_out));
        CUDA_CHECK(cudaFree(x));
        CUDA_CHECK(cudaFree(ln2));
        CUDA_CHECK(cudaFree(mlp_temp));
        CUDA_CHECK(cudaFree(mlp_out));
    }

} // namespace cuda_nn

extern "C" {

    void create_causal_mask_cuda(int seq_len, bool* mask) {
        cuda_utils::create_causal_mask_cuda(seq_len, mask);
    }

    void index_causal_mask_cuda(const int* input_pos, int batch_size, int seq_len, int max_seq_len, bool* result_mask) {
        cuda_utils::index_causal_mask_cuda(input_pos, batch_size, seq_len, max_seq_len, result_mask);
    }

    void sample_topk_cuda(int* indices, float* probs, int topk, int batch_size, int vocab_size, int* sample_token) {
        cuda_nn::sample_topk_cuda(indices, probs, topk, batch_size, vocab_size, sample_token);
    }

    void embedding_forward_cuda(const int* input, int batch_size, int seq_len, int vocab_size, int embedding_dim, const float* weight, float* output) {
        cuda_nn::embedding_forward_cuda(input, batch_size, seq_len, vocab_size, embedding_dim, weight, output);
    }

    void linear_forward_cuda(const float* input, int batch_size, int seq_len, int in_features, int out_features,
                             const float* weight, const float* bias, float* output) {
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
        const bool* mask, int max_seq_len, float* output) {
        cuda_nn::transformer_decoder_forward_cuda(input, batch_size, seq_len, embedding_dim, num_heads, intermediate_dim,
            q_weights, q_biases, k_weights, k_biases, v_weights, v_biases,
            out_weights, out_biases, mlp_w1_weights, mlp_w1_biases,
            mlp_w2_weights, mlp_w2_biases, sa_norm_weights, sa_norm_biases,
            mlp_norm_weights, mlp_norm_biases, mask, max_seq_len, output);
    }
}

