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

// ### Utility Functions ###

// Kernel to create a causal mask for attention (column-major order)
__global__ void create_causal_mask_kernel(int seq_len, bool* mask) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column index
    if (i < seq_len && j < seq_len) {
        mask[j * seq_len + i] = (i >= j);  // column-major: index = col * height + row
    }
}

// Function to create a causal mask on the GPU
void create_causal_mask_cuda(int seq_len, bool* mask) {
    dim3 block_dim(32, 32);
    dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x,
                  (seq_len + block_dim.y - 1) / block_dim.y);
    create_causal_mask_kernel<<<grid_dim, block_dim>>>(seq_len, mask);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to index a causal mask based on input positions
__global__ void index_causal_mask_kernel(const int* input_pos, int batch_size, int max_seq_len, bool* result_mask) {
    int b = blockIdx.y * blockDim.y + threadIdx.y; // batch index
    int j = blockIdx.x * blockDim.x + threadIdx.x; // sequence index
    if (b < batch_size && j < max_seq_len) {
        result_mask[j * batch_size + b] = (j <= input_pos[b]);  // column-major
    }
}

// Function to index a causal mask on the GPU
void index_causal_mask_cuda(const int* input_pos, int batch_size, int seq_len, int max_seq_len, bool* result_mask) {
    dim3 block_dim(32, 32);
    dim3 grid_dim((max_seq_len + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);
    index_causal_mask_kernel<<<grid_dim, block_dim>>>(input_pos, batch_size, max_seq_len, result_mask);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to compute top-k probabilities for sampling
__global__ void compute_topk_probs_kernel(float* logits, int* indices, int topk, int vocab_size, int batch_size, float* probs) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    int* batch_indices = indices + b * vocab_size;
    float max_val = -INFINITY;
    for (int k = 0; k < topk; k++) {
        int idx = batch_indices[k];
        float val = logits[idx * batch_size + b];  // column-major
        if (val > max_val) max_val = val;
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < topk; k++) {
        int idx = batch_indices[k];
        float exp_val = expf(logits[idx * batch_size + b] - max_val);
        probs[k * batch_size + b] = exp_val;
        sum_exp += exp_val;
    }
    for (int k = 0; k < topk; k++) {
        probs[k * batch_size + b] /= sum_exp;
    }
}

// Kernel for GELU activation
__global__ void gelu_kernel(float* x, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float val = x[i];
        float gelu = 0.5f * val * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (val + 0.044715f * val * val * val)));
        x[i] = gelu;
    }
}

// Function to apply GELU activation on the GPU
void gelu_cuda(float* x, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    gelu_kernel<<<blocks_per_grid, threads_per_block>>>(x, size);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to add two tensors element-wise
__global__ void add_kernel(const float* a, const float* b, float* c, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        c[idx] = a[idx] + b[idx];
    }
}

} // namespace cuda_utils

namespace cuda_nn {

// ### Neural Network Functions ###

// Kernel for embedding forward pass (column-major layout)
__global__ void embedding_forward_kernel(const int* input, int batch_size, int seq_len, int embedding_dim, const float* weight, float* output) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;  // batch index
    int s = blockIdx.x * blockDim.x + threadIdx.x;  // sequence index
    if (b < batch_size && s < seq_len) {
        int token = input[b * seq_len + s];  // row-major: token index
        float* out_ptr = output + (b * seq_len + s) * embedding_dim;
        const float* weight_ptr = weight + token * embedding_dim;
        for (int d = 0; d < embedding_dim; ++d) {
            out_ptr[d] = weight_ptr[d];
        }
    }
}

// Function for embedding forward pass on the GPU
void embedding_forward_cuda(const int* input, int batch_size, int seq_len, int embedding_dim, const float* weight, float* output) {
    dim3 block_dim(32, 8);
    dim3 grid_dim((seq_len + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);
    embedding_forward_kernel<<<grid_dim, block_dim>>>(input, batch_size, seq_len, embedding_dim, weight, output);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to add bias to the output of a linear layer
__global__ void add_bias_kernel(float* output, const float* bias, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m * n) {
        int row = idx % m;
        int col = idx / m;
        output[col * m + row] += bias[col];
    }
}

// Function for linear forward pass (assumes column-major layout)
void linear_forward_cuda(const float* input, int batch_size, int seq_len, int in_features, int out_features,
                         const float* weight, const float* bias, float* output) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f;
    float beta = 0.0f;
    int m = batch_size * seq_len;  // number of rows
    int n = out_features;          // number of columns
    int k = in_features;
    
    // cuBLAS uses column-major ordering, but our tensors are row-major
    // So we compute: output = weight * input^T (which is equivalent to input * weight^T in row-major)
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                             n, m, k, 
                             &alpha, 
                             weight, k,
                             input, k,
                             &beta,
                             output, n));
                             
    // Add bias
    int total = m * n;
    int threads_per_block = 256;
    int blocks_per_grid = (total + threads_per_block - 1) / threads_per_block;
    add_bias_kernel<<<blocks_per_grid, threads_per_block>>>(output, bias, n, m);
    
    CUDA_CHECK(cudaGetLastError());
    CUBLAS_CHECK(cublasDestroy(handle));
}

// Kernel for layer normalization
__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, 
                                  int batch_size, int seq_len, int embedding_dim, float eps) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;  // batch index
    int s = blockIdx.y * blockDim.y + threadIdx.y;  // sequence index
    if (b >= batch_size || s >= seq_len) return;
    const float* row = input + (s * batch_size + b) * embedding_dim;
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
    float* out_row = output + (s * batch_size + b) * embedding_dim;
    for (int i = 0; i < embedding_dim; i++) {
        out_row[i] = (row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

// Function for layer normalization on the GPU
void layer_norm_cuda(const float* input, float* output, int batch_size, int seq_len, int embedding_dim,
                     const float* gamma, const float* beta, float eps) {
    dim3 block_dim(16, 16);
    dim3 grid_dim((batch_size + block_dim.x - 1) / block_dim.x,
                  (seq_len + block_dim.y - 1) / block_dim.y);
    layer_norm_kernel<<<grid_dim, block_dim>>>(input, gamma, beta, output, batch_size, seq_len, embedding_dim, eps);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to reshape tensor for multi-head attention
__global__ void reshape_to_heads_kernel(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * seq_len * head_dim;
    if (idx >= total) return;
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int nh = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);
    int input_idx = (s * batch_size + b) * (num_heads * head_dim) + (nh * head_dim + d);
    output[idx] = input[input_idx];
}

// Function to reshape tensor for multi-head attention
void reshape_to_heads(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int total = batch_size * num_heads * seq_len * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_to_heads_kernel<<<blocks, threads>>>(input, output, batch_size, seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel to reshape tensor from multi-head attention back to original shape
__global__ void reshape_from_heads_kernel(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * num_heads * head_dim;
    if (idx >= total) return;
    int d = idx % head_dim;
    int nh = (idx / head_dim) % num_heads;
    int s = (idx / (head_dim * num_heads)) % seq_len;
    int b = idx / (head_dim * num_heads * seq_len);
    int output_idx = (s * batch_size + b) * (num_heads * head_dim) + (nh * head_dim + d);
    output[output_idx] = input[idx];
}

// Function to reshape tensor from multi-head attention back to original shape
void reshape_from_heads(const float* input, float* output, int batch_size, int seq_len, int num_heads, int head_dim) {
    int total = batch_size * seq_len * num_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_from_heads_kernel<<<blocks, threads>>>(input, output, batch_size, seq_len, num_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

// Kernel for masked softmax
__global__ void masked_softmax_kernel(const float* scores, float* weights, int batch_heads, int seq_len, 
                                      int num_heads, const bool* mask, int max_seq_len) {
    int bh = blockIdx.x * blockDim.x + threadIdx.x;  // batch-head index
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // row index
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

// Kernel to transpose a matrix (for each batch-head)
__global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int row = idx % rows;
    int col = idx / rows;
    out[row * cols + col] = in[col * rows + row];
}

// Kernel for sampling from top-k probabilities
__global__ void sample_kernel(int* indices, float* probs, int topk, int batch_size, int vocab_size, int* sample_token, 
                              unsigned long long seed) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    curandState state;
    curand_init(seed, b, 0, &state);
    float u = curand_uniform(&state);
    float cumsum = 0.0f;
    for (int k = 0; k < topk; k++) {
        cumsum += probs[k * batch_size + b];
        if (cumsum > u) {
            sample_token[b] = indices[k * batch_size + b];
            return;
        }
    }
    sample_token[b] = indices[(topk - 1) * batch_size + b];
}

// Function to sample from top-k probabilities on the GPU
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

// Function for transformer decoder forward pass on the GPU
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

    // Allocate temporary buffers with proper scoping
    float *ln1, *Q, *K, *V, *Q_reshaped, *K_reshaped, *K_transposed, *V_reshaped, *scores, *weights_buf,
          *attention_output, *attn_out_reshaped, *attn_out, *x, *ln2, *mlp_temp, *mlp_out;
    CUDA_CHECK(cudaMalloc(&ln1, hidden_size));
    CUDA_CHECK(cudaMalloc(&Q, hidden_size));
    CUDA_CHECK(cudaMalloc(&K, hidden_size));
    CUDA_CHECK(cudaMalloc(&V, hidden_size));
    CUDA_CHECK(cudaMalloc(&Q_reshaped, attention_size));
    CUDA_CHECK(cudaMalloc(&K_reshaped, attention_size));
    CUDA_CHECK(cudaMalloc(&K_transposed, attention_size));
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

    // Self-attention block: layer normalization
    layer_norm_cuda(input, ln1, batch_size, seq_len, embedding_dim, sa_norm_weights, sa_norm_biases, 1e-5f);

    // Compute Q, K, V via linear layers
    linear_forward_cuda(ln1, batch_size, seq_len, embedding_dim, embedding_dim, q_weights, q_biases, Q);
    linear_forward_cuda(ln1, batch_size, seq_len, embedding_dim, embedding_dim, k_weights, k_biases, K);
    linear_forward_cuda(ln1, batch_size, seq_len, embedding_dim, embedding_dim, v_weights, v_biases, V);

    // Reshape for multi-head attention
    reshape_to_heads(Q, Q_reshaped, batch_size, seq_len, num_heads, head_dim);
    reshape_to_heads(K, K_reshaped, batch_size, seq_len, num_heads, head_dim);
    reshape_to_heads(V, V_reshaped, batch_size, seq_len, num_heads, head_dim);

    // Precompute the transpose of K_reshaped (for each batch-head)
    int total_elements = batch_size * num_heads * seq_len * head_dim;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    for (int i = 0; i < batch_size * num_heads; i++) {
        const float* in_ptr = K_reshaped + i * seq_len * head_dim;
        float* out_ptr = K_transposed + i * seq_len * head_dim;
        transpose_kernel<<< (seq_len * head_dim + threads - 1) / threads, threads >>>(in_ptr, out_ptr, seq_len, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize()); 
    }
    CUDA_CHECK(cudaGetLastError());

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f / sqrtf(static_cast<float>(head_dim));
    float beta = 0.0f;

    // Compute attention scores: scores = Q_reshaped * K_transposed
    int m = seq_len, n = seq_len, k = head_dim;
    int strideA = seq_len * head_dim;
    int strideB = head_dim * seq_len;
    int strideC = seq_len * seq_len;
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                           n, m, k,
                                           &alpha,
                                           K_reshaped, k, strideB,
                                           Q_reshaped, k, strideA,
                                           &beta,
                                           scores, n, strideC,
                                           batch_size * num_heads));

    // Apply masked softmax
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid(
        (batch_size * num_heads + threads_per_block.x - 1) / threads_per_block.x,
        (seq_len + threads_per_block.y - 1) / threads_per_block.y
    );
    masked_softmax_kernel<<<blocks_per_grid, threads_per_block>>>(scores, weights_buf, batch_size * num_heads,
                                                                  seq_len, num_heads, mask, max_seq_len);
    CUDA_CHECK(cudaGetLastError());

    // Compute attention output: attention_output = weights_buf * V_reshaped
    m = seq_len;
    n = head_dim;
    k = seq_len;
    strideA = seq_len * seq_len;
    strideB = seq_len * head_dim;
    strideC = seq_len * head_dim;
    alpha = 1.0f;
    beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           m, n, k,
                                           &alpha,
                                           weights_buf, m, strideA,
                                           V_reshaped, m, strideB,
                                           &beta,
                                           attention_output, m, strideC,
                                           batch_size * num_heads));

    // Reshape and project attention output back to (batch_size*seq_len x embedding_dim)
    reshape_from_heads(attention_output, attn_out_reshaped, batch_size, seq_len, num_heads, head_dim);
    linear_forward_cuda(attn_out_reshaped, batch_size, seq_len, embedding_dim, embedding_dim, out_weights, out_biases, attn_out);

    // Add residual connection: x = input + attn_out
    int num_elements = batch_size * seq_len * embedding_dim;
    int threads_per_add = 256;
    int add_blocks = (num_elements + threads_per_add - 1) / threads_per_add;
    cuda_utils::add_kernel<<<add_blocks, threads_per_add>>>(input, attn_out, x, num_elements);

    // MLP block
    layer_norm_cuda(x, ln2, batch_size, seq_len, embedding_dim, mlp_norm_weights, mlp_norm_biases, 1e-5f);
    linear_forward_cuda(ln2, batch_size, seq_len, embedding_dim, intermediate_dim, mlp_w1_weights, mlp_w1_biases, mlp_temp);
    cuda_utils::gelu_cuda(mlp_temp, batch_size * seq_len * intermediate_dim);
    linear_forward_cuda(mlp_temp, batch_size, seq_len, intermediate_dim, embedding_dim, mlp_w2_weights, mlp_w2_biases, mlp_out);
    cuda_utils::add_kernel<<<add_blocks, threads_per_add>>>(x, mlp_out, output, num_elements);

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(ln1));
    CUDA_CHECK(cudaFree(Q));
    CUDA_CHECK(cudaFree(K));
    CUDA_CHECK(cudaFree(V));
    CUDA_CHECK(cudaFree(Q_reshaped));
    CUDA_CHECK(cudaFree(K_reshaped));
    CUDA_CHECK(cudaFree(K_transposed));
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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace cuda_nn

extern "C" {

// ### Exported Functions for External Use ###

void create_causal_mask_cuda(int seq_len, bool* mask) {
    cuda_utils::create_causal_mask_cuda(seq_len, mask);
}

void index_causal_mask_cuda(const int* input_pos, int batch_size, int seq_len, int max_seq_len, bool* result_mask) {
    cuda_utils::index_causal_mask_cuda(input_pos, batch_size, seq_len, max_seq_len, result_mask);
}

void sample_topk_cuda(int* indices, float* probs, int topk, int batch_size, int vocab_size, int* sample_token) {
    cuda_nn::sample_topk_cuda(indices, probs, topk, batch_size, vocab_size, sample_token);
}

void embedding_forward_cuda(const int* input, int batch_size, int seq_len, int embedding_dim, 
                            const float* weight, float* output) {
    cuda_nn::embedding_forward_cuda(input, batch_size, seq_len, embedding_dim, weight, output);
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

