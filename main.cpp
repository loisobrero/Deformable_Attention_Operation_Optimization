#include <iostream>
#include <vector>
#include <chrono>
#include "matrix_multiplication.h"
#include "deformable_attention.h"

int main() {
    const int N = 1024; // Increase matrix size to 1024 (N x N)
    const int num_heads = 8;
    const int head_dim = N / num_heads;
    const int seq_len = N;

    // Initialize matrices and deformable attention tensors (for simplicity, fill with ones)
    std::vector<float> A(N * N, 1.0f), B(N * N, 1.0f), C_scalar(N * N, 0.0f), C_avx2(N * N, 0.0f);
    std::vector<float> Q(seq_len * num_heads * head_dim, 1.0f), K(seq_len * num_heads * head_dim, 1.0f), V(seq_len * num_heads * head_dim, 1.0f), output_deformable(seq_len * num_heads * head_dim, 0.0f);

    // Benchmark scalar matrix multiplication
    auto start_scalar = std::chrono::high_resolution_clock::now();
    scalar_matrix_multiply(A, B, C_scalar, N);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto duration_scalar = std::chrono::duration_cast<std::chrono::microseconds>(end_scalar - start_scalar);
    std::cout << "Scalar Matrix Multiplication took: " << duration_scalar.count() << " microseconds.\n";

    // Benchmark AVX2-optimized matrix multiplication
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    avx2_matrix_multiply(A, B, C_avx2, N);
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    auto duration_avx2 = std::chrono::duration_cast<std::chrono::microseconds>(end_avx2 - start_avx2);
    std::cout << "AVX2-Optimized Matrix Multiplication took: " << duration_avx2.count() << " microseconds.\n";

    // Benchmark deformable attention operation
    auto start_deformable = std::chrono::high_resolution_clock::now();
    deformable_attention(Q, K, V, output_deformable, seq_len, num_heads, head_dim);
    auto end_deformable = std::chrono::high_resolution_clock::now();
    auto duration_deformable = std::chrono::duration_cast<std::chrono::microseconds>(end_deformable - start_deformable);
    std::cout << "Deformable Attention Operation took: " << duration_deformable.count() << " microseconds.\n";

    // Verify correctness (compare C_scalar, C_avx2, and output_deformable)
    // (Add verification code if needed)

    return 0;
}
