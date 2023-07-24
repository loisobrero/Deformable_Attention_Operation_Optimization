#include "matrix_multiplication.h"
#include "eigen/Eigen/Dense" 

void scalar_matrix_multiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void avx2_matrix_multiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N) {
    // Use Eigen for AVX2-optimized matrix multiplication
    Eigen::Map<const Eigen::MatrixXf> matA(A.data(), N, N);
    Eigen::Map<const Eigen::MatrixXf> matB(B.data(), N, N);
    Eigen::Map<Eigen::MatrixXf> matC(C.data(), N, N);

    matC = matA * matB;
}



