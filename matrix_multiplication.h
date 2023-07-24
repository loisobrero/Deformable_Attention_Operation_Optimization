#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include <vector>

void scalar_matrix_multiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N);
void avx2_matrix_multiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int N);

#endif // MATRIX_MULTIPLICATION_H
