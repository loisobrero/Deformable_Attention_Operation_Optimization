#ifndef DEFORMABLE_ATTENTION_H
#define DEFORMABLE_ATTENTION_H

#include <vector>

void deformable_attention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& output, int seq_len, int num_heads, int head_dim);

#endif // DEFORMABLE_ATTENTION_H
