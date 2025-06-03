import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, token_dim, hidden_dim, dropout=0.3):
        super(GATLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(token_dim, hidden_dim, bias=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # initial the weight for Adjacency Matrix
        self.adj_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # Initialize the attention mechanism
        self.attn_fc = nn.Linear(2 * hidden_dim, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding):
        """
        token_embedding: (batch_size, max_len, token_dim)
        return: updated token embeddings (batch_size, max_len, hidden_dim)
        """
        h = self.fc(token_embedding)
        h = self.layer_norm(h)  # Apply layer normalization after the linear transformation
        batch_size, max_len, H = h.size()

        # === step 1: adaptive adjacency matrix ===
        HW = torch.matmul(h, self.adj_weight)  # (batch_size, max_len, hidden_dim)
        A = torch.matmul(HW, h.transpose(1, 2))
        A = torch.sigmoid(A)  # Apply sigmoid to the adjacency matrix

        # === step 2: concat(h_i, h_j) of all token pairs ===
        h_i = h.unsqueeze(2).expand(-1, -1, max_len, -1)  # (batch_size, max_len, max_len, hidden_dim)
        h_j = h.unsqueeze(1).expand(-1, max_len, -1, -1)  # (batch_size, max_len, max_len, hidden_dim)
        h_ij = torch.cat([h_i, h_j], dim=-1)  # (batch_size, max_len, max_len, 2 * hidden_dim)
        h_ij = self.dropout(h_ij)

        # === step 3: attention mechanism ===
        e_ij = self.attn_fc(h_ij).squeeze(-1)  # (batch_size, max_len, max_len)
        e_ij = self.leakyrelu(e_ij)

        # === step 4: filtering adjacency matrix by threshold ===
        delta = torch.quantile(A, 0.7, dim=-1, keepdim=True)  # (batch_size, 1, max_len)
        mask = (A > delta).float()  # (batch_size, max_len, max_len)
        eye = torch.eye(max_len, device=A.device).unsqueeze(0).expand(batch_size, -1, -1)
        mask = torch.clamp(mask + eye, 0, 1)
        e_ij = e_ij.masked_fill(mask == 0, float('-inf'))  # Apply mask to attention scores

        # === step 5: softmax attention weights ===
        alpha = F.softmax(e_ij, dim=-1)

        h_prime = torch.bmm(alpha, h)
        return h_prime


# class MultiHeadGAT(nn.Module):
#     def __init__(self, nhead, token_dim, hidden_dim, output_dim=768, dropout=0.3):
#         super(MultiHeadGAT, self).__init__()
#
#         self.heads = nn.ModuleList()
#         for _ in range(nhead):
#             self.heads.append(GATLayer(token_dim, hidden_dim, dropout))
#
#         self.fc_concat = nn.Linear(nhead * hidden_dim, output_dim)
#         self.fcs = nn.ModuleList()
#         for _ in range(2):  # 2 dimensions
#             self.fcs.append(nn.Linear(output_dim, output_dim))
#
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(output_dim)
#
#     def forward(self, token_embedding):
#         out = []
#         for attn_head in self.heads:
#             out.append(attn_head(token_embedding))
#         concat_out = torch.cat(out, dim=2)
#
#         # Pooling over the tokens (average pooling)
#         sent_embedding = concat_out.mean(dim=1)
#
#         # Pass through a dense layer to adjust the dimensions
#         sent_embedding = self.fc_concat(sent_embedding)
#         sent_embedding = self.layer_norm(sent_embedding)  # Apply layer normalization
#
#         dims_out = [fc(self.dropout(sent_embedding)) for fc in self.fcs]  # Apply dropout before passing through the fc
#
#         return dims_out, sent_embedding


class MultiHeadGAT(nn.Module):
    def __init__(self, nhead, token_dim, hidden_dim, output_dim=768, dropout=0.3):
        super(MultiHeadGAT, self).__init__()
        assert nhead % 2 == 0, "nhead 必须为偶数，便于平分给两个维度"

        self.nhead = nhead
        half_head = nhead // 2
        self.dropout = nn.Dropout(dropout)

        # 分别为语言学和结构建立注意力头组
        self.heads_ling = nn.ModuleList([
            GATLayer(token_dim, hidden_dim, dropout) for _ in range(half_head)
        ])
        self.heads_struct = nn.ModuleList([
            GATLayer(token_dim, hidden_dim, dropout) for _ in range(half_head)
        ])

        # 每个维度一个 fc 层进行拼接后的降维
        self.fc_ling = nn.Linear(half_head * hidden_dim, output_dim)
        self.fc_struct = nn.Linear(half_head * hidden_dim, output_dim)

        self.norm_ling = nn.LayerNorm(output_dim)
        self.norm_struct = nn.LayerNorm(output_dim)

        # 全体头拼接后用于共享平均表征（avg_repr）
        self.fc_concat = nn.Linear(nhead * hidden_dim, output_dim)
        self.norm_concat = nn.LayerNorm(output_dim)

    def forward(self, token_embedding):
        # 各自注意力输出
        ling_outs = [head(token_embedding) for head in self.heads_ling]
        struct_outs = [head(token_embedding) for head in self.heads_struct]

        # 拼接注意力输出
        ling_concat = torch.cat(ling_outs, dim=2)         # [B, T, H_ling]
        struct_concat = torch.cat(struct_outs, dim=2)     # [B, T, H_struct]

        # 构建 shared average embedding（全头拼接）
        all_concat = torch.cat(ling_outs + struct_outs, dim=2)  # [B, T, H_total]
        avg_pooled = all_concat.mean(dim=1)                     # [B, H_total]
        avg_repr = self.fc_concat(self.dropout(avg_pooled))
        avg_repr = self.norm_concat(avg_repr)

        # 各维度自己的投影与归一化
        ling_repr = self.fc_ling(self.dropout(ling_concat.mean(dim=1)))
        ling_repr = self.norm_ling(ling_repr)

        struct_repr = self.fc_struct(self.dropout(struct_concat.mean(dim=1)))
        struct_repr = self.norm_struct(struct_repr)

        return [ling_repr, struct_repr], avg_repr


def contrastive_loss(dims_representations, cl_adj, tau=0.1):
    total_loss = 0
    for dim, reps_dim in enumerate(dims_representations):
        edge_dim_sample = cl_adj[:, :, dim]

        # Calculate similarity matrix
        sim_matrix = torch.einsum('be,ae->bae', reps_dim, reps_dim)
        # 降温系数，这将使相似性得分在较大范围内波动，有助于模型更好地区分正面和负面对比。
        sim_matrix = F.sigmoid(sim_matrix / tau)

        # Prepare edge_dim_sample to match the shape
        edge_dim_sample_expanded = edge_dim_sample.unsqueeze(-1).expand_as(sim_matrix)

        # Compare with edge_dim_sample_expanded for the loss calculation
        pos_mask = (edge_dim_sample_expanded == 1).float()
        neg_mask = (edge_dim_sample_expanded == 0).float()


        pos_loss = -torch.log(sim_matrix + 1e-8) * pos_mask
        pos_loss = torch.masked_select(pos_loss, pos_mask.bool()).mean()

        neg_loss = -torch.log(1.0 - sim_matrix + 1e-8) * neg_mask
        neg_loss = torch.masked_select(neg_loss, neg_mask.bool()).mean()

        total_loss += pos_loss + neg_loss
    return total_loss

