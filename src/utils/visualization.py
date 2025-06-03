import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import time

def tsne_visualization(bert_embeddings, avg_embeddings,
                       dim1_embeddings, dim2_embeddings,
                       labels=None, save_path="tsne_output.png"):
    """
    分别对输入（BERT）和输出（GNN）做 t-SNE 降维并绘图保存。
    - input_embeddings: Tensor [batch_size, 768]
    - output_embeddings: Tensor [batch_size, output_dim]
    - labels: 可选，分类标签，用于颜色区分
    - save_path: 图片保存路径
    """
    # 创建保存目录，安装时间戳命名
    save_path = os.path.join(save_path, f"t-SNE/{time.strftime('%Y%m%d-%H%M')}")
    os.makedirs(save_path, exist_ok=True)
    bert_up = torch.cat(bert_embeddings[:2000], dim=0).numpy()
    avg_np = torch.cat(avg_embeddings[:2000], dim=0).numpy()
    dim1_np = torch.cat(dim1_embeddings[:2000], dim=0).numpy()
    dim2_np = torch.cat(dim2_embeddings[:2000], dim=0).numpy()
    # bert_up = input_embeddings.detach().cpu().numpy()
    # avg_np = output_embeddings.detach().cpu().numpy()

    tsne_bert = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(bert_up)
    tsne_avg = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(avg_np)
    tsne_dim1 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(dim1_np)
    tsne_dim2 = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(dim2_np)

    # 图像配置
    plots = [
        (tsne_bert, "BERT Input Embedding (t-SNE)", "#EEB422", "BERT"),
        (tsne_avg, "GNN Output Embedding (t-SNE)", "#FF7D40", "GNN_AVG"),
        (tsne_dim1, "GNN dim1 Output Embedding (t-SNE)", "#00C957", "GNN_DIM1"),
        (tsne_dim2, "GNN dim2 Output Embedding (t-SNE)", "#1E90FF", "GNN_DIM2")
    ]

    for i, (data, title, color, save_name) in enumerate(plots):
        plt.figure(figsize=(5, 5))
        if labels is not None:
            plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='tab10', alpha=0.7)
        else:
            plt.scatter(data[:, 0], data[:, 1], alpha=0.7, color=color, s=5)

        plt.title(title, pad=15)

        # 去除坐标轴
        plt.xticks([])
        plt.yticks([])
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        # plt.gca().spines['top'].set_visible(False)
        # plt.gca().spines['right'].set_visible(False)
        # plt.gca().spines['left'].set_visible(False)
        # plt.gca().spines['bottom'].set_visible(False)

        # 保存
        image_path = os.path.join(save_path, f"{save_name}_tsne.png")
        plt.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[√] t-SNE 图 {save_name} 已保存至: {image_path}")


    # plt.figure(figsize=(12, 5))
    #
    # # BERT 输入
    # plt.subplot(1, 4, 1)
    # if labels is not None:
    #     plt.scatter(tsne_bert[:, 0], tsne_bert[:, 1], c=labels, cmap='tab10', alpha=0.7)
    # else:
    #     plt.scatter(tsne_bert[:, 0], tsne_bert[:, 1], alpha=0.7, color="#EEB422")
    # plt.title("BERT Input Embedding (t-SNE)")
    # plt.xlabel("dim 1")
    # plt.ylabel("dim 2")
    #
    # # GNN avg 输出
    # plt.subplot(1, 4, 2)
    # if labels is not None:
    #     plt.scatter(tsne_avg[:, 0], tsne_avg[:, 1], c=labels, cmap='tab10', alpha=0.7)
    # else:
    #     plt.scatter(tsne_avg[:, 0], tsne_avg[:, 1], alpha=0.7, color="#FF7D40")
    # plt.title("GNN Output Embedding (t-SNE)")
    # plt.xlabel("dim 1")
    # plt.ylabel("dim 2")
    #
    # # GNN dim1 输出
    # plt.subplot(1, 4, 3)
    # if labels is not None:
    #     plt.scatter(tsne_dim1[:, 0], tsne_dim1[:, 1], c=labels, cmap='tab10', alpha=0.7)
    # else:
    #     plt.scatter(tsne_dim1[:, 0], tsne_dim1[:, 1], alpha=0.7, color="#00C957")
    # plt.title("GNN dim1 Output Embedding (t-SNE)")
    # plt.xlabel("dim 1")
    # plt.ylabel("dim 2")
    #
    # # GNN dim2 输出
    # plt.subplot(1, 4, 4)
    # if labels is not None:
    #     plt.scatter(tsne_dim2[:, 0], tsne_dim2[:, 1], c=labels, cmap='tab10', alpha=0.7)
    # else:
    #     plt.scatter(tsne_dim2[:, 0], tsne_dim2[:, 1], alpha=0.7, color="#1E90FF")
    #
    # plt.tight_layout()
    #
    # # 创建保存目录
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # plt.savefig(save_path, dpi=300)
    # plt.close()
    # print(f"[√] t-SNE 图已保存至: {save_path}")
