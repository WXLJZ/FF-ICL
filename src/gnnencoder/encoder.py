import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import re
import json
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')

from gnnencoder.models_x import MultiHeadGAT

nhead = 2
token_dim = 768
hidden_dim = 128
output_dim = 512

def parse_output(s):
    groups = re.findall(r'\[(.*?)\]', s)
    elements = [re.split(r',\s*', group) for group in groups]
    return elements

class SentenceEncoder:
    def __init__(self, gnn_model_path, bert_model_path, max_len=256):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 初始化GNN模型并加载权重
        self.gnn_model = MultiHeadGAT(nhead, token_dim, hidden_dim, output_dim).to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.gnn_model = nn.DataParallel(self.gnn_model, device_ids=[i for i in range(torch.cuda.device_count())])
        self.gnn_model.load_state_dict(torch.load(gnn_model_path))
        self.gnn_model.eval()
        
        # 初始化BERT模型用于编码句子
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertModel.from_pretrained(bert_model_path).to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.bert_model = nn.DataParallel(self.bert_model)
        self.bert_model.eval()
        
        self.max_len = max_len
        self.tsne_data = {
            "bert_embeddings": [],
            "gnn_embeddings_d1": [],
            "gnn_embeddings_d2": [],
            "gnn_embeddings_avg": [],
            "labels": []
        }

    def encode(self, sentences):
        # 1. 使用BERT模型编码句子
        encoded_sentences = self._batch_encode_nodes(sentences)

        # 2. 使用GNN模型进一步编码得到句子表征
        with torch.no_grad():
            dims_representations, avg_representation = self.gnn_model(encoded_sentences)

        bert_avg = encoded_sentences.mean(dim=1)

        self.tsne_data["bert_embeddings"].append(bert_avg.detach().cpu())
        self.tsne_data["gnn_embeddings_d1"].append(dims_representations[0].detach().cpu())
        self.tsne_data["gnn_embeddings_d2"].append(dims_representations[1].detach().cpu())
        self.tsne_data["gnn_embeddings_avg"].append(avg_representation.detach().cpu())
        # print("bert_avg:", bert_avg.shape)
        # print("gnn_avg:", avg_representation.shape)
        # print("gnn_dim1:", dims_representations[0].shape)
        # print("gnn_dim2:", dims_representations[1].shape)

        return dims_representations, avg_representation

    def _batch_encode_nodes(self, inputs):
        """
        批量编码文本为节点特征
        """
        encoded_inputs = self.tokenizer.batch_encode_plus(inputs, return_tensors='pt', padding='max_length',
                                                          max_length=self.max_len, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)  # 获取attention_mask

        with torch.no_grad():
            embeddings = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state  # 使用attention_mask
        return embeddings

if __name__ == '__main__':
    # 使用示例：
    gnn_model_path = '/home/xyou/workspace/simile_component_extraction/checkpoints/gnn/best_gnn_model.pt'
    bert_model_path = '/home/xyou/workspace/models/bert-base-chinese'
    encoder = SentenceEncoder(gnn_model_path, bert_model_path)

    # 假设你有一个句子列表
    sentences_list = [
        '主脉像一宽敞的大道从叶尾直通到叶尖，支持着整片叶子，也引导着叶子的生长方向。',
        '翠绿清新的阳光草坪，柔美多姿的垂吊花园，绚丽多彩的大型花田，古朴静谧的亲水木栈道，令人跃跃欲试的多样游乐设施，你们就像可爱的精灵也样永远留在我心底！',
        '下课铃响了，拍皮球的小朋友拍的津津有味，踢毽子的小朋友手脚轻松，毽子像燕子一样在飞舞。',
        '交流就像在那纵横交错的要道上约束人们的法律，准则一般，维护次序，保护生命。',
        '没有理想的人就像没有方向的孤舟，迟早会沉在“现实”这片大海里。',
        '从“季节变迁”的例子来看，冬季更像是生活中那一重重的困难，一次次考验和筛选，不合格的总要被淘汰，而晋级的则可以获得成功。',
        '你又何曾不像菊花一样坚强呢？',
        '他几声叹气，说：“要是我不在后面就好了”我们又一次玩老鹰捉小鸡的游戏了，这一次可要比上一次玩的充满团结和力量，我们小鸡排成一排，老鹰捉我们的时候，我们像一条辰龙一样摇摆着，老鹰怎么也捉不住我们。'
    ]
    # 得到句子的表征
    dims_representations, representations = encoder.encode(sentences_list)
    print(representations.shape)
    # print(representations[0][:100])