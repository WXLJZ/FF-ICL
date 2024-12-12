import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import time
import argparse
from tqdm import tqdm

from linguistics import SentenceAnalyzer
from structure import compute_structural_similarity
import warnings

warnings.filterwarnings('ignore')

def split_dev(dataset, dev_ratio):
    """
    将数据集划分为训练集和验证集。
    """
    dev_size = int(len(dataset) * dev_ratio)
    train_size = len(dataset) - dev_size
    return random_split(dataset, [train_size, dev_size])

def parse_output(s):
    groups = re.findall(r'\[(.*?)\]', s)
    elements = [re.split(r',\s*', group) for group in groups]
    return elements


class GNNDataset(Dataset):
    def __init__(self, args):
        self.max_len = args.max_len
        self.bert_model_path = args.bert_model_path
        self.lig_top_k = args.lig_top_k
        self.struct_top_k = args.struct_top_k
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 节点编码器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_path)
        self.model = BertModel.from_pretrained(self.bert_model_path).to(self.device)
        # 如果有多于一块的 GPU，并行编码
        # if torch.cuda.device_count() > 1:  
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)
        self.model.eval()
        # 边编码器
        self.analyzer = SentenceAnalyzer(top_k=self.lig_top_k)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx]['input']
        outputs = parse_output(self.data[idx]['output'])
        if self.dataset_name == 'CMRE':
            span_types = parse_output(self.data[idx]['spans_type'])
            return inputs, outputs, span_types
        elif self.dataset_name == 'CSR':
            return inputs, outputs
        else:
            raise ValueError('Invalid dataset name')
    
    def batch_encode_nodes(self, inputs):
        """
        批量编码文本为节点特征
        """
        encoded_inputs = self.tokenizer.batch_encode_plus(inputs, return_tensors='pt', padding='max_length',
                                                          max_length=self.max_len, truncation=True).to(self.device)
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']  # 获取attention_mask

        with torch.no_grad():
            embeddings = self.model(input_ids, attention_mask=attention_mask).last_hidden_state  # 使用attention_mask
        return embeddings


    def batch_generate_edge_features(self, inputs, outputs, span_types=None):
        """
        :return: 边特征，维度为 [b, b, 2]，最后一维为[语言学相似, 结构相似]
        """
        # 语言学相似度
        lig_features = self.analyzer.linguistic_feature(inputs, outputs)
        # 结构相似
        struct_features = compute_structural_similarity(inputs, outputs, span_types, dataset_name=self.dataset_name, top_k=self.struct_top_k)

        # 拼接合并 [b, b] * 2 -> [b, b, 2]  确保是三维的
        edge_features = np.stack((lig_features, struct_features), axis=-1)

        return torch.from_numpy(edge_features)

    def collate_fn(self, batch):
        '''
        :return node_features: 节点特征，维度为 [b, max_len, dim]
        :return edge_features: 边特征，维度为 [b, b, 2]
        '''
        if self.dataset_name == 'CMRE':
            inputs, outputs, span_types = zip(*batch)
            # 获取边 对比损失 矩阵
            edge_features = self.batch_generate_edge_features(inputs, outputs, span_types)
        elif self.dataset_name == 'CSR':
            inputs, outputs = zip(*batch)
            # 获取边 对比损失 矩阵
            edge_features = self.batch_generate_edge_features(inputs, outputs)
        else:
            raise ValueError('Invalid dataset name')
        # 获取节点特征
        node_features = self.batch_encode_nodes(inputs)

        return node_features, edge_features

