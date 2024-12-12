import faiss
import json
import jieba
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import random
import numpy as np
from bert_score import score
from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from tqdm import tqdm
import sys
sys.path.append("..")
from gnnencoder.encoder import SentenceEncoder

class Ex_Retriver():
    def __init__(self, ex_file, paths=None, encode_method='KATE'):
        '''
        input: ex_file: 需要构建检索的例子文件（一般为原始训练集）
        '''
        self.encode_method = encode_method
        self.selected_k = 4

        with open(ex_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.sents = []
            self.labels = []
            self.predict = []
            self.feedback = []
            if encode_method == 'gnn':
                for d in data:
                    self.sents.append(d['input'])
                    self.labels.append(d['output'])
                    self.predict.append(d['predict'])
                    self.feedback.append(d['feedback'])
            else:
                for d in data:
                    self.sents.append(d['input'])
                    self.labels.append(d['output'])
        self.data_dict = {}
        if encode_method == 'gnn':
            for sent, label, predict, feedback in zip(self.sents, self.labels, self.predict, self.feedback):
                self.data_dict[sent] = [label, predict, feedback]
        else:
            for sent, label in zip(self.sents, self.labels):
                self.data_dict[sent] = label

        # Initialize different models based on the specified method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if encode_method == 'KATE':
            self.model = AutoModel.from_pretrained(paths['bert_path']).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(paths['bert_path'])
            self.init_embeddings(self.sents)
        elif encode_method == 'sbert' or encode_method == 'kmeans':
            self.model = AutoModel.from_pretrained(paths['sbert_path']).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(paths['sbert_path'])
            self.init_embeddings(self.sents)
        elif encode_method == 'gnn':
            gnn_model_path = paths['gnn_path']
            bert_model_path = paths['bert_path']
            self.gnn_encoder = SentenceEncoder(gnn_model_path, bert_model_path)
            self.init_embeddings(self.sents)
        elif encode_method == 'bm25':
            print("Initializing BM25...")
            self.tokenized_sents = [list(jieba.cut(sent, cut_all=False)) for sent in self.sents]
            self.bm25 = BM25Okapi(self.tokenized_sents)
        elif encode_method == 'mmr':
            print("Initializing MMR...")
            self.bert_model_path = paths['bert_path']
            self.model = AutoModel.from_pretrained(paths['bert_path']).to(self.device)
        elif encode_method == 'random':
            pass
        else:
            raise NotImplementedError
        
    def encode_sentences(self, sents, batch_size=512):
        '''
        sents: 所有需要编码的句子
        '''
        if self.encode_method == 'KATE':
            return self.bert_encode_sentences(sents)
        elif self.encode_method == 'sbert' or self.encode_method == 'kmeans':
            return self.bert_encode_sentences(sents)
        elif self.encode_method == 'gnn':
            return self.gnn_encode_sentences(sents)
        elif self.encode_method is None:
            return None
        else:
            raise NotImplementedError
        
    def gnn_encode_sentences(self, sents, batch_size=128):
        '''
        sents: 所有需要编码的句子
        '''
        all_dims_embeddings = [[], [], []]

        for i in range(0, len(sents), batch_size):
            # 每个维度分别构建检索
            batch_sents = sents[i:i + batch_size]
            dims_representations, avg_representation = self.gnn_encoder.encode(batch_sents)
            avg_embeddings = avg_representation.cpu().numpy()

            for i in range(2):
                all_dims_embeddings[i].append(dims_representations[i].cpu().numpy())
            all_dims_embeddings[2].append(avg_embeddings)

        # return np.concatenate(all_embeddings, axis=0)
        return [np.concatenate(all_dims_embeddings[i], axis=0) for i in range(3)]

    def bert_encode_sentences(self, sents, batch_size=512):
        '''
        sents: 所有需要编码的句子
        batch_size: 每次编码的batch size
        '''
        all_embeddings = []

        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i:i + batch_size]
            encoded_input = self.tokenizer(batch_sents, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}  # Move input to device

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            if self.encode_method == 'sbert' or self.encode_method == 'kmeans' or self.encode_method == 'KATE':
                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def init_embeddings(self, sents):
        print("Initializing embeddings...")
        # build the index using FAISS
        embeddings = self.encode_sentences(sents)
        if self.encode_method == 'gnn':
            # 针对每个特征维度分别构建检索
            d = embeddings[0].shape[1]
            self.index = [faiss.IndexFlatL2(d) for i in range(3)]
            for i in range(3):
                self.index[i].add(embeddings[i])
        elif self.encode_method == 'KATE' or self.encode_method == 'sbert':
            d = embeddings.shape[1]
            if self.encode_method == 'KATE':
                self.index = faiss.IndexFlatL2(d)
            else:
                self.index = faiss.IndexFlatIP(d)
            self.index.add(embeddings)
        elif self.encode_method == 'kmeans':
            min_samples_per_cluster = 350
            for rs in range(25): # Limit the number of attempts to avoid an infinite loop
                self.index = KMeans(n_clusters=self.selected_k, random_state=rs).fit(embeddings)
                # print([len(np.where(self.index.labels_ == cluster)[0]) for cluster in range(self.selected_k)])
                if np.min(np.bincount(self.index.labels_)) >= min_samples_per_cluster:
                    break
            # Group train indices by their assigned cluster
            cluster_indices = {i: np.where(self.index.labels_ == i)[0] for i in range(self.selected_k)}
            # assert
            for cluster in range(self.selected_k):
                assert len(cluster_indices[cluster]) >= min_samples_per_cluster
        else:
            raise NotImplementedError

    def search_examples(self, query, selected_k, verbose=False):
        if verbose:
            print(f"\nSearching for: {query}")

        if selected_k is None:
            selected_k = self.selected_k

        if self.encode_method == 'random':
            return random.sample(list(zip(self.data_dict.keys(), self.data_dict.values())), selected_k)
        elif self.encode_method == 'gnn':
            query_embeddings = self.encode_sentences([query])
            choosed_idxs = {} # 已选择的索引
            # 每个维度要选择的数量
            n_dims = [1, 1, 2]
            right_n = [1, 1, 0]
            wrong_n = [0, 0, 2]
            feture_types = ['lig', 'struct', 'avg']
            for i in range(3):
                distances, indices = self.index[i].search(query_embeddings[i], self.index[i].ntotal)
                # 按照距离远近升序排序，最相似的在列表最前面
                sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=False)
                # 每个维度取对应数量，且去重
                for idx, dist in sorted_results:
                    # 去重，同一句子只出现一次
                    if idx not in choosed_idxs.keys() and n_dims[i] > 0 and right_n[i] > 0 and self.sents[idx] != query and \
                            '你对了！' in self.data_dict[self.sents[idx]][2]:
                        choosed_idxs[idx] = dist
                        n_dims[i] -= 1
                        right_n[i] -= 1
                        if verbose:
                            print(f'{feture_types[i]}: {self.sents[idx]}')
                    elif idx not in choosed_idxs.keys() and n_dims[i] > 0 and wrong_n[i] > 0 and self.sents[idx] != query and \
                            '你错了！' in self.data_dict[self.sents[idx]][2]:
                        choosed_idxs[idx] = dist
                        n_dims[i] -= 1
                        wrong_n[i] -= 1
                        if verbose:
                            print(f'{feture_types[i]}: {self.sents[idx]}')

                    if n_dims[i] == 0:
                        break
            # 将字典转换为列表，然后按照距离排序，距离大的放到前面，距离小的放到后面（离输入更近）
            choosed_idxs = sorted(choosed_idxs.items(), key=lambda x: x[1], reverse=True)
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx, dist in choosed_idxs]
            return res
        elif self.encode_method == 'KATE' or self.encode_method == 'sbert':
            query_embedding = self.encode_sentences([query])
            distances, indices = self.index.search(query_embedding, self.index.ntotal)
            if self.encode_method == 'KATE':
                sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=False)
            else:
                sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=True)
            # Getting the top k results, excluding the input example
            top_results = sorted_results[1: 1 + selected_k]
            # Sort the selected examples from greatest to smallest distance, so that the most similar examples are closest to the input.
            top_results.reverse()
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx, dist in top_results]
            return res
        elif self.encode_method == 'kmeans':
            cluster_type = np.unique(self.index.labels_)
            random_samples_indices = []
            for cluster in cluster_type:
                cluster_indices = np.where(self.index.labels_ == cluster)[0]
                valid_indices = [idx for idx in cluster_indices if self.sents[idx] != query]
                random_samples_indices.append(np.random.choice(valid_indices))
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx in random_samples_indices]
            return res
        elif self.encode_method == 'bm25':
            # 提取检索文件中的句子
            tokenized_query = jieba.cut(query, cut_all=False)
            scores = self.bm25.get_scores(tokenized_query)
            top_results = list(np.argsort(scores)[::-1])
            selected_examples = []
            for idx in top_results:
                if self.sents[idx] != query:
                    selected_examples.append(idx)
                if len(selected_examples) == selected_k:
                    break
            selected_examples = selected_examples[::-1]
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx in selected_examples]
            return res
        elif self.encode_method == 'mmr':
            # Avoid time-consuming calculation due to large search dataset.
            mmr_corpus = self.sents[:300]
            P, R, F1 = score([query] * len(mmr_corpus), mmr_corpus, model_type=self.bert_model_path,
                             num_layers=self.model.config.num_hidden_layers,verbose=False,device=self.device, batch_size=512, nthreads=16 * 4)
            # Map each training example to its F1 score
            scores = {idx: f1_score for idx, f1_score in enumerate(F1.tolist())}
            T = []
            alpha = 0.5
            while len(T) < len(self.sents):
                mmr_score = float('-inf')
                selected_idx = None
                for idx, f1_score in scores.items():
                    if query == self.sents[idx]:
                        continue
                    if idx not in T:
                        relevance = alpha * f1_score
                        diversity = max((1 - alpha) * scores[j] for j in T) if T else 0
                        mmr_score_tmp = relevance - diversity
                        # find the maximum mmr score
                        if mmr_score_tmp > mmr_score:
                            mmr_score = mmr_score_tmp
                            selected_idx = idx

                if selected_idx is not None:
                    T.append(selected_idx)
                else:
                    break
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx in T[:selected_k]]
            return res
        else:
            raise ValueError(f"Invalid encode method: {self.encode_method}")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

