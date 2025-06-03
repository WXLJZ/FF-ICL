import json
import torch
import random
import sys
import os

from Icl.templates import *
from tqdm import tqdm
from utils.logger import get_logger

logger = get_logger(__name__)

def construct_instruct(json_path, save_path, retriever=None, ICL=True, selected_k=3, dataset_name='CMRE', verbose=False, method='gnn'):
    '''
    json_path: 原始，需要建立索引的数据路径
    save_path: 保存的路径
    '''
    with open(json_path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)
        for d in tqdm(data, desc="Instruction Construction"):
            if ICL:
                examples = retriever.search_examples(d['input'], selected_k=selected_k, verbose=verbose)

                examples_str = ""
                if method == 'gnn':
                    if dataset_name == 'LCC':
                        for id, example in enumerate(examples):
                            examples_str += f'Example {id+1}:\nInput: "{example[0]}"\nPredicted Output: "{example[1][1]}"\nCorrect Output: "{example[1][0]}"\n{example[1][2]}\n'
                    else:
                        for id, example in enumerate(examples):
                            examples_str += f'示例 {id+1}:\n输入："{example[0]}"\n预测输出："{example[1][1]}"\n正确输出："{example[1][0]}"\n{example[1][2]}\n'
                else:
                    if dataset_name == 'LCC':
                        for id, example in enumerate(examples):
                            examples_str += f'Example {id+1}:\nInput: "{example[0]}"\nOutput: "{example[1]}"\n'
                    else:
                        for id, example in enumerate(examples):
                            examples_str += f'示例 {id+1}:\n输入："{example[0]}"\n输出："{example[1]}"\n'

                if dataset_name == 'CMRE':
                    prompt = random.choice(cmre_icl_template).format(example=examples_str, input=d['input'])
                elif dataset_name == 'CSR':
                    prompt = random.choice(csr_icl_template).format(example=examples_str, input=d['input'])
                elif dataset_name == 'LCC':
                    prompt = random.choice(lcc_icl_template).format(example=examples_str, input=d['input'])
                else:
                    raise ValueError("Dataset name not supported")
            else:
                if dataset_name == 'CMRE':
                    prompt = random.choice(cmre_template).format(input=d['input'])
                elif dataset_name == 'CSR':
                    prompt = random.choice(csr_template).format(input=d['input'])
                elif dataset_name == 'LCC':
                    prompt = random.choice(lcc_template).format(input=d['input'])
                else:
                    raise ValueError("Dataset name not supported")

            d['instruction'] = prompt
            d['input'] = ""

    # 直接将数据保存到对应位置
    with open(save_path, 'w', encoding='UTF-8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)

def construct_fixed_instruct(json_path, save_path, dataset_name):
    '''
    json_path: 原始，需要建立索引的数据路径
    save_path: 保存的路径
    '''
    with open(json_path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)
        for d in tqdm(data, desc="Instruction Construction"):
            if dataset_name == 'CMRE':
                examples_str = fixed_examples_cmre
                prompt = random.choice(cmre_icl_template).format(example=examples_str, input=d['input'])
            elif dataset_name == 'CSR':
                examples_str = fixed_examples_csr
                prompt = random.choice(csr_icl_template).format(example=examples_str, input=d['input'])
            elif dataset_name == 'LCC':
                examples_str = fixed_examples_lcc
                prompt = random.choice(lcc_icl_template).format(example=examples_str, input=d['input'])
            else:
                raise ValueError("Dataset name not supported")

            d['instruction'] = prompt
            d['input'] = ""

    # 直接将数据保存到对应位置
    with open(save_path, 'w', encoding='UTF-8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)

def train_data_process(ICL, prefix, inst_prefix, paths=None, dataset_name='CMRE', method='random', selected_k=3):
    if dataset_name != 'LCC':
        from Icl.ex_retriver import Ex_Retriver
        logger.info(f"Using Ex_Retriver for dataset: {dataset_name}")
    else:
        logger.info(f"Using Ex_Retriver_EN for dataset: {dataset_name}")
        from Icl.ex_retriver_en import Ex_Retriver

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not os.path.exists(inst_prefix):
        os.makedirs(inst_prefix)

    if method == 'gnn':
        train_path = prefix + 'train_ficl.json'
    else:
        train_path = prefix + 'train.json'

    inst_train_path = inst_prefix + method.lower() + '_inst_train.json'

    if ICL:
        if method == 'random':
            retriever = Ex_Retriver(ex_file=train_path, encode_method='random')
            construct_instruct(train_path, inst_train_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'KATE':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='KATE')
            construct_instruct(train_path, inst_train_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'sbert':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='sbert')
            construct_instruct(train_path, inst_train_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'gnn':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='gnn')
            construct_instruct(train_path, inst_train_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'bm25':
            retriever = Ex_Retriver(ex_file=train_path, encode_method='bm25')
            construct_instruct(train_path, inst_train_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'fixed':
            construct_fixed_instruct(train_path, inst_train_path, dataset_name)
        elif method == 'kmeans':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='kmeans')
            construct_instruct(train_path, inst_train_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'mmr':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='mmr')
            construct_instruct(train_path, inst_train_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        else:
            raise NotImplementedError
    else:
        construct_instruct(train_path, inst_prefix + 'noicl_inst_train.json', ICL=False, dataset_name=dataset_name, method=method)

    torch.cuda.empty_cache()

def test_data_process(ICL, prefix, inst_prefix, paths=None, dataset_name='CMRE', method='random', selected_k=3):
    if dataset_name != 'LCC':
        from Icl.ex_retriver import Ex_Retriver
        logger.info(f"Using Ex_Retriver for dataset: {dataset_name}")
    else:
        from Icl.ex_retriver_en import Ex_Retriver
        logger.info(f"Using Ex_Retriver_EN for dataset: {dataset_name}")

    if not os.path.exists(prefix):
        os.makedirs(prefix)

    if not os.path.exists(inst_prefix):
        os.makedirs(inst_prefix)

    if method == 'gnn':
        train_path = prefix + 'train_ficl.json'
    else:
        train_path = prefix + 'train.json'

    test_path = prefix + 'test.json'

    inst_test_path = inst_prefix + method.lower() + '_inst_test.json'

    if ICL:
        if method == 'random':
            retriever = Ex_Retriver(ex_file=train_path, encode_method='random')
            construct_instruct(test_path, inst_test_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'KATE':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='KATE')
            construct_instruct(test_path, inst_test_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'sbert':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='sbert')
            construct_instruct(test_path, inst_test_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'gnn':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='gnn')
            construct_instruct(test_path, inst_test_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'bm25':
            retriever = Ex_Retriver(ex_file=train_path, encode_method='bm25')
            construct_instruct(test_path, inst_test_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'fixed':
            construct_fixed_instruct(test_path, inst_test_path, dataset_name)
        elif method == 'kmeans':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='kmeans')
            construct_instruct(test_path, inst_test_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        elif method == 'mmr':
            retriever = Ex_Retriver(ex_file=train_path, paths=paths, encode_method='mmr')
            construct_instruct(test_path, inst_test_path, retriever, ICL=True, selected_k=selected_k, dataset_name=dataset_name, method=method)
            del retriever
        else:
            raise NotImplementedError
    else:
        construct_instruct(test_path, inst_prefix + 'noicl_inst_test.json', ICL=False, dataset_name=dataset_name, method=method)

    torch.cuda.empty_cache()

    with open(inst_test_path, 'r', encoding='utf-8') as f:
        inst_test_data = json.load(f)

    return inst_test_data


