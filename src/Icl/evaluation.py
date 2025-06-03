import time
import pandas as pd
import re
import random
import json
import os

from tqdm import tqdm
from Icl.predict import Model

def parse_output(s):
    groups = re.findall(r'\[(.*?)\]', s)
    elements = [re.split(r',\s*', group) for group in groups]
    return elements

def get_correct_chunk(y_true, y_pred):
    tensor_true, vehicle_true = y_true
    tensor_pred, vehicle_pred = y_pred
    correct_chunk = dict()
    correct_chunk['tensor'] = 0
    correct_chunk['vehicle'] = 0

    for tensor_entry in tensor_pred:
        if tensor_entry in tensor_true and tensor_entry:
            correct_chunk['tensor'] += 1

    for vehicle_entry in vehicle_pred:
        if vehicle_entry in vehicle_true and vehicle_entry:
            correct_chunk['vehicle'] += 1

    return correct_chunk


def get_csr_metric(y_trues, y_preds, report=False):
    true_pair, pred_pair, pair_hit = 0, 0, 0
    for y_true, y_pred in zip(y_trues, y_preds):
        # 以防生成的结果长度不足，或者长度过大（0,1 or >2）
        if len(y_pred) == 0:
            y_pred = [[], []]
        elif len(y_pred) == 1:
            y_pred.append([])
        elif len(y_pred) > 2:
            y_pred = y_pred[:2]
        # 去除 tensor 和 vehicle 列表中的空字符串（若有）
        tensor_true_list, vehicle_true_list = [item for item in y_true[0] if item], [item for item in y_true[1] if item]
        tensor_pred_list, vehicle_pred_list = [item for item in y_pred[0] if item], [item for item in y_pred[1] if item]
        # 计算标签(真实或预测)中本体及喻体的个数
        num_tensor_true, num_vehicle_true = len(tensor_true_list), len(vehicle_true_list)
        num_tensor_pred, num_vehicle_pred = len(tensor_pred_list), len(vehicle_pred_list)
        true_pair += max(num_tensor_true, num_vehicle_true)
        pred_pair += max(num_tensor_pred, num_vehicle_pred)
        correct_chunk = get_correct_chunk(y_true, y_pred)
        if num_tensor_true == 1 and num_vehicle_true == 1:
            if correct_chunk['tensor'] == 1 and correct_chunk['vehicle'] == 1:
                pair_hit += 1
        elif num_tensor_true > 1 and num_vehicle_true > 1:
            if correct_chunk['tensor'] == 0 or correct_chunk['v'] == 0:
                pair_hit += 0
            else:
                pair_hit += max(correct_chunk['vehicle'], correct_chunk['tensor'])
        elif num_tensor_true > 1 and num_vehicle_true == 1:
            pair_hit += correct_chunk['tensor'] if correct_chunk['vehicle'] == 1 else 0
        elif num_tensor_true == 1 and num_vehicle_true > 1:
            pair_hit += correct_chunk['vehicle'] if correct_chunk['tensor'] == 1 else 0
        elif num_tensor_true == 0 and num_vehicle_true == 0:
            pass
        else:
            assert(False)

    precision = 0.0 if pred_pair == 0 else (1.0 * pair_hit / pred_pair)
    recall = 0.0 if true_pair == 0 else (1.0 * pair_hit / true_pair)
    f1 = 0.0 if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)

    if report:
        print('precision :%6.2f%% ; recall :%6.2f%% ; f1 :%6.2f%%' % (100. * precision, 100. * recall, 100. * f1))

    return precision, recall, f1

def get_lcc_metric(y_trues, y_preds, report=True):
    true_pair, pred_pair, pair_hit = 0, 0, 0
    true_tenor, pred_tenor, tenor_hit = 0, 0, 0
    true_vehicle, pred_vehicle, vehicle_hit = 0, 0, 0
    for y_true, y_pred in zip(y_trues, y_preds):
        # 以防生成的结果长度不足，或者长度过大（0,1 or >2）
        if len(y_pred) == 0:
            y_pred = [[], []]
        elif len(y_pred) == 1:
            y_pred.append([])
        elif len(y_pred) > 2:
            y_pred = y_pred[:2]
        # print(f"y_true: {y_true}, y_pred: {y_pred}")
        tenor_true_list, vehicle_true_list = [item for item in y_true[0] if item], [item for item in y_true[1] if item]
        tenor_pred_list, vehicle_pred_list = [item for item in y_pred[0] if item], [item for item in y_pred[1] if item]
        # print(f"tenor_true_list: {tenor_true_list}, vehicle_true_list: {vehicle_true_list}")
        # print(f"tenor_pred_list: {tenor_pred_list}, vehicle_pred_list: {vehicle_pred_list}")
        # 计算本体指标
        true_tenor += len(tenor_true_list)
        pred_tenor += len(tenor_pred_list)
        for tenor_entry in tenor_pred_list:
            if tenor_entry in tenor_true_list and tenor_entry:
                tenor_hit += 1
        # 计算喻体指标
        true_vehicle += len(vehicle_true_list)
        pred_vehicle += len(vehicle_pred_list)
        for vehicle_entry in vehicle_pred_list:
            if vehicle_entry in vehicle_true_list and vehicle_entry:
                vehicle_hit += 1
        # 计算pair指标
        true_pair += len(tenor_true_list)
        pred_pair += len(tenor_pred_list)
        for t, v in zip(tenor_pred_list, vehicle_pred_list):
            if t in tenor_true_list and v in vehicle_true_list:
                pair_hit += 1
    # pair指标
    precision_pair = 0 if pred_pair == 0 else pair_hit / pred_pair
    recall_pair = 0 if true_pair == 0 else pair_hit / true_pair
    f1_pair = 0 if precision_pair + recall_pair == 0 else 2 * precision_pair * recall_pair / (
                precision_pair + recall_pair)
    # 本体指标
    precision_tenor = 0 if pred_tenor == 0 else tenor_hit / pred_tenor
    recall_tenor = 0 if true_tenor == 0 else tenor_hit / true_tenor
    f1_tenor = 0 if precision_tenor + recall_tenor == 0 else 2 * precision_tenor * recall_tenor / (
                precision_tenor + recall_tenor)
    # 喻体指标
    precision_vehicle = 0 if pred_vehicle == 0 else vehicle_hit / pred_vehicle
    recall_vehicle = 0 if true_vehicle == 0 else vehicle_hit / true_vehicle
    f1_vehicle = 0 if precision_vehicle + recall_vehicle == 0 else 2 * precision_vehicle * recall_vehicle / (
                precision_vehicle + recall_vehicle)
    if report:
        print('pair precision :%6.2f%% ; pair recall :%6.2f%% ; pair f1 :%6.2f%%' % (
        100. * precision_pair, 100. * recall_pair, 100. * f1_pair))
        print('tenor precision :%6.2f%% ; tenor recall :%6.2f%% ; tenor f1 :%6.2f%%' % (
        100. * precision_tenor, 100. * recall_tenor, 100. * f1_tenor))
        print('vehicle precision :%6.2f%% ; vehicle recall :%6.2f%% ; vehicle f1 :%6.2f%%' % (
        100. * precision_vehicle, 100. * recall_vehicle, 100. * f1_vehicle))
    return precision_pair, recall_pair, f1_pair

def get_cmre_metric(y_trues, y_preds, report=False):
    true_num, pred_num, right_num = 0, 0, 0
    for y_true, y_pred in zip(y_trues, y_preds):
        y_true = list({string_true for sublist_true in y_true for string_true in sublist_true})
        true_num += len(y_true)
        y_pred = list({string_pred for sublist_pred in y_pred for string_pred in sublist_pred if string_pred != ''})
        pred_num += len(y_pred)

        # 计算句子中的span在金标签中的数量
        right_word = 0
        for word_pred in y_pred:
            if word_pred in y_true:
                right_word += 1
        right_num += right_word

    precision = 0 if pred_num == 0 else right_num / pred_num
    recall = 0 if true_num == 0 else right_num / true_num
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    if report:
        print('precision :%6.2f%% ; recall :%6.2f%% ; f1 :%6.2f%%' % (100. * precision, 100. * recall, 100. * f1))

    return precision, recall, f1

def evaluate(test_data, model_name_or_path, checkpoint_dir, template, temperature=0.1, top_p=0.9, finetuning_type="lora", results_path_prefix=None, dataset_name='CMRE'):
    model = Model(model_name_or_path, checkpoint_dir, template, temperature, top_p, finetuning_type)

    y_trues, y_preds = [], []
    all_results = [] # 用于保存全部记录
    error_records = []  # 用于保存错误的记录
    error_id = 0 # 记录错误记录的条数
    # for data in tqdm(random.sample(test_data, 100)):
    for data in tqdm(test_data, desc="Predicting the Test_Data"):
        instruction = data['instruction']

        message = [
            {"role": "user", "content": instruction}
        ]
        # print(f"instruction: {instruction}")
        pred = model.generate(message)[0].response_text
        # print(f"predict: {pred}")
        # print("====="*20)
        true = data['output']
        
        try:
            pred_list = parse_output(pred)
            true_list = parse_output(true)
        
        except Exception as e:
            # 如果在生成或解析预测过程中出现异常，将预测结果设置为异常信息，将预测结果设置为空列表
            pred = str(e)
            pred_list = []

        y_trues.append(true_list)
        y_preds.append(pred_list)

        all_results.append({
            'sent_id': data['id'],
            'instruction': instruction,
            'true': true,
            'predict': pred
        })
        # 如果预测的输出与真实的输出不匹配，将它们添加到错误记录中
        if pred != true:
            error_id += 1
            error_records.append({
                'error_id': str(error_id),
                'sent_id': data['id'],
                'instruction': instruction,
                'true': true,
                'predict': pred
            })

    if not os.path.exists(results_path_prefix):
        os.makedirs(results_path_prefix)

    # 保存错误记录到json文件
    if error_records:
        error_path = f"{results_path_prefix}error_records_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
        results_path = f"{results_path_prefix}all_results_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json"
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_records, f, ensure_ascii=False, indent=4)
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    # 计算p,r,f1
    if dataset_name == 'CMRE':
        precision, recall, f1 = get_cmre_metric(y_trues, y_preds, report=False)
    elif dataset_name == 'CSR':
        precision, recall, f1 = get_csr_metric(y_trues, y_preds, report=False)
    elif dataset_name == 'LCC':
        precision, recall, f1 = get_lcc_metric(y_trues, y_preds, report=True)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return precision, recall, f1