# --coding:utf-8--
import re
import json
import torch
import argparse
from tqdm import tqdm
from llamafactory import ChatModel
from Icl.templates import *
from Icl.ex_retriver import Ex_Retriver
from Icl.evaluation import *

'''使用原始基座模型（或者微调后合并权重导出的模型）预测'''

class Model:
    def __init__(self, model_name_or_path, template):
        args = {
            "model_name_or_path": model_name_or_path,
            "template": template
        }
        self.chat_model = ChatModel(args=args)

    def generate(self, query):
        res = self.chat_model.chat(query)
        return res

def parser_output_str(output, dataset_name):
    items = re.findall(r'\[[^]]*\]', output)
    if dataset_name == 'CSR':
        items = items[:2]
    result = ', '.join(items)
    if '"' in result:
        result = result.replace('"', '')
    return result

def construct_prompt(dataset_path, inst_save_path, dataset_name, paths=None, ex_file_path=None, method=None, selected_k=None):
    retriever = None
    if method not in ['fixed', 'noicl']:
        retriever = Ex_Retriver(ex_file_path, paths=paths, encode_method=method)
    else:
        pass
    with open(dataset_path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)
        for d in tqdm(data, desc="Instruction Construction"):
            examples_str = ""
            if method == 'gnn':
                examples = retriever.search_examples(d["input"], selected_k=selected_k)
                for id, example in enumerate(examples):
                    examples_str += f'示例 {id + 1}:\n输入："{example[0]}"\n预测输出："{example[1][1]}"\n正确输出："{example[1][0]}"\n{example[1][2]}\n'
            elif method == 'fixed':
                if dataset_name == 'CMRE':
                    examples_str = fixed_examples_cmre
                elif dataset_name == 'CSR':
                    examples_str = fixed_examples_csr
                else:
                    raise ValueError("Dataset name not supported")
            else:
                examples = retriever.search_examples(d["input"], selected_k=selected_k)
                for id, example in enumerate(examples):
                    examples_str += f'示例 {id + 1}:\n输入："{example[0]}"\n输出："{example[1]}"\n'

            if dataset_name == 'CMRE':
                prompt = cmre_no_sft_template.format(example=examples_str, input=d["input"])
            elif dataset_name == 'CSR':
                prompt = csr_no_sft_template.format(example=examples_str, input=d["input"])
            else:
                raise ValueError("Unknown dataset name")
            d['instruction'] = prompt

    # 直接将数据保存到对应位置
    with open(inst_save_path, 'w', encoding='UTF-8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)

    return data


def Inference(args):
    paths = {
        "bert_path": args.bert_path,
        "gnn_path": args.gnn_path,
        "sbert_path": args.sbert_path
    }
    # 处理数据集/读取数据集
    if args.do_ficl:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        if os.path.exists(args.inst_save_path):
            print(f"===========================Retrieve mode is {args.method}, Constructing instruction...===========================")
            print("The processed test data already exists, skip encoding...")
            with open(args.inst_save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"=======================================================================================================")
        else:
            print(f"===========================Retrieve mode is {args.method}, Constructing instruction...===========================")
            data = construct_prompt(args.dataset_path, args.inst_save_path, args.dataset_name,
                                    paths=paths, ex_file_path=args.ex_file_path, method=args.method,
                                    selected_k=args.selected_k)
            print(f"==============================Instruction constructed successfully!====================================")
    torch.cuda.empty_cache()
    # 加载LLM模型
    model = Model(model_name_or_path=args.model_name_or_path, template=args.template)

    all_results = [] # 用于保存全部记录
    y_trues, y_preds = [], []
    right, wrong = 0, 0 # 用于统计正确和错误的示例数量
    for data_entry in tqdm(data, desc="Reasoning..."):
        if args.do_ficl:
            if args.dataset_name == 'CMRE':
                instruction = cmre_ficl_template.format(input=input)
            elif args.dataset_name == 'CSR':
                instruction = csr_ficl_template.format(input=input)
            else:
                raise ValueError("Unknown dataset name")
            message = [
                {"role": "user", "content": instruction}
            ]
            true = data_entry['output']
            pred = model.generate(message)[0].response_text
            pred = str(parser_output_str(pred, args.dataset_name))

            if pred == true:
                feedback = "你对了！请保持状态，继续前进。"
                right += 1
            else:
                feedback = "你错了！请确保你的预测是准确的。"
                wrong += 1
            if args.dataset_name == 'CSR':
                all_results.append({
                    'id': data_entry['id'],
                    'instruction': instruction,
                    'input': data_entry['input'],
                    'output': true,
                    'predict': pred,
                    'feedback': feedback
                })
            elif args.dataset_name == 'CMRE':
                all_results.append({
                    'id': data_entry['id'],
                    'instruction': instruction,
                    'input': data_entry['input'],
                    'output': true,
                    'predict': pred,
                    'feedback': feedback,
                    'spans_type': data_entry['spans_type']
                })
            else:
                raise ValueError("Unknown dataset name")
        else:
            instruction = data_entry['instruction']
            message = [
                {"role": "user", "content": instruction}
            ]
            true = data_entry['output']
            pred_original = model.generate(message)[0].response_text
            pred = str(parser_output_str(pred_original, args.dataset_name))

            try:
                pred_list = parse_output(pred)
                true_list = parse_output(true)
            except Exception as e:
                pred = str(e)
                pred_list = []

            y_trues.append(true_list)
            y_preds.append(pred_list)
            all_results.append({
                'id': data_entry['id'],
                'instruction': instruction,
                'input': data_entry['input'],
                'output': true,
                'predict': pred
            })
    if args.do_ficl:
        print(f"=================================={args.dataset_name}-{args.dataset_type} Feedback Results==================================")
        print(f"Inference completed, Right Feedback Total: {right}, Wrong Feedback Total: {wrong}")
        print(f"Right Rate: {right / (right + wrong):.2%}, Wrong Rate: {wrong / (right + wrong):.2%}")
        print(f"=================================={args.dataset_name}-{args.dataset_type} Feedback Results==================================")
    else:
        if args.dataset_name == 'CMRE':
            precision, recall, f1 = get_cmre_metric(y_trues, y_preds, report=False)
        elif args.dataset_name == 'CSR':
            precision, recall, f1 = get_csr_metric(y_trues, y_preds, report=False)
        else:
            raise ValueError(f"Unknown dataset_name: {args.dataset_name}")
        print(f"==================================Retrieve mode-{args.method} Dataset-{args.dataset_name} Results==================================")
        print(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
        print(f"==================================Retrieve mode-{args.method} Dataset-{args.dataset_name} Results==================================")
    with open(args.results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference on LLM for feedback label")
    parser.add_argument("--model_name_or_path", type=str, help="The path of the LLM")
    parser.add_argument("--gnn_path", type=str, help="The path of the GNN model")
    parser.add_argument("--bert_path", type=str, help="The path of the BERT model")
    parser.add_argument("--sbert_path", type=str, help="The path of all-MiniLM-L6-v2")
    parser.add_argument("--template", type=str, help="The template for the LLM")
    parser.add_argument("--method", type=str, help="The method of retrieval examples")
    parser.add_argument("--dataset_name", type=str, help="The name of the dataset")
    parser.add_argument("--dataset_type", type=str, help="The type of the dataset(train or test)")
    parser.add_argument("--dataset_path", type=str, help="The path of the dataset")
    parser.add_argument("--inst_save_path", type=str, help="The path of the instruction data save")
    parser.add_argument("--ex_file_path", type=str, help="The path of the examples file")
    parser.add_argument("--results_path", type=str, help="The path of the output results")
    parser.add_argument("--selected_k", type=int, help="The number of selected examples")
    parser.add_argument("--do_ficl", action='store_true', help="Whether to do feedback instruction construction")
    args = parser.parse_args()
    # 手动设置参数
    if args.do_ficl:
        args.results_path = f"./data/{args.dataset_name}/{args.dataset_type}_ficl.json"
    else:
        args.results_path = f"./results/{args.dataset_name}/{args.method}/nosft_llama3_results.json"
        args.gnn_path = f"./checkpoints/gnn_checkpoints/{args.dataset_name}/best_gnn_model.pt"
        args.inst_save_path = f"./data/{args.dataset_name}/Instruction/nosft_inst_{args.method.lower()}.json"
        if args.method == 'gnn':
            args.ex_file_path = f"./data/{args.dataset_name}/train_ficl.json"
        else:
            args.ex_file_path = f"./data/{args.dataset_name}/train.json"
    args.dataset_path = f"./data/{args.dataset_name}/{args.dataset_type}.json"
    Inference(args)
