# main.py
import argparse
import torch
import time
import os
import warnings

from Icl.data_utils import train_data_process, test_data_process
from Icl.evaluation import evaluate
from train import run_exp

warnings.filterwarnings("ignore")

train_params_list = [
    "do_train",
    "seed",
    "model_name_or_path",
    "template",
    "lora_target",
    "dataset",
    "dataset_dir",
    "finetuning_type",
    "save_safetensors",
    "lora_rank",
    "output_dir",
    "overwrite_output_dir",
    "overwrite_cache",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
    "evaluation_strategy",
    "logging_steps",
    "save_steps",
    "save_total_limit",
    "val_size",
    "learning_rate",
    "num_train_epochs",
    "load_best_model_at_end",
    "bf16",
    "fp16",
    "plot_loss",
    "ddp_find_unused_parameters",
    "packing"
]

def init_args():
    parser = argparse.ArgumentParser(description="Run training and other operations.")
    
    train_args = parser.add_argument_group('train_args', 'Arguments for train_bash.py')
    
    # 添加train_bash.py所需的参数到 train_args 组
    train_args.add_argument("--do_train", action="store_true")
    train_args.add_argument("--seed", type=int, default=42)
    train_args.add_argument("--model_name_or_path", type=str)
    train_args.add_argument("--template", type=str)
    train_args.add_argument("--lora_target", type=str, default="all")
    train_args.add_argument("--dataset", type=str)
    train_args.add_argument("--dataset_dir", type=str, default="./data")
    train_args.add_argument("--finetuning_type", type=str, default="lora")
    train_args.add_argument("--lora_rank", type=int, default=64)
    train_args.add_argument("--output_dir", type=str, default="")
    train_args.add_argument("--overwrite_output_dir", action="store_true", default=True)
    train_args.add_argument("--overwrite_cache", action="store_true")
    train_args.add_argument("--per_device_train_batch_size", type=int, default=4)
    train_args.add_argument("--per_device_eval_batch_size", type=int, default=4)
    train_args.add_argument("--gradient_accumulation_steps", type=int, default=2)
    train_args.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train_args.add_argument("--evaluation_strategy", type=str, default="steps")
    train_args.add_argument("--logging_steps", type=int, default=20)
    train_args.add_argument("--save_steps", type=int, default=400)
    train_args.add_argument("--save_total_limit", type=int, default=2)
    train_args.add_argument("--val_size", type=float, default=0.1)
    train_args.add_argument("--save_safetensors", action="store_true", default=False)
    train_args.add_argument("--learning_rate", type=float, default=8e-5)
    train_args.add_argument("--num_train_epochs", type=float, default=3.0)
    train_args.add_argument("--load_best_model_at_end", action="store_true")
    train_args.add_argument("--bf16", action="store_true")
    train_args.add_argument("--fp16", action="store_true")
    train_args.add_argument("--plot_loss", action="store_true")
    train_args.add_argument("--ddp_find_unused_parameters", action="store_true")
    train_args.add_argument("--packing", action="store_true", default=True)

    data_args = parser.add_argument_group('main_args', 'Arguments for main.py')
    # 添加main.py所需的参数到 data_args 组
    data_args.add_argument("--method", type=str, default="gnn")
    data_args.add_argument("--dataset_name", type=str, help="The name of the dataset")
    data_args.add_argument("--selected_k", type=int, help="The number of selected in-context examples")
    data_args.add_argument("--bert_path", type=str, help="The path to the pre-trained BERT model")
    data_args.add_argument("--sbert_path", type=str, help="The path to all-MiniLM-L6-v2")
    data_args.add_argument("--gnn_path", type=str, help="The path to the trained GNN model")
    data_args.add_argument("--prefix", type=str, help="path to the data directory")
    data_args.add_argument("--inst_prefix", type=str, help="path to the directory that you want to store instructions")
    data_args.add_argument("--results_path_prefix", type=str, help="path to save predict results")

    args = parser.parse_args()
    # 使用字典推导式从 args 中提取 train_args 和 data_args 的参数
    train_params = {param: getattr(args, param) for param in args.__dict__ if param in train_params_list}
    data_params = {param: getattr(args, param) for param in args.__dict__ if param not in train_params_list}

    # 手动添加参数
    if 'dataset' not in train_params.keys() or train_params["dataset"] is None:
        train_params["dataset"] = f'{data_params["dataset_name"].lower()}_{data_params["method"].lower()}_inst_train'
    if 'output_dir' not in train_params.keys() or train_params["output_dir"] is None or train_params["output_dir"] == "":
        train_params["output_dir"] = f'./checkpoints/fine_tuning_checkpoints/{data_params["dataset_name"]}/{train_params["template"]}/{data_params["method"]}'
    if 'gnn_path' not in data_params.keys() or data_params["gnn_path"] is None:
        data_params["gnn_path"] = f'./checkpoints/gnn_checkpoints/{data_params["dataset_name"]}/best_gnn_model.pt'
    data_params["is_ICL"] = False if data_params["method"] == "noicl" else True
    data_params["results_path_prefix"] = f'./results/{data_params["dataset_name"]}/{data_params["method"]}/'
    data_params["prefix"] = f'./data/{data_params["dataset_name"]}/'
    data_params["inst_prefix"] = f'./data/{data_params["dataset_name"]}/Instruction/'

    return train_params, data_params
    
def main():
    train_params, data_params = init_args()
    # 加载测试数据
    paths = {
        "bert_path": data_params["bert_path"],
        "gnn_path": data_params["gnn_path"],
        "sbert_path": data_params["sbert_path"]
    }

    if train_params["do_train"]:
        print("\n===========================================================================================================")
        print(f"Current retrieving mode is {data_params['method']}, starting encoding... Now gnn model is {data_params['gnn_path']}")
        print("===========================================================================================================\n")

        if os.path.exists(f"{data_params['inst_prefix']}{data_params['method'].lower()}_inst_train.json") and data_params['method'] in ['mmr', 'KATE', 'sbert']:
            print("The processed training data already exists, skip encoding...")
        else:
            train_data_process(method=data_params['method'], ICL=data_params['is_ICL'], paths=paths,
                               prefix=data_params["prefix"], inst_prefix=data_params["inst_prefix"],
                               dataset_name=data_params["dataset_name"], selected_k=data_params["selected_k"])

        print("\n========================================")
        print("Start training...")
        print("========================================\n")

        # 开始训练模型
        run_exp(train_params)

        torch.cuda.empty_cache()
    else:
        print("\n======================================================================================================================================================================")
        print(f"Start evaluating..., Current retrieving mode is {data_params['method']}, Now evaluating LLM is {train_params['output_dir']}, gnn model is {data_params['gnn_path']}")
        print("======================================================================================================================================================================\n")
        test_data = test_data_process(method=data_params['method'], ICL=data_params['is_ICL'], paths=paths,
                                      prefix=data_params["prefix"], inst_prefix=data_params["inst_prefix"],
                                      dataset_name=data_params["dataset_name"], selected_k=data_params["selected_k"])
        p, r, f1 = evaluate(test_data, model_name_or_path=train_params["model_name_or_path"], checkpoint_dir=train_params["output_dir"],
                            template=train_params["template"], results_path_prefix=data_params["results_path_prefix"],
                            temperature=0.1, top_p=0.9, finetuning_type=train_params["finetuning_type"], dataset_name=data_params["dataset_name"])
        print(f'precision: {100. * p:.2f}%; recall: {100. * r:.2f}%; f1: {100. * f1:.2f}%')
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()