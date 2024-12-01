
# Train Runing
python ./src/main.py \
    --method gnn \
    --selected_k 4\
    --dataset_name CSR \
    --dataset_dir ./data \
    --bert_path xxxx/bert-base-chinese \
    --do_train \
    --seed 42 \
    --model_name_or_path xxxx/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3 \
    --finetuning_type lora \
    --template llama3 \
    --lora_target all \
    --lora_rank 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 100 \
    --save_steps 300 \
    --val_size 0.1 \
    --learning_rate 8e-5 \
    --num_train_epochs 3.0 \
    --fp16

# Test Running
python ./src/main.py \
    --method gnn \
    --selected_k 4\
    --dataset_name CSR \
    --dataset_dir ./data \
    --bert_path xxxx/bert-base-chinese \
    --model_name_or_path xxxx/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3 \
    --finetuning_type lora \
    --template llama3 \
    --output_dir ./checkpoints/fine_tuning_checkpoints/CSR/llama3/gnn

# Train Runing
python ./src/main.py \
    --method gnn \
    --selected_k 4\
    --dataset_name CMRE \
    --dataset_dir ./data \
    --bert_path xxxx/bert-base-chinese \
    --do_train \
    --seed 42 \
    --model_name_or_path xxxx/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3 \
    --finetuning_type lora \
    --template llama3 \
    --lora_target all \
    --lora_rank 64 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 100 \
    --save_steps 300 \
    --val_size 0.1 \
    --learning_rate 8e-5 \
    --num_train_epochs 3.0 \
    --fp16

# Test Running
python ./src/main.py \
    --method gnn \
    --selected_k 4\
    --dataset_name CMRE \
    --dataset_dir ./data \
    --bert_path xxxx/bert-base-chinese \
    --model_name_or_path xxxx/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3 \
    --finetuning_type lora \
    --template llama3 \
    --output_dir ./checkpoints/fine_tuning_checkpoints/CMRE/llama3/gnn