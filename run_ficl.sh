
python ./src/predict_used_LLM.py \
    --model_name_or_path xxxx/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3 \
    --template llama3 \
    --dataset_name CSR \
    --dataset_type train

python ./src/predict_used_LLM.py \
    --model_name_or_path xxxx/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3 \
    --template llama3 \
    --dataset_name CSR \
    --dataset_type test

python ./src/predict_used_LLM.py \
    --model_name_or_path xxxx/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3 \
    --template llama3 \
    --dataset_name CMRE \
    --dataset_type train

python ./src/predict_used_LLM.py \
    --model_name_or_path xxxx/ChineseAlpacaGroup/llama-3-chinese-8b-instruct-v3 \
    --template llama3 \
    --dataset_name CMRE \
    --dataset_type test

