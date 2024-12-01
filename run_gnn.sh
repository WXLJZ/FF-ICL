python src/gnnencoder/train.py \
    --bert_model_path xxx/models/bert-base-chinese \
    --dataset_name CMRE \
    --batch_size 128 \
    --max_len 128 \
    --lig_top_k 0.2 \
    --struct_top_k 0.2 \
    --epoch_num 12 \
    --lr 1e-4

python src/gnnencoder/train.py \
    --bert_model_path xxx/models/bert-base-chinese \
    --dataset_name CSR \
    --batch_size 128 \
    --max_len 128 \
    --lig_top_k 0.2 \
    --struct_top_k 0.2 \
    --epoch_num 12 \
    --lr 1e-4
