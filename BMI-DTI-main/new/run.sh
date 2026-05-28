
#!/bin/bash

# cd /mnt/ST8000/xiaaobo/MY DTI best

# conda activate base

model_name=AttentionDTA
cuda_id=0

for lr in 5e-6 1e-5 3e-5 1e-4
do
    for train_batch_size in 32 64 
    do
    CUDA_VISIBLE_DEVICES=${cuda_id} python AttentionDTA_main.py \
        --data_path datasets \
        --gradient_accumulation_steps 1 \
        --dataset Davis \
        --seed 4321 \
        --k_fold 5 \
        --lr ${lr} \
        --warmup_steps 500 \
        --train_batch_size ${train_batch_size} \
        --patience 20 \
        --epochs 80
    done
done