#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_PATH="/home/users/ntu/adnan002/scratch/data/DIHARD3_2"
BASE_MODEL_PATH="/home/users/ntu/adnan002/scratch/repos/Adnan/2/TSVAD_pytorch/ts-vad/exps/p2_4ngpus/model"
BASE_OUTPUT_PATH="exps/eval_p2_4ngpus_dihard2"

# Loop through models 1 to 16
for i in $(seq -f "%04g" 1 16); do
    MODEL_PATH="${BASE_MODEL_PATH}/model_${i}.model"
    OUTPUT_PATH="${BASE_OUTPUT_PATH}/model_${i}"
    
    echo "Evaluating model: ${MODEL_PATH}"
    echo "Output path: ${OUTPUT_PATH}"
    
    python main.py \
        --train_list ${DATA_PATH}/third_dihard_challenge_dev/data/ts_infer.json \
        --train_path ${DATA_PATH}/third_dihard_challenge_dev/data/ \
        --eval_list ${DATA_PATH}/third_dihard_challenge_eval/data/ts_infer.json \
        --eval_path ${DATA_PATH}/third_dihard_challenge_eval/data/ \
        --musan_path /home/users/ntu/adnan002/scratch/data/augment/musan \
        --rir_path /home/users/ntu/adnan002/scratch/data/augment/RIRS_NOISES/simulated_rirs \
        --save_path ${OUTPUT_PATH} \
        --rs_len 4 \
        --test_shift 4 \
        --min_silence 0.32 \
        --min_speech 0.00 \
        --threshold 0.50 \
        --n_cpu 14 \
        --eval \
        --init_model ${MODEL_PATH} \
        --max_speaker 2
done