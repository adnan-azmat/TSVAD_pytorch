DATA_PATH="/home/users/ntu/adnan002/scratch/data/DIHARD3"
OUTPUT_PATH="exps/d8_mse_eval_model"

# MODEL_PATH="{Paste the path of the model you want to evaluate}"

python main.py \
    --train_list ${DATA_PATH}/third_dihard_challenge_dev/data/ts_infer.json \
    --eval_list ${DATA_PATH}/third_dihard_challenge_eval/data/ts_infer.json \
    --train_path ${DATA_PATH}/third_dihard_challenge_dev/data \
    --eval_path ${DATA_PATH}/third_dihard_challenge_eval/data \
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
    --max_speaker 8