DATA_PATH="/home/users/ntu/adnan002/scratch/data/DIHARD3_denoise"
OUTPUT_PATH="exps/d8_denoise_ft_model"

python main.py \
    --train_list ${DATA_PATH}/third_dihard_challenge_dev/data/ts_infer.json \
    --train_path ${DATA_PATH}/third_dihard_challenge_dev/data \
    --eval_list ${DATA_PATH}/third_dihard_challenge_eval/data/ts_infer.json \
    --eval_path ${DATA_PATH}/third_dihard_challenge_eval/data \
    --musan_path /home/users/ntu/adnan002/scratch/data/augment/musan \
    --rir_path /home/users/ntu/adnan002/scratch/data/augment/RIRS_NOISES/simulated_rirs \
    --save_path ${OUTPUT_PATH} \
    --max_speaker 8 \
    --warm_up_epoch 5 \
    --batch_size 40 \
    --rs_len 4 \
    --test_shift 4 \
    --lr 0.0001 \
    --test_step 1 \
    --max_epoch 41 \
    --train
