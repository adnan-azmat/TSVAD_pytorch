DATA_PATH="/home/users/ntu/adnan002/scratch/data/DIHARD3_denoise"
OUTPUT_PATH="exps/d8_denoise_ft_model_eval_wav_denoise"
MODEL_PATH="/home/users/ntu/adnan002/scratch/repos/Adnan/2/TSVAD_pytorch/ts-vad/pretrained_models/model_0019_finetuned.model"

python main.py \
    --train_list ${DATA_PATH}/third_dihard_challenge_dev/data/ts_infer.json \
    --train_path ${DATA_PATH}/third_dihard_challenge_dev/data \
    --eval_list ${DATA_PATH}/third_dihard_challenge_eval/data/ts_infer.json \
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