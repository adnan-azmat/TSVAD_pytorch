export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_PATH="/home/users/ntu/adnan002/scratch/data/v2_16khz_librispeech_simulated_data"
DATA_PATH2="/home/users/ntu/adnan002/scratch/data/DIHARD3"

OUTPUT_PATH="exps/p8_4gpus"

python main.py \
    --train_list ${DATA_PATH}/ts_infer.json \
    --train_path ${DATA_PATH} \
    --eval_list ${DATA_PATH2}/third_dihard_challenge_eval/data/ts_infer.json \
    --eval_path ${DATA_PATH2}/third_dihard_challenge_eval/data \
    --musan_path /home/users/ntu/adnan002/scratch/data/augment/musan \
    --rir_path /home/users/ntu/adnan002/scratch/data/augment/RIRS_NOISES/simulated_rirs \
    --save_path ${OUTPUT_PATH} \
    --max_speaker 8 \
    --warm_up_epoch 5 \
    --batch_size 160 \
    --rs_len 4 \
    --test_shift 4 \
    --lr 0.0001 \
    --test_step 1 \
    --max_epoch 40 \
    --train




# export CUDA_VISIBLE_DEVICES=0,1,2,3
# DATA_PATH="/home/users/ntu/adnan002/scratch/data/DIHARD3"
# OUTPUT_PATH="exps/d8_4gpus"

# python main.py \
#     --train_list ${DATA_PATH}/third_dihard_challenge_dev/data/ts_infer.json \
#     --train_path ${DATA_PATH}/third_dihard_challenge_dev/data \
#     --eval_list ${DATA_PATH}/third_dihard_challenge_eval/data/ts_infer.json \
#     --eval_path ${DATA_PATH}/third_dihard_challenge_eval/data \
#     --musan_path /home/users/ntu/adnan002/scratch/data/augment/musan \
#     --rir_path /home/users/ntu/adnan002/scratch/data/augment/RIRS_NOISES/simulated_rirs \
#     --save_path ${OUTPUT_PATH} \
#     --max_speaker 8 \
#     --warm_up_epoch 5 \
#     --batch_size 160 \
#     --rs_len 4 \
#     --test_shift 4 \
#     --lr 0.0001 \
#     --test_step 1 \
#     --max_epoch 40 \
#     --train