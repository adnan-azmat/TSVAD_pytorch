DATA_PATH="/home/users/ntu/adnan002/scratch/data/klass_v2/recorded_corpusV2/single"
OUTPUT_PATH="exps/eval_klass_v2_4spk_amitrain"
# MODEL_PATH="/home/users/ntu/adnan002/scratch/repos/Adnan/2/TSVAD_pytorch/ts-vad/pretrained_models/model_0019_finetuned.model"

MODEL_PATH="/home/users/ntu/adnan002/scratch/repos/Adnan/2/TSVAD_pytorch/ts-vad/exps/p418_voxconversedev17_voxconversetest8_msd5_amitrain/model/model_0012.model"

python main.py \
    --train_list ${DATA_PATH}/ts_infer.json \
    --train_path ${DATA_PATH} \
    --eval_list ${DATA_PATH}/ts_infer.json \
    --eval_path ${DATA_PATH} \
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
    --max_speaker 4