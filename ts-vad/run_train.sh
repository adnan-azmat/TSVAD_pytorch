python /home/users/ntu/adnan002/scratch/repos/TSVAD_pytorch/ts-vad/main.py \
--train_list /home/users/ntu/adnan002/scratch/data/v2_simulated_data_Switchboard_SRE_small_16k/data/simu3/data/all_files/all_simtrain.json \
--eval_list /home/users/ntu/adnan002/scratch/data/DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path /home/users/ntu/adnan002/scratch/data/v2_simulated_data_Switchboard_SRE_small_16k/data/simu3/data/all_files \
--eval_path /home/users/ntu/adnan002/scratch/data/DIHARD3/third_dihard_challenge_eval/data \
--save_path exps/res23 \
--warm_up_epoch 10 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--test_step 1 \
--max_epoch 40 \
--train