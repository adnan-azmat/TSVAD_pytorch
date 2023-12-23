python main.py \
--train_list data/DIHARD3/third_dihard_challenge_dev/data/ts_dev.json \
--eval_list data/DIHARD3/third_dihard_challenge_eval/data/ts_eval.json \
--train_path data/DIHARD3/third_dihard_challenge_dev/data \
--eval_path data/DIHARD3/third_dihard_challenge_eval/data \
--save_path exps/res23 \
--warm_up_epoch 10 \
--batch_size 40 \
--rs_len 4 \
--test_shift 4 \
--lr 0.0001 \
--test_step 1 \
--max_epoch 40 \
--train