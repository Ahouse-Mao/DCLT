#!/usr/bin/env bash
set -euo pipefail

# Fixed pretrain_weather_v4.sh
# - 修复原脚本中不完整/错误的 for-loop 语法
# - 使用正确的 argparse 参数名（例如 --epochs 而非 --epoch）
# - 在每组预训练后运行 run_longExp_v4.py 并把日志写入 logs/LongForecasting

model_name=PatchTST_pretrained_v4

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021
seq_len=96

# 参数集合（按需修改）
epochs_list=(100)
patch_lens=(16)
patch_strides=(2)
max_train_lengths=(600)
repr_dims_list=(256)
pred_lens=(96)

mkdir -p logs/LongForecasting

for dataset in weather; do
    for epoch in "${epochs_list[@]}"; do
        for patch_len in "${patch_lens[@]}"; do
            for patch_stride in "${patch_strides[@]}"; do
                for max_train_len in "${max_train_lengths[@]}"; do
                    for repr_dims in "${repr_dims_list[@]}"; do
                        echo "=== Pretrain: dataset=$dataset epoch=$epoch patch_len=$patch_len patch_stride=$patch_stride max_train_len=$max_train_len repr_dims=$repr_dims ==="
                        TS=$(date +%d_%H_%M)
                        python -u ./DCLT_main_pretrain_v4.py \
                            --dataset "$dataset" \
                            --epochs "$epoch" \
                            --patch_len "$patch_len" \
                            --patch_stride "$patch_stride" \
                            --max-train-length "$max_train_len" \
                            --repr-dims "$repr_dims" \
                            --TS "$TS"

                        # 预训练完成后，针对若干 pred_len 运行长预测评估
                        for pred_len in "${pred_lens[@]}"; do
                            model_id="${model_id_name}_${seq_len}_${pred_len}"
                            logfile="./checkpoints/${model_id_name}/${TS}_${epoch}_${patch_len}_${patch_stride}_${max_train_len}_${repr_dims}/${model_name}_${TS}_${model_id}_${patch_len}_${patch_stride}_${repr_dims}.log"
                            echo ">>> Run forecasting: pred_len=${pred_len}  (log: ${logfile})"
                            python -u ./PatchTST_supervised/run_longExp_v4.py \
                                --random_seed $random_seed \
                                --is_training 1 \
                                --root_path "$root_path_name" \
                                --data_path "$data_path_name" \
                                --model_id "$model_id" \
                                --model "$model_name" \
                                --data "$data_name" \
                                --features M \
                                --seq_len $seq_len \
                                --pred_len $pred_len \
                                --enc_in 21 \
                                --e_layers 3 \
                                --n_heads 16 \
                                --d_model $repr_dims \
                                --d_ff 256 \
                                --dropout 0.2 \
                                --fc_dropout 0.2 \
                                --head_dropout 0 \
                                --patch_len $patch_len \
                                --stride $patch_stride \
                                --des 'Exp' \
                                --train_epochs 100 \
                                --patience 20 \
                                --itr 1 --batch_size 128 --learning_rate 0.0001 >"${logfile}"
                        done
                    done
                done
            done
        done
    done
done
