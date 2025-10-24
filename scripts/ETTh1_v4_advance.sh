#!/usr/bin/env bash
# pretrain_weather_v4_run.sh
# 功能：
#  - 单次训练/评估失败后继续执行下一个实验（不会中断整个脚本）
#  - 若目标 log 文件已存在则跳过该实验（重名侦测）
#  - 自动创建日志目录并在失败时把 exit code 写入 log 文件末尾

set -u -o pipefail

model_name=PatchTST_pretrained_v4

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=custom

random_seed=2021
seq_len=96
default_batch_size=16

# 参数集合（按需修改）
epochs_list=(20 20 200 200)
tau_temp=(0.5)
patch_lens=(16)
patch_strides=(2)
max_train_lengths=(336)
repr_dims_list=(256)
pred_lens=(96 336)

# 原子化的 run_and_continue：第一个参数必须是 logfile
# 通过创建 lockdir 原子化判重（只有第一个能拿到锁），拿到锁后创建父目录并执行命令，
# 执行结束（或失败）会删除锁。
# run_and_continue_atomic() {
#     local logfile="$1"; shift
#     local lockdir="${logfile}.lockdir"

#     # 先确保日志父目录存在（否则后面重定向会失败）
#     mkdir -p "$(dirname "${logfile}")"

#     # 原子尝试拿锁
#     if ! mkdir "${lockdir}" 2>/dev/null; then
#         echo "SKIP: logfile exists or locked by other process: ${logfile}"
#         return 0
#     fi

#     # 确保释放锁（无论如何都删除 lockdir）
#     # 使用 subshell 的 trap 以免影响外层脚本的 trap
#     (
#         trap 'rm -rf "${lockdir}"' EXIT

#         echo "RUNNING: ${*}  -> ${logfile}"
#         # 执行并把 stdout/stderr 写到 logfile
#         "$@" >"${logfile}" 2>&1
#         local rc=$?
#         if [[ ${rc} -ne 0 ]]; then
#             echo -e "\n=== FAILED with exit code ${rc} ===" >>"${logfile}"
#             echo "ERROR: command failed with exit code ${rc}. See ${logfile}"
#         else
#             echo "DONE: ${logfile}"
#         fi
#         return ${rc}
#     )
#     # 子 Shell 执行完毕，锁已删除（trap）
#     return $?
# }

# 主循环
for dataset in ETTh1; do
    for epoch in "${epochs_list[@]}"; do
        for tau in "${tau_temp[@]}"; do
            for patch_len in "${patch_lens[@]}"; do
                for patch_stride in "${patch_strides[@]}"; do
                    for max_train_len in "${max_train_lengths[@]}"; do
                        for repr_dims in "${repr_dims_list[@]}"; do
                            batch_size=${default_batch_size}
                            if (( max_train_len > 600 )); then
                                batch_size=8
                            fi

                            echo "=== Pretrain: dataset=${dataset} epoch=${epoch} patch_len=${patch_len} patch_stride=${patch_stride} max_train_len=${max_train_len} repr_dims=${repr_dims} ==="
                            # 避免同分钟内冲突
                            TS=$(date +%d_%H)
                            # 定义 pretrain_log（必须在调用前定义）
                            # pretrain_log="./logs/pretrain/${model_name}_${dataset}_ep${epoch}_pl${patch_len}_ps${patch_stride}_mtl${max_train_len}_rd${repr_dims}.log"

                            # run_and_continue_atomic "${pretrain_log}" 
                            python -u ./DCLT_main_pretrain_v4.py \
                                --batch_size "${batch_size}" \
                                --dataset "${dataset}" \
                                --epochs "${epoch}" \
                                --tau_temp "${tau}" \
                                --patch_len "${patch_len}" \
                                --patch_stride "${patch_stride}" \
                                --max-train-length "${max_train_len}" \
                                --repr-dims "${repr_dims}" \
                                --TS "${TS}"
                            # rc_pretrain=$?
                            # if [[ ${rc_pretrain} -ne 0 ]]; then
                            #     echo "WARN: pretrain failed for TS=${TS}, skipping forecasting for this TS."
                            #     # 跳过本次配置的后续预测，继续下一个 repr_dims（或下一个循环）
                            #     continue
                            # fi

                            # 预训练成功后，针对若干 pred_len 运行长预测评估（单个日志文件命名避免覆盖）
                            for pred_len in "${pred_lens[@]}"; do
                                model_id="${model_id_name}_${seq_len}_${pred_len}"
                                logfile="./checkpoints/${model_id_name}/${TS}_${epoch}_${patch_len}_${patch_stride}_${max_train_len}_${repr_dims}/${model_name}_${TS}_${model_id}_16_8_${repr_dims}.log"

                                # 运行评估并自动跳过已有日志
                                # run_and_continue_atomic "${logfile}" 
                                mkdir -p "$(dirname "${logfile}")"
                                python -u ./PatchTST_supervised/run_longExp_v4.py \
                                    --pretrain_folder "${TS}_${epoch}_${patch_len}_${patch_stride}_${max_train_len}_${repr_dims}" \
                                    --random_seed "${random_seed}" \
                                    --is_training 1 \
                                    --root_path "${root_path_name}" \
                                    --data_path "${data_path_name}" \
                                    --model_id "${model_id}" \
                                    --model "${model_name}" \
                                    --data "${data_name}" \
                                    --features M \
                                    --seq_len "${seq_len}" \
                                    --pred_len "${pred_len}" \
                                    --enc_in 7 \
                                    --e_layers 3 \
                                    --n_heads 4 \
                                    --d_model "${repr_dims}" \
                                    --d_ff 256 \
                                    --dropout 0.3 \
                                    --fc_dropout 0.3 \
                                    --head_dropout 0 \
                                    --patch_len 16 \
                                    --stride 8 \
                                    --des 'Exp' \
                                    --train_epochs 100 \
                                    --patience 20 \
                                    --itr 1 --batch_size 512 --learning_rate 0.0001 >"${logfile}" 2>&1
                                rc_pretrain_eval=$?
                                if [[ ${rc_pretrain_eval} -ne 0 ]]; then
                                    echo -e "\n=== FAILED with exit code ${rc_pretrain_eval} ===" >>"${logfile}"
                                fi
                                # 不需要在这里中断脚本 — run_and_continue 会记录失败并返回 non-zero，但循环会继续
                            done

                            # for pred_len in "${pred_lens[@]}"; do
                            #     model_id="${model_id_name}_${seq_len}_${pred_len}"
                            #     logfile="./checkpoints/${model_id_name}/${TS}_${epoch}_${patch_len}_${patch_stride}_${max_train_len}_${repr_dims}/${model_name}_${TS}_${model_id}_16_2_${repr_dims}.log"

                            #     # 运行评估并自动跳过已有日志
                            #     # run_and_continue_atomic "${logfile}" 
                            #     mkdir -p "$(dirname "${logfile}")"
                            #     python -u ./PatchTST_supervised/run_longExp_v4.py \
                            #         --pretrain_folder "${TS}_${epoch}_${patch_len}_${patch_stride}_${max_train_len}_${repr_dims}" \
                            #         --random_seed "${random_seed}" \
                            #         --is_training 1 \
                            #         --root_path "${root_path_name}" \
                            #         --data_path "${data_path_name}" \
                            #         --model_id "${model_id}" \
                            #         --model "${model_name}" \
                            #         --data "${data_name}" \
                            #         --features M \
                            #         --seq_len "${seq_len}" \
                            #         --pred_len "${pred_len}" \
                            #         --enc_in 7 \
                            #         --e_layers 3 \
                            #         --n_heads 16 \
                            #         --d_model "${repr_dims}" \
                            #         --d_ff 256 \
                            #         --dropout 0.2 \
                            #         --fc_dropout 0.2 \
                            #         --head_dropout 0 \
                            #         --patch_len 16 \
                            #         --stride 2 \
                            #         --des 'Exp' \
                            #         --train_epochs 100 \
                            #         --patience 20 \
                            #         --itr 1 --batch_size 256 --learning_rate 0.0001 >"${logfile}" 2>&1
                            #     rc_pretrain_eval=$?
                            #     if [[ ${rc_pretrain_eval} -ne 0 ]]; then
                            #         echo -e "\n=== FAILED with exit code ${rc_pretrain_eval} ===" >>"${logfile}"
                            #     fi
                                # 不需要在这里中断脚本 — run_and_continue 会记录失败并返回 non-zero，但循环会继续
                            done

                        done
                    done
                done
            done
        done
    done
done

echo "All experiments finished (some may have failed or been skipped)."
