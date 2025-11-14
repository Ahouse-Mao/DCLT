if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PCLE_v4

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

#random_seed=2021


TS=$(date +%y_%d_%H_%M)
for random_seed in 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031; do
for pred_len in 192 336 720; do
    for cl_weight in 0.01; do
        for pcle_outdims in 256; do
            for stride in 8; do
                python -u run_longExp_v4.py \
                --random_seed $random_seed \
                --is_training 1 \
                --root_path $root_path_name \
                --data_path $data_path_name \
                --model_id $model_id_name_$seq_len'_'$pred_len \
                --model $model_name \
                --data $data_name \
                --features M \
                --seq_len $seq_len \
                --pred_len $pred_len \
                --cl_weight $cl_weight \
                --pcle_outdims $pcle_outdims \
                --pcle_proj_hidden_dims 64 \
                --tau_temp 0.5 \
                --enc_in 7 \
                --e_layers 3 \
                --n_heads 4 \
                --d_model 16 \
                --d_ff 128 \
                --dropout 0.3\
                --fc_dropout 0.3\
                --head_dropout 0\
                --patch_len 16\
                --stride $stride\
                --enable_cross_attn True \
                --des 'Exp' \
                --train_epochs 100\
                --patience 20\
                --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/LongForecasting/$TS'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$cl_weight'_'$pcle_outdims'_'$stride'_'$random_seed.log 
            done
        done
    done
done
done

