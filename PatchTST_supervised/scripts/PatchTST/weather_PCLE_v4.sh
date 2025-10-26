if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=PCLE_v4

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

random_seed=2021
random_seed2=2025

TS=$(date +%y_%d_%H_%M)

for pred_len in 96 336; do
    for cl_weight in 0.01; do
        for pcle_outdims in 512; do
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
                --enc_in 21 \
                --e_layers 3 \
                --n_heads 16 \
                --d_model 128 \
                --d_ff 256 \
                --dropout 0.2\
                --fc_dropout 0.2\
                --head_dropout 0\
                --patch_len 16\
                --stride $stride\
                --enable_cross_attn True \
                --des 'Exp' \
                --train_epochs 100\
                --patience 20\
                --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$TS'_'$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$cl_weight'_'$pcle_outdims'_'$stride.log 
            done
        done
    done
done

