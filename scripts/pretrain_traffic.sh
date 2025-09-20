#!/usr/bin/env bash
set -euo pipefail

# 使用说明（Usage examples）：
#   bash scripts/pretrain_weather.sh
#   （可在脚本顶部数组中修改要批跑的数据集、种子、批大小等超参）
#
# 说明（Notes）：
# - 该脚本通过 Hydra 的“命令行覆盖”能力修改 cl_conf/pretrain_cfg.yaml 及其嵌套字段。
# - 可以覆盖嵌套键，例如：light_model.trainer.batch_size=64。
# - 下面的多重 for 循环会生成超参组合批量运行；按需修改数组即可。
# - 如果更偏好 Hydra 自带的多运行（-m），脚本底部也给出示例。

# 切换到项目根目录（脚本位于 scripts/ 下）
cd "$(dirname "$0")/.."

# 横向扫描的参数数组（可自由调整）
DATASETS=(traffic)   # 数据集名称（映射到 cfg.dataset.name）
SEEDS=(42)                   # 随机种子（cfg.seed）
BATCH_SIZES=(32)              # 批大小（cfg.light_model.trainer.batch_size）
MAX_EPOCHS=(100)                  # 最大训练轮数（cfg.light_model.trainer.max_epochs）
DEVICES=(1)                      # 设备数量（cfg.devices）

# 其它可选覆盖项（示例）
# - 学习率（若你的优化器配置中使用了该字段）
LRS=(1e-4)                  # （示例：cfg.light_model.optimizer.lr）

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
      for me in "${MAX_EPOCHS[@]}"; do
        for dev in "${DEVICES[@]}"; do
          for lr in "${LRS[@]}"; do
            # 使用系统默认 CUDA 设置（不显式指定 CUDA_VISIBLE_DEVICES）
            echo "[RUN] dataset=$ds seed=$seed batch_size=$bs max_epochs=$me devices=$dev lr=$lr"

            python DCLT_main_pretrain.py \
              dataset.name="$ds" \
              seed="$seed" \
              devices="$dev" \
              light_model.trainer.batch_size="$bs" \
              light_model.trainer.max_epochs="$me" \
              light_model.optimizer.lr="$lr" \
              dataset.k=10 \
              dataset.P=5 \
              dataset.N=20 \
              light_model.head.hidden_dim_1=128 \
              light_model.head.out_dim_1=64 \
              light_model.head.hidden_dim_2=2048 \
              light_model.head.final_out_dim=1024
              # 如需更多覆盖示例，可取消注释并按需修改： \
              # model_name=DCLT_patchtst_pretrained_cl \
              # light_model.trainer.monitor=train_loss \
              # light_model.trainer.monitor_mode=min \
              # dataset.k=10 dataset.P=5 dataset.N=20 \
              # dataset.neg_sampling=hard dataset.use_mutual=True \
              # +new.key=value   # 若需新增不存在的键，使用 + 前缀

          done
        done
      done
    done
  done
done

echo "All runs finished."

# 提示（Tips）：
# 1) 使用 Hydra 多运行（默认串行），无需 bash 循环：
#    python DCLT_main_pretrain.py -m \
#      dataset.name=weather,electricity \
#      seed=42,123 \
#      light_model.trainer.batch_size=32,64 \
#      light_model.early_stop.enable=True,False \
#      hydra.sweep.dir=./multirun \
#      hydra.sweep.subdir='${dataset.name}/${now:%Y%m%d_%H%M%S}/${hydra.job.num}'
#
# 2) Hydra 运行时会改变工作目录，代码内部应使用 get_original_cwd() 来拼接基于项目根目录的路径。
