#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: $(basename "$0") [options]

Options:
    -d, --datasets "DS1 DS2"   要遍历的数据集列表，空格或逗号分隔；默认: electricity traffic
    -b, --batch-size "B1 B2"   对应的 batch_size 列表（长度不足时按首个值复用）；默认: 16
    -l, --seq-len "L1 L2"      输入序列长度列表；默认: 96
    -r, --stride "S1 S2"       滑窗步长列表；默认: 4
    -h, --help                 打印帮助

示例:
  ./scripts/$(basename "$0") -d "ETTh1 ETTh2" -b 32 -l 96 -r 4
EOF
}

DATASET_LIST=("weather")
SEQ_LEN_LIST=(96 96 96)
STRIDE_LIST=(1 2 4 )
BATCH_SIZE_LIST=(256 512 1024 )


while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--datasets)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] 缺少 --datasets 的参数" >&2
                exit 1
            fi
            ds_string=${2//,/ }
            read -r -a DATASET_LIST <<< "$ds_string"
            shift 2
            ;;
        -b|--batch-size)
            [[ $# -ge 2 ]] || { echo "[ERROR] 缺少 --batch-size 的参数" >&2; exit 1; }
            bs_string=${2//,/ }
            read -r -a BATCH_SIZE_LIST <<< "$bs_string"
            shift 2
            ;;
        -l|--seq-len)
            [[ $# -ge 2 ]] || { echo "[ERROR] 缺少 --seq-len 的参数" >&2; exit 1; }
            sl_string=${2//,/ }
            read -r -a SEQ_LEN_LIST <<< "$sl_string"
            shift 2
            ;;
        -r|--stride)
            [[ $# -ge 2 ]] || { echo "[ERROR] 缺少 --stride 的参数" >&2; exit 1; }
            st_string=${2//,/ }
            read -r -a STRIDE_LIST <<< "$st_string"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] 未知参数: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ ${#DATASET_LIST[@]} -eq 0 ]]; then
    DATASET_LIST=("electricity" "traffic")
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "$PROJECT_ROOT"

echo "[INFO] 项目根目录: $PROJECT_ROOT"
echo "[INFO] 数据集: ${DATASET_LIST[*]}"
if [[ ${#SEQ_LEN_LIST[@]} -eq 0 ]]; then
    echo "[ERROR] 至少需要一个 seq_len" >&2
    exit 1
fi

if [[ ${#BATCH_SIZE_LIST[@]} -eq 0 ]]; then
    BATCH_SIZE_LIST=(16)
fi

if [[ ${#STRIDE_LIST[@]} -eq 0 ]]; then
    STRIDE_LIST=(4)
fi

echo "[INFO] seq_len 列表: ${SEQ_LEN_LIST[*]}"
echo "[INFO] batch_size 列表: ${BATCH_SIZE_LIST[*]}"
echo "[INFO] stride 列表: ${STRIDE_LIST[*]}"

for DATASET in "${DATASET_LIST[@]}"; do
    for idx in "${!SEQ_LEN_LIST[@]}"; do
        seq_len=${SEQ_LEN_LIST[$idx]}

        if [[ $idx -lt ${#BATCH_SIZE_LIST[@]} ]]; then
            batch_size=${BATCH_SIZE_LIST[$idx]}
        else
            batch_size=${BATCH_SIZE_LIST[0]}
        fi

        if [[ $idx -lt ${#STRIDE_LIST[@]} ]]; then
            stride=${STRIDE_LIST[$idx]}
        else
            stride=${STRIDE_LIST[0]}
        fi

        echo "[INFO] 开始运行: dataset=$DATASET | seq_len=$seq_len | batch_size=$batch_size | stride=$stride"
        python DCLT_main_pretrain_v3.py \
            dataset.name="$DATASET" \
            train.batch_size="$batch_size" \
            model.seq_len="$seq_len" \
            model.stride="$stride"
        echo "[INFO] 完成: dataset=$DATASET | seq_len=$seq_len"
        echo "----------------------------------------"
        sleep 1
    done
done

echo "[INFO] 批量运行完成"
