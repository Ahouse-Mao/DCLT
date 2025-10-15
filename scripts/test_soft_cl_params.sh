#!/usr/bin/env bash
set -euo pipefail

# 可通过环境变量自定义执行 Python 解释器
PYTHON_BIN=${PYTHON_BIN:-python}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

# 测试列表可按需修改，或在执行时通过环境变量注入
PATCH_WEIGHT_LIST=${PATCH_WEIGHT_LIST:-"0 0.3 0.6 1"}
TAU_TEMPORAL_LIST=${TAU_TEMPORAL_LIST:-"0.05 0.1 0.2 0.3 0.5 0.7"}

# 兼容以逗号分隔的输入，统一替换为单个空格
PATCH_WEIGHT_LIST=${PATCH_WEIGHT_LIST//,/ }
TAU_TEMPORAL_LIST=${TAU_TEMPORAL_LIST//,/ }

# 透传额外 Hydra 参数，例如：EXTRA_ARGS='train.max_epochs=5'
EXTRA_ARGS=${EXTRA_ARGS:-""}

for patch_weight in ${PATCH_WEIGHT_LIST}; do
  for tau_temporal in ${TAU_TEMPORAL_LIST}; do
    clean_patch_weight=${patch_weight%,}
    clean_tau_temporal=${tau_temporal%,}

    tag_pw=${clean_patch_weight//./p}
    tag_tau=${clean_tau_temporal//./p}
    run_dir="outputs/soft_cl_tests/patch_${tag_pw}_tau_${tag_tau}"

    echo "=== Running patch_sim_weight=${clean_patch_weight}, tau_temporal=${clean_tau_temporal} ==="
    cmd=(
      "${PYTHON_BIN}" "DCLT_main_pretrain_v3.py"
      "model.soft_cl_params.patch_sim_weight=${clean_patch_weight}"
      "model.soft_cl_params.tau_temporal=${clean_tau_temporal}"
      "hydra.run.dir=${run_dir}"
    )

    if [[ -n "${EXTRA_ARGS}" ]]; then
      # shellcheck disable=SC2206
      extra_args=( ${EXTRA_ARGS} )
      cmd+=("${extra_args[@]}")
    fi

    HYDRA_FULL_ERROR=1 "${cmd[@]}"
  done
done
