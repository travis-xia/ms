#!/usr/bin/env bash
# 单卡对 grpo_vl7b_full checkpoint 做一次推理冒烟（与 vllm_vl7b_full.sh 环境变量对齐）
# 改下面「配置」块即可，不要传命令行参数。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ---------- 配置（按需修改）----------
CKPT="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/ms-swift/output/grpo_vl7b_full_v1/v0-20260427-050640/checkpoint-2800"
# system 文件：相对仓库根的路径，或绝对路径；留空则不传 --system（Python 侧会用默认 spatiotemporal）
SYSTEM_FILE="examples/train/grpo/prompt.txt"
# 测试 jsonl：存在则跑该文件第 JSONL_LINE 条；不存在则自动 --image-fallback
JSONL="data/expanded/vl_grpo_all_expanded.jsonl"
JSONL_LINE=0

export CUDA_VISIBLE_DEVICES=0
export MAX_PIXELS=401408
export VIDEO_MAX_PIXELS=401408
export FPS_MAX_FRAMES=16
# ---------- 配置结束 ----------

SYSTEM_ARGS=()
if [[ -n "$SYSTEM_FILE" ]]; then
  if [[ "$SYSTEM_FILE" != /* ]]; then
    SYSTEM_FILE="$ROOT/$SYSTEM_FILE"
  fi
  if [[ -f "$SYSTEM_FILE" ]]; then
    SYSTEM_ARGS=(--system "$SYSTEM_FILE")
  else
    echo "警告: 找不到 system 文件: $SYSTEM_FILE，将不传 --system" >&2
  fi
fi

JSONL_PATH="$JSONL"
if [[ "$JSONL_PATH" != /* ]]; then
  JSONL_PATH="$ROOT/$JSONL_PATH"
fi

if [[ -f "$JSONL_PATH" ]]; then
  python examples/train/grpo/test_grpo_checkpoint_infer.py --ckpt "$CKPT" "${SYSTEM_ARGS[@]}" --jsonl "$JSONL_PATH" --jsonl-line "$JSONL_LINE"
else
  echo "警告: 找不到 jsonl: $JSONL_PATH，使用 --image-fallback" >&2
  python examples/train/grpo/test_grpo_checkpoint_infer.py --ckpt "$CKPT" "${SYSTEM_ARGS[@]}" --image-fallback
fi
