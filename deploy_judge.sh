#!/usr/bin/env bash
# Deploy Qwen2.5-VL-7B as grounding judge on GPU 0, port 8100.
# Run this BEFORE starting training:  bash deploy_judge.sh
set -euo pipefail

MODEL_PATH="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct"
PORT=8100

echo "[judge] Deploying on GPU 0, port $PORT ..."
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --infer_backend vllm \
    --gpu_memory_utilization 0.85 \
    --max_model_len 4096
