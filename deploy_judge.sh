#!/usr/bin/env bash
# Deploy Qwen2.5-VL-7B as grounding judge on GPU 0, port 12999.
# Run this BEFORE starting training:  bash deploy_judge.sh
#
# swift deploy 会拉起 vLLM EngineCore 等子进程；若仅主进程崩溃，子进程可能残留占 GPU。
# 这里用 setsid + 进程组，在退出 / Ctrl+C / 报错时尽量整组清理。
set -euo pipefail

MODEL_PATH="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct"
PORT=12999

DEPLOY_PID=""
DEPLOY_PGID=""
_CLEANUP_DONE=0

cleanup() {
    ((_CLEANUP_DONE)) && return 0
    _CLEANUP_DONE=1
    if [[ -z "${DEPLOY_PGID:-}" ]]; then
        return 0
    fi
    echo "[judge] Cleaning up process group ${DEPLOY_PGID} (swift / vLLM) ..."
    kill -- -"${DEPLOY_PGID}" 2>/dev/null || true
    sleep 2
    kill -9 -- -"${DEPLOY_PGID}" 2>/dev/null || true
    if [[ -n "${DEPLOY_PID:-}" ]]; then
        wait "${DEPLOY_PID}" 2>/dev/null || true
    fi
}

trap cleanup INT TERM EXIT

echo "[judge] Deploying on GPU 0, port $PORT ..."
setsid env CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_max_model_len 4096 &
DEPLOY_PID=$!
DEPLOY_PGID=$(ps -o pgid= -p "${DEPLOY_PID}" | tr -d ' ')

wait "${DEPLOY_PID}"
