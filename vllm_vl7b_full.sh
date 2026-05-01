# 7*H200 GPU for training + GPU0 for grounding judge
set -o pipefail

MODEL_PATH="/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct"
JUDGE_PORT=8100
EXP_NAME="grpo_vl7b_full_v0501"
mkdir -p "output/$EXP_NAME"

# ── 1. Launch judge on GPU 0 in background ──
echo "[judge] Starting judge service on GPU 0, port $JUDGE_PORT ..."
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --model "$MODEL_PATH" \
    --port "$JUDGE_PORT" \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_max_model_len 4096 \
    > "output/$EXP_NAME/judge.log" 2>&1 &
JUDGE_PID=$!

cleanup() { echo "[judge] Stopping judge (pid=$JUDGE_PID)"; kill $JUDGE_PID 2>/dev/null; wait $JUDGE_PID 2>/dev/null; }
trap cleanup EXIT

echo "[judge] Waiting for judge to be ready (pid=$JUDGE_PID) ..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$JUDGE_PORT/v1/models" >/dev/null 2>&1; then
        echo "[judge] Ready after ${i}s."
        break
    fi
    if ! kill -0 $JUDGE_PID 2>/dev/null; then
        echo "[judge] Process died. Check output/$EXP_NAME/judge.log"; exit 1
    fi
    sleep 1
done
if ! curl -s "http://localhost:$JUDGE_PORT/v1/models" >/dev/null 2>&1; then
    echo "[judge] Timed out after 120s. Check output/$EXP_NAME/judge.log"; exit 1
fi

# ── 2. Training on GPU 1-7 ──
GROUNDING_JUDGE_API_BASE=http://localhost:$JUDGE_PORT/v1 \
MAX_PIXELS=401408 \
VIDEO_MAX_PIXELS=401408 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
NPROC_PER_NODE=7 \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --dataset data/expanded/vl_grpo_all_expanded.jsonl \
    --reward_funcs external_vl_task_reward external_vl_format_reward external_grounding_judge \
    --reward_weights 1 0.5 0.5 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 1 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --max_length 8192 \
    --max_completion_length 512 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_steps 400 \
    --save_total_limit 20 \
    --logging_steps 1 \
    --output_dir output/$EXP_NAME \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --system examples/train/grpo/prompt_spatiotemporal.txt \
    --deepspeed zero2 \
    --log_completions true \
    --report_to tensorboard \
    --num_iterations 1 \
    --beta 0.04 \
    --max_grad_norm 1 \
    2>&1 | tee "output/$EXP_NAME/console.log"
