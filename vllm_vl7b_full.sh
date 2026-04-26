# 8*H200 GPU
set -o pipefail

EXP_NAME="grpo_vl7b_full_v1"
mkdir -p "output/$EXP_NAME"

MAX_PIXELS=401408 \
VIDEO_MAX_PIXELS=401408 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model /inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/Qwen2.5-VL-7B-Instruct \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --dataset data/expanded/vl_grpo_all_expanded.jsonl \
    --reward_funcs external_vl_task_reward external_vl_format_reward \
    --reward_weights 1 0.5 \
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
    --beta 0.01 \
    --max_grad_norm 1 \
    2>&1 | tee "output/$EXP_NAME/console.log"
