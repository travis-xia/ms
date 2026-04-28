#!/usr/bin/env python3
"""对 GRPO 全参 checkpoint 做一次推理冒烟测试（单条样本，打印模型输出）。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

DEFAULT_CKPT = (
    '/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/ms-swift/output/'
    'grpo_vl7b_full_v1/v0-20260427-050640/checkpoint-2800'
)


def load_jsonl_row(path: Path, line: int) -> dict:
    with path.open('r', encoding='utf-8') as f:
        for i, line_ in enumerate(f):
            if i == line:
                return json.loads(line_)
    raise IndexError(f'jsonl 行号 {line} 超出文件范围: {path}')


def build_infer_request(
    *,
    jsonl_path: Path | None,
    jsonl_line: int,
    system_path: Path | None,
    use_image_fallback: bool,
) -> tuple[object, str | None]:
    """返回 (InferRequest, 可选的 golden solution 字符串)."""
    from swift.infer_engine import InferRequest

    solution = None
    if jsonl_path is not None:
        row = load_jsonl_row(jsonl_path, jsonl_line)
        solution = row.get('solution')
        messages = [dict(m) for m in row['messages']]
        videos = list(row.get('videos') or [])
        images = list(row.get('images') or [])
        audios = list(row.get('audios') or [])
        if system_path and system_path.is_file():
            system_text = system_path.read_text(encoding='utf-8').strip()
            if system_text and (not messages or messages[0].get('role') != 'system'):
                messages.insert(0, {'role': 'system', 'content': system_text})
        return InferRequest(messages=messages, videos=videos, images=images, audios=audios), solution

    if not use_image_fallback:
        raise ValueError('未指定 --jsonl 且未开启 --image-fallback')

    demo_image = os.environ.get(
        'GRPO_TEST_IMAGE',
        'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png',
    )
    user_text = os.environ.get(
        'GRPO_TEST_PROMPT',
        '<image>\n请简要描述图片里有什么；按训练格式用 <think>...</think>'
        '<answer>...</answer> 输出。',
    )
    messages = []
    if system_path and system_path.is_file():
        st = system_path.read_text(encoding='utf-8').strip()
        if st:
            messages.append({'role': 'system', 'content': st})
    messages.append({'role': 'user', 'content': user_text})
    return InferRequest(messages=messages, images=[demo_image]), solution


def main():
    parser = argparse.ArgumentParser(description='GRPO VL checkpoint 单条推理测试')
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    default_jsonl = repo / 'data' / 'expanded' / 'vl_grpo_all_expanded.jsonl'
    default_system = here / 'prompt_spatiotemporal.txt'

    parser.add_argument('--ckpt', default=DEFAULT_CKPT, help='全参微调 checkpoint 目录（含权重与 args.json）')
    parser.add_argument('--jsonl', type=Path, default=None, help='与训练相同格式的 jsonl；默认尝试仓库内 expanded 数据')
    parser.add_argument('--jsonl-line', type=int, default=0, help='使用的行号（0-based）')
    parser.add_argument(
        '--system',
        type=Path,
        default=None,
        help='system 提示词文件（与训练时 swift 的 --system 一致）；可写相对仓库路径。'
        '未传且未设置环境变量 GRPO_TEST_SYSTEM 时，默认使用 examples/train/grpo/prompt_spatiotemporal.txt',
    )
    parser.add_argument('--no-system', action='store_true', help='不注入 system（忽略 --system 与 GRPO_TEST_SYSTEM）')
    parser.add_argument(
        '--backend',
        choices=('transformers', 'vllm'),
        default='transformers',
        help='推理后端；单卡冒烟建议 transformers',
    )
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument(
        '--image-fallback',
        action='store_true',
        help='无 jsonl 或视频路径不可用时，用公开图片 URL 跑一条纯视觉示例',
    )
    args = parser.parse_args()

    jsonl_path = args.jsonl
    if jsonl_path is None and default_jsonl.is_file():
        jsonl_path = default_jsonl

    if args.no_system:
        system_path = None
    elif args.system is not None:
        system_path = args.system
    elif os.environ.get('GRPO_TEST_SYSTEM'):
        system_path = Path(os.environ['GRPO_TEST_SYSTEM'])
    else:
        system_path = default_system

    if jsonl_path is not None and not jsonl_path.is_file():
        print(f'警告: 找不到 jsonl {jsonl_path}，改用 --image-fallback')
        jsonl_path = None
        args.image_fallback = True

    infer_request, solution = build_infer_request(
        jsonl_path=jsonl_path,
        jsonl_line=args.jsonl_line,
        system_path=system_path,
        use_image_fallback=args.image_fallback or jsonl_path is None,
    )

    print('=== InferRequest（摘要）===')
    print(infer_request.to_printable())
    if solution:
        print('=== 数据中的 solution（仅对比，不参与推理）===')
        print(solution)

    from swift.infer_engine import RequestConfig

    req_cfg = RequestConfig(max_tokens=args.max_tokens, temperature=args.temperature, stream=False)

    if args.backend == 'transformers':
        from swift.infer_engine import TransformersEngine

        engine = TransformersEngine(args.ckpt, torch_dtype=torch.bfloat16)
    else:
        from swift.infer_engine import VllmEngine

        engine = VllmEngine(
            args.ckpt,
            torch_dtype=torch.bfloat16,
            tensor_parallel_size=int(os.environ.get('VLLM_TP', '1')),
        )

    responses = engine.infer([infer_request], req_cfg)
    text = responses[0].choices[0].message.content
    print('=== 模型输出 ===')
    print(text)


if __name__ == '__main__':
    main()
