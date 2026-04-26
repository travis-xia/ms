#!/usr/bin/env python3
# Dataset conversion rules for VL GRPO training:
# 1. Current sampling policy:
#    - Charades-STA keeps at most the first 2 sentence-span pairs per video.
#    - ActivityNet randomly keeps half of the videos, then randomly keeps
#      1 sentence-span pair for each selected video.
#    - TimeLens randomly keeps 1 event per sample.
#    - Video-R1 keeps only "multiple choice" samples, then randomly keeps half.
# 2. Charades-STA, ActivityNet, and TimeLens are pure temporal grounding tasks.
#    They ask the model to locate the start/end timestamps for a description or
#    query, with one-decimal precision. Final answer format:
#    <answer>[start, end]</answer>
# 3. Video-R1 is kept here as answer-only QA. We only convert
#    "multiple choice" samples and filter out all "numerical" samples.
#    Final answer format:
#    <answer>B</answer>
# 4. Therefore the merged dataset contains two output patterns:
#    timestamp-only and answer-only.
import argparse
import json
import random
from pathlib import Path


def fmt_time(value):
    return f'{float(value):.1f}'


def fmt_span(span):
    start, end = span
    return f'[{fmt_time(start)}, {fmt_time(end)}]'


def timestamp_answer(span):
    return f'<answer>{fmt_span(span)}</answer>'


def add_video_token(question, enabled=True):
    return f'<video>\n{question}' if enabled else question


def resolve_video_path(video_root, relative_path):
    if not video_root:
        return relative_path
    relative_path = relative_path.removeprefix('./')
    return str(Path(video_root) / relative_path)


def make_record(sample_id, dataset, question, video, solution, add_token=True):
    return {
        'id': sample_id,
        'dataset': dataset,
        'messages': [{
            'role': 'user',
            'content': add_video_token(question, add_token),
        }],
        'videos': [video],
        'solution': solution,
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count


def concat_jsonl(output_path, input_paths):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open('w', encoding='utf-8') as out:
        for input_path in input_paths:
            with input_path.open('r', encoding='utf-8') as src:
                for line in src:
                    out.write(line)
                    count += 1
    return count


def load_json(path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def iter_jsonl(path):
    # timelens has a few dirty bytes; replace them so HF datasets can load the converted jsonl.
    with path.open('r', encoding='utf-8', errors='replace') as f:
        skipped = 0
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
        if skipped:
            print(f'{path.name}: skipped {skipped} malformed jsonl rows')


def choose_indices(size, keep, rng):
    keep = min(size, keep)
    return sorted(rng.sample(range(size), keep))


def convert_charades(path, video_root, add_token=True):
    data = load_json(path)
    for video_id, ex in data.items():
        for i, (description, span) in enumerate(zip(ex['sentences'][:2], ex['timestamps'][:2])):
            question = (
                f'You are a helpful assistant.\n'
                f'The video duration is {float(ex["duration"]):.1f} seconds.\n'
                f'Locate the start and end timestamps of the video segment corresponding to the description: '
                f'{description}\n'
                f'Please provide the start and end timestamps in seconds, precise to one decimal place.\n'
                f'Your final answer must be in the format: '
                f'<answer>[start, end]</answer>.'
            )
            video = resolve_video_path(video_root, f'{video_id}.mp4')
            yield make_record(
                f'charades-{video_id}-{i}',
                'charades_sta',
                question,
                video,
                timestamp_answer(span),
                add_token,
            )


def convert_activitynet(path, video_root, rng, add_token=True):
    data = load_json(path)
    video_ids = list(data)
    for video_i in choose_indices(len(video_ids), len(video_ids) // 2, rng):
        video_id = video_ids[video_i]
        ex = data[video_id]
        pairs = list(zip(ex['sentences'], ex['timestamps']))
        for i in choose_indices(len(pairs), 1, rng):
            description, span = pairs[i]
            question = (
                f'You are a helpful assistant.\n'
                f'The video duration is {float(ex["duration"]):.1f} seconds.\n'
                f'Locate the start and end timestamps of the video segment corresponding to the description: '
                f'{description}\n'
                f'Please provide the start and end timestamps in seconds, precise to one decimal place.\n'
                f'Your final answer must be in the format: '
                f'<answer>[start, end]</answer>.'
            )
            video = resolve_video_path(video_root, f'{video_id}.mp4')
            yield make_record(
                f'activitynet-{video_id}-{i}',
                'activitynet',
                question,
                video,
                timestamp_answer(span),
                add_token,
            )


def convert_timelens(path, video_root, rng, add_token=True):
    for sample_i, sample in enumerate(iter_jsonl(path)):
        events = sample['events']
        for event_i in choose_indices(len(events), 1, rng):
            event = events[event_i]
            span = event['span'][0]
            question = (
                f'You are a helpful assistant.\n'
                f'The video duration is {float(sample["duration"]):.1f} seconds.\n'
                f'Locate the start and end timestamps of the video segment corresponding to the query: '
                f'{event["query"]}\n'
                f'Please provide the start and end timestamps in seconds, precise to one decimal place.\n'
                f'Your final answer must be in the format: '
                f'<answer>[start, end]</answer>.'
            )
            yield make_record(
                f'timelens-{sample_i}-{event_i}',
                'timelens',
                question,
                resolve_video_path(video_root, sample['video_path']),
                timestamp_answer(span),
                add_token,
            )


def convert_video_r1(path, video_root, rng, add_token=True):
    data = load_json(path)
    data = [ex for ex in data if ex.get('problem_type') == 'multiple choice']
    for i in choose_indices(len(data), len(data) // 2, rng):
        ex = data[i]
        options = '\n'.join(ex['options'])
        question = (
            f'You are a helpful assistant.\n'
            f'Question: {ex["problem"]}\n'
            f'Options:\n{options}\n'
            f'Please answer the question.\n'
            f'Your final answer must be in the format: '
            f'<answer>B</answer>.'
        )
        yield make_record(
            f'video-r1-{ex["problem_id"]}',
            'video_r1',
            question,
            resolve_video_path(video_root, ex['path']),
            ex['solution'],
            add_token,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'expanded')
    parser.add_argument(
        '--charades-video-root',
        default='/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/VideoAuto-R1-Data/CharadesSTA/Charades_v1_480')
    parser.add_argument(
        '--activitynet-video-root',
        default='/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/TVG-R1/video-r1/data/GroundedVLLM/activitynet/videos')
    parser.add_argument(
        '--timelens-video-root',
        default='/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/TimeLens-100K/video_shards')
    parser.add_argument(
        '--video-r1-video-root',
        default='/inspire/qb-ilm/project/traffic-congestion-management/xiacheng-240108120111/hf_download/VideoAuto-R1-Data/Video-R1/videos')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-video-token', action='store_true')
    args = parser.parse_args()

    add_token = not args.no_video_token
    rng = random.Random(args.seed)
    converters = {
        'charades_train_expanded.jsonl': convert_charades(
            args.data_dir / 'Charades_train.json', args.charades_video_root, add_token),
        'activitynet_train_expanded.jsonl': convert_activitynet(
            args.data_dir / 'ActivityNet_train.json', args.activitynet_video_root, rng, add_token),
        'timelens_10k_expanded.jsonl': convert_timelens(
            args.data_dir / 'timelens-10k.jsonl', args.timelens_video_root, rng, add_token),
        'video_r1_filter_17k_expanded.jsonl': convert_video_r1(
            args.data_dir / 'Video-R1-filter-17k.json', args.video_r1_video_root, rng, add_token),
    }

    counts = {}
    output_paths = []
    for name, rows in converters.items():
        output_path = args.output_dir / name
        counts[name] = write_jsonl(output_path, rows)
        output_paths.append(output_path)

    combined_path = args.output_dir / 'vl_grpo_all_expanded.jsonl'
    counts[combined_path.name] = concat_jsonl(combined_path, output_paths)

    for name, count in counts.items():
        print(f'{name}: {count}')


if __name__ == '__main__':
    main()
