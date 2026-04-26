#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def iter_jsonl(path):
    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def missing_message(dataset, index, video_path):
    return f'[{dataset}] row={index} missing: {video_path}'


def print_summary(name, stats, limit):
    print(f'{name}: total={stats["total"]}, missing={stats["missing"]}, ok={stats["total"] - stats["missing"]}')
    for example in stats['examples'][:limit]:
        print(f'  {example}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-file',
        type=Path,
        default=Path(__file__).resolve().parent / 'expanded' / 'vl_grpo_all_expanded.jsonl')
    parser.add_argument('--missing-limit', type=int, default=20)
    args = parser.parse_args()

    stats_by_dataset = defaultdict(lambda: {'total': 0, 'missing': 0, 'examples': []})
    all_stats = {'total': 0, 'missing': 0, 'examples': []}

    for index, sample in enumerate(iter_jsonl(args.input_file)):
        dataset = sample.get('dataset', 'unknown')
        video_path = sample['videos'][0]
        is_missing = not Path(video_path).is_file()

        stats_by_dataset[dataset]['total'] += 1
        all_stats['total'] += 1
        if not is_missing:
            continue

        message = missing_message(dataset, index, video_path)
        stats_by_dataset[dataset]['missing'] += 1
        all_stats['missing'] += 1

        if len(stats_by_dataset[dataset]['examples']) < args.missing_limit:
            stats_by_dataset[dataset]['examples'].append(message)
        if len(all_stats['examples']) < args.missing_limit:
            all_stats['examples'].append(message)

    for dataset in sorted(stats_by_dataset):
        print_summary(dataset, stats_by_dataset[dataset], args.missing_limit)
    print_summary('all', all_stats, args.missing_limit)


if __name__ == '__main__':
    main()
