import argparse
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/timelens-100k.jsonl')
    parser.add_argument('--output', default='data/timelens-10k.jsonl')
    parser.add_argument('--size', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    lines = input_path.read_text(encoding='utf-8').splitlines()
    sample_size = min(args.size, len(lines))

    sampled_lines = random.Random(args.seed).sample(lines, sample_size)
    output_path.write_text('\n'.join(sampled_lines) + '\n', encoding='utf-8')

    print(f'Wrote {len(sampled_lines)} samples to {output_path}')


if __name__ == '__main__':
    main()
