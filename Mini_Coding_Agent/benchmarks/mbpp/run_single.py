"""Run the frozen MBPP tasks once against a single OpenAI-compatible model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from collections import Counter
from .run import ROOT, run_one


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path, help='New output directory')
    ap.add_argument('--base-url', required=True, help='Model service or local adapter /v1 base URL')
    ap.add_argument('--model', required=True, help='Model ID')
    ap.add_argument('--timeout', type=int, default=240)
    ap.add_argument('--max-steps', type=int, default=12)
    ap.add_argument('--max-tokens', type=int, default=1024)
    args = ap.parse_args()
    if args.timeout <= 0 or not 1 <= args.max_steps <= 40 or not 64 <= args.max_tokens <= 4096:
        ap.error('Invalid timeout, max-steps or max-tokens')
    prepared = args.prepared.resolve(strict=True)
    output = args.out.resolve()
    if output == prepared or output.is_relative_to(prepared) or prepared.is_relative_to(output):
        ap.error('Output and prepared directories must not contain each other')
    metadata = json.loads((prepared / 'prepared.json').read_text())
    tasks = metadata['tasks']
    original = json.loads((ROOT / 'results/mbpp20/protocol.json').read_text())
    ids = [t['task_id'] for t in tasks]
    frozen = [t['task_id'] for t in original['tasks']]
    if not ids or ids != frozen[:len(ids)] or metadata['dataset_sha256'] != original['dataset_sha256']:
        raise SystemExit('Prepared task list or dataset provenance mismatch')
    expected = {f'inputs/{i}/{suffix}' for i in ids for suffix in ('solution.py', 'task.txt', 'tests/test_public.py')}
    expected |= {f'private/{i}/test_heldout.py' for i in ids}
    hashes = metadata.get('file_sha256', {})
    if set(hashes) != expected or any(hashlib.sha256((prepared / name).read_bytes()).hexdigest() != digest
                                      for name, digest in hashes.items()):
        raise SystemExit('Prepared inputs or held-out tests changed')
    if any(t['reference_preflight'] != 'passed' for t in tasks):
        raise SystemExit('Reference preflight failed')
    output.mkdir(parents=True, exist_ok=False)
    config = {'model': args.model, 'dataset_sha256': metadata['dataset_sha256'], 'task_ids': ids,
              'timeout_s': args.timeout, 'max_steps': args.max_steps, 'max_tokens': args.max_tokens,
              'concurrency': 1, 'no_retry': True,
              'note': 'Wall time excludes held-out grading; local run traces may contain the URL.'}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    rows = []
    with (output / 'episodes.jsonl').open('x') as stream:
        for task in tasks:
            try:
                row = run_one(prepared, output, task, 'single', args.model, args.base_url,
                              args.timeout, args.max_steps, args.max_tokens)
            except Exception as exc:
                row = {'task_id': task['task_id'], 'model_key': 'single', 'model_id': args.model,
                       'status': 'runner_error', 'correctness': 'unknown', 'error_type': type(exc).__name__}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    fields = ['task_id', 'model_key', 'model_id', 'status', 'correctness', 'agent_status', 'public_status',
              'heldout_status', 'wall_seconds', 'exit_code', 'security_status', 'input_tokens', 'output_tokens', 'error_type']
    with (output / 'episodes.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    times = [r['wall_seconds'] for r in rows if 'wall_seconds' in r]
    counts = Counter(r['correctness'] for r in rows)
    summary = {'planned': len(tasks), 'recorded': len(rows),
               **{k: counts[k] for k in ('passed', 'failed', 'unknown')}, 'timed': len(times),
               'wall_mean_s': round(statistics.mean(times), 3) if times else None,
               'wall_median_s': round(statistics.median(times), 3) if times else None}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
