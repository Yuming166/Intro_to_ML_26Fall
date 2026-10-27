"""Run the prepared MBPP tasks sequentially against two OpenAI-compatible services."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from agent.tools import Workspace

ROOT = Path(__file__).resolve().parents[2]


def heldout(workspace, test_path):
    if any(p.is_symlink() for p in workspace.rglob('*')):
        return {'status': 'invalid_workspace', 'ok': False}
    with tempfile.TemporaryDirectory(prefix='mbpp-heldout-') as tmp:
        project = Path(tmp) / 'project'
        shutil.copytree(workspace, project, ignore=shutil.ignore_patterns('tests', '__pycache__', '*.pyc', '.git'))
        (project / 'tests').mkdir()
        shutil.copy2(test_path, project / 'tests/test_heldout.py')
        return Workspace(project).run_tests()


def run_one(prepared, output, task, key, model, url, timeout, steps, tokens):
    taskid = str(task['task_id'])
    row = {'task_id': task['task_id'], 'model_key': key, 'model_id': model,
           'status': 'not_run', 'correctness': 'unknown'}
    if task['reference_preflight'] != 'passed':
        row['status'] = 'reference_preflight_failed'
        return row
    source = prepared / 'inputs' / taskid
    run = output / 'runs' / key / taskid
    run.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-m', 'agent.cli', '--workspace', str(source),
           '--task', (source / 'task.txt').read_text(), '--run-dir', str(run),
           '--model', model, '--base-url', url, '--max-steps', str(steps), '--max-tokens', str(tokens)]
    start = time.monotonic()
    with (output / f'{key}-{taskid}.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            row.update(status='timeout', wall_seconds=round(time.monotonic()-start, 3))
            return row
    row.update(status='process_completed', wall_seconds=round(time.monotonic()-start, 3), exit_code=code)
    result_file = run / 'result.json'
    if not result_file.is_file():
        row['status'] = 'missing_result'
        return row
    result = json.loads(result_file.read_text())
    if result.get('model') != model:
        row['status'] = 'model_identity_mismatch'
        return row
    public = result.get('final_tests', {}).get('status', 'missing')
    row.update(agent_status=result.get('status'), public_status=public,
               security_status=result.get('security_review', {}).get('status', 'missing'),
               input_tokens=result.get('input_tokens'), output_tokens=result.get('output_tokens'))
    if result.get('status') in ('model_error', 'verification_unavailable') or not (run / 'workspace/solution.py').is_file():
        row['status'] = 'agent_or_verification_error'
        return row
    grade = heldout(run / 'workspace', prepared / 'private' / taskid / 'test_heldout.py')
    (output / f'{key}-{taskid}-heldout.json').write_text(json.dumps(grade, indent=2) + '\n')
    row['heldout_status'] = grade['status']
    if public not in ('passed', 'failed') or grade['status'] not in ('passed', 'failed'):
        row['status'] = 'verification_error'
        return row
    row['correctness'] = 'passed' if public == grade['status'] == 'passed' else 'failed'
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True, help='New output directory')
    ap.add_argument('--4b-url', required=True, dest='url4')
    ap.add_argument('--35b-url', required=True, dest='url35')
    ap.add_argument('--4b-model', default='Qwen3.5-4B', dest='model4')
    ap.add_argument('--35b-model', default='Qwen3.6-35B-A3B', dest='model35')
    ap.add_argument('--timeout', type=int, default=240)
    ap.add_argument('--max-steps', type=int, default=12)
    ap.add_argument('--max-tokens', type=int, default=1024)
    a = ap.parse_args()
    if a.timeout <= 0 or not 1 <= a.max_steps <= 40 or not 64 <= a.max_tokens <= 4096:
        ap.error('Invalid timeout, max-steps or max-tokens')
    prepared = a.prepared.resolve(strict=True)
    output = a.out.resolve()
    if output == prepared or output.is_relative_to(prepared) or prepared.is_relative_to(output):
        ap.error('Output and prepared directories must not contain each other')
    metadata = json.loads((prepared / 'prepared.json').read_text())
    tasks = metadata['tasks']
    protocol = json.loads((ROOT / 'results/mbpp20/protocol.json').read_text())
    frozen = [t['task_id'] for t in protocol['tasks']]
    ids = [t['task_id'] for t in tasks]
    if not ids or ids != frozen[:len(ids)] or metadata['dataset_sha256'] != protocol['dataset_sha256']:
        raise SystemExit('Prepared task list or dataset provenance mismatch')
    expected_files = {f'inputs/{i}/{suffix}' for i in ids for suffix in ('solution.py', 'task.txt', 'tests/test_public.py')}
    expected_files |= {f'private/{i}/test_heldout.py' for i in ids}
    hashes = metadata.get('file_sha256', {})
    if set(hashes) != expected_files or any(
            hashlib.sha256((prepared / name).read_bytes()).hexdigest() != digest
            for name, digest in hashes.items()):
        raise SystemExit('Prepared inputs or held-out tests changed')
    if any(x['reference_preflight'] != 'passed' for x in tasks):
        raise SystemExit('Reference preflight failed')
    output.mkdir(parents=True, exist_ok=False)
    config = {'4b': (a.model4, a.url4), '35b': (a.model35, a.url35)}
    # Avoid saving endpoint URLs or API keys in the comparison summary.
    record = {'models': {'4b': a.model4, '35b': a.model35}, 'dataset_sha256': metadata['dataset_sha256'],
              'task_ids': [t['task_id'] for t in tasks], 'timeout_s': a.timeout,
              'max_steps': a.max_steps, 'max_tokens': a.max_tokens, 'concurrency': 1,
              'no_retry': True, 'note': 'Wall time excludes held-out grading; local logs may contain URLs.'}
    (output / 'config.json').write_text(json.dumps(record, indent=2) + '\n')
    rows = []
    with (output / 'episodes.jsonl').open('x') as file:
        for i, task in enumerate(tasks):
            for key in (('4b', '35b') if i % 2 == 0 else ('35b', '4b')):
                model, url = config[key]
                try:
                    row = run_one(prepared, output, task, key, model, url, a.timeout, a.max_steps, a.max_tokens)
                except Exception as exc:
                    row = {'task_id': task['task_id'], 'model_key': key, 'model_id': model,
                           'status': 'runner_error', 'correctness': 'unknown', 'error_type': type(exc).__name__}
                rows.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
                file.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    fields = ['task_id', 'model_key', 'model_id', 'status', 'correctness', 'agent_status', 'public_status',
              'heldout_status', 'wall_seconds', 'exit_code', 'security_status', 'input_tokens', 'output_tokens', 'error_type']
    with (output / 'episodes.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    summary = {}
    for key in config:
        subset = [r for r in rows if r['model_key'] == key]
        times = [r['wall_seconds'] for r in subset if 'wall_seconds' in r]
        summary[key] = {'planned': len(tasks), 'recorded': len(subset),
                        **{status: sum(r['correctness'] == status for r in subset)
                           for status in ('passed', 'failed', 'unknown')},
                        'timed': len(times), 'wall_mean_s': round(statistics.mean(times), 3) if times else None,
                        'wall_median_s': round(statistics.median(times), 3) if times else None}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

if __name__ == '__main__':
    main()
