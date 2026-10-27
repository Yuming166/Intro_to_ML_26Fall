"""Recreate the recorded 20 MBPP task inputs without leaking answers to the agent."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from agent.tools import Workspace

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/mbpp/sanitized-mbpp.json'
SEED = 'course-mbpp-comparison-20-20261028'
EXCLUDE_IDS = {11, 462, 476, 172, 83}

def normalized(test):
    tree = ast.parse(test)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        raise ValueError('Expected a single assert statement')
    return ast.dump(tree, include_attributes=False)


def candidate(row):
    if not 11 <= row['task_id'] <= 510 or row['task_id'] in EXCLUDE_IDS or row['test_imports']:
        return None
    tests = row['test_list']
    try:
        norms = [normalized(t) for t in tests]
        if len(norms) < 3 or len(set(norms)) != len(norms):
            return None
        tree = ast.parse(row['code'])
        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        if len(defs) != 1 or any(isinstance(n, ast.ClassDef) for n in tree.body):
            return None
        fn = defs[0]
        a = fn.args
        if fn.decorator_list or fn.returns or a.defaults or a.kw_defaults or a.vararg or a.kwarg:
            return None
        if any(x.annotation for x in a.posonlyargs + a.args + a.kwonlyargs):
            return None
        if not all(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fn.name
                       for n in ast.walk(ast.parse(test))) for test in tests):
            return None
        signature = ast.unparse(a)
        compile(f'def {fn.name}({signature}):\n    raise NotImplementedError\n', '<stub>', 'exec')
        return {'task_id': row['task_id'], 'entry_point': fn.name, 'signature': signature,
                'public_tests': tests[:2], 'heldout_tests': tests[2:]}
    except (SyntaxError, ValueError, TypeError):
        return None


def test_file(entry, tests):
    lines = ['import unittest', f'from solution import {entry}', '', 'class MBPPTests(unittest.TestCase):']
    for i, test in enumerate(tests):
        lines += [f'    def test_case_{i + 1}(self):', '        ' + test]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True, help='New output directory outside task workspaces')
    ap.add_argument('--limit', type=int, default=20, help='First N frozen tasks (1–20)')
    args = ap.parse_args()
    if not 1 <= args.limit <= 20:
        ap.error('limit must be between 1 and 20')
    protocol = json.loads((ROOT / 'results/mbpp20/protocol.json').read_text())
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if digest != protocol['dataset_sha256']:
        raise SystemExit('Dataset hash mismatch')
    rows = json.loads(DATA.read_text())
    eligible = [(r, candidate(r)) for r in rows]
    eligible = [(r, c) for r, c in eligible if c is not None]
    eligible.sort(key=lambda x: hashlib.sha256(f"{SEED}|{x[0]['task_id']}".encode()).hexdigest())
    frozen = [t['task_id'] for t in protocol['tasks']]
    if len(eligible) != protocol['eligible_count'] or [r['task_id'] for r, _ in eligible[:20]] != frozen:
        raise SystemExit('Task selection does not match recorded protocol')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    entries = []
    for row, item in eligible[:args.limit]:
        taskid = str(item['task_id'])
        ws = out / 'inputs' / taskid
        (ws / 'tests').mkdir(parents=True)
        (ws / 'solution.py').write_text(f"def {item['entry_point']}({item['signature']}):\n    raise NotImplementedError('Implement this function')\n")
        (ws / 'tests/test_public.py').write_text(test_file(item['entry_point'], item['public_tests']))
        task = (row['prompt'] + f"\nImplement `{item['entry_point']}({item['signature']})` in solution.py. "
                'Use only the standard Python library. Run the public tests and report what you changed.')
        (ws / 'task.txt').write_text(task + '\n')
        private = out / 'private' / taskid
        private.mkdir(parents=True, mode=0o700)
        (private / 'test_heldout.py').write_text(test_file(item['entry_point'], item['heldout_tests']))
        with tempfile.TemporaryDirectory(prefix='mbpp-reference-') as tmp:
            d = Path(tmp)
            (d / 'tests').mkdir()
            (d / 'solution.py').write_text(row['code'])
            (d / 'tests/test_original.py').write_text(test_file(item['entry_point'], row['test_list']))
            result = Workspace(d).run_tests()
        (private / 'reference_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        entries.append({'task_id': item['task_id'], 'entry_point': item['entry_point'],
                        'reference_preflight': result['status']})
        print(f"{taskid}: reference {result['status']}", flush=True)
    bound = sorted([*out.glob('inputs/*/solution.py'), *out.glob('inputs/*/task.txt'),
                    *out.glob('inputs/*/tests/test_public.py'), *out.glob('private/*/test_heldout.py')])
    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound}
    metadata = {'dataset_sha256': digest, 'selection_seed': SEED, 'tasks': entries, 'file_sha256': hashes,
                'note': 'First N frozen tasks; held-out tests are public dataset material.'}
    (out / 'prepared.json').write_text(json.dumps(metadata, indent=2) + '\n')
    if any(e['reference_preflight'] != 'passed' for e in entries):
        raise SystemExit('Reference preflight failed; do not start model evaluation')

if __name__ == '__main__':
    main()
