"""Read-only Bandit review; finding status is separate from functional tests."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
MAX_FILES = 200
MAX_FILE_BYTES = 200_000
MAX_TOTAL_BYTES = 5_000_000


def review(root, python=None):
    root = Path(root).resolve(strict=True)
    python = str(python or os.getenv('COURSE_AGENT_BANDIT_PYTHON') or
                 (ROOT / '.venv/bin/python' if (ROOT / '.venv/bin/python').exists() else sys.executable))
    base = {'scanner': 'bandit', 'scanner_version': None, 'status': 'error', 'files_considered': 0,
            'findings': [], 'errors': []}
    try:
        entries = list(root.rglob('*'))
        if any(p.is_symlink() for p in entries):
            return {**base, 'status': 'incomplete', 'errors': ['Workspace contains a symlink; review aborted.']}
        files = sorted(p for p in entries if p.is_file() and p.suffix == '.py' and
                       not ({'.git', '__pycache__', '.venv'} & set(p.relative_to(root).parts)))
        oversized = [str(p.relative_to(root)) for p in files if p.stat().st_size > MAX_FILE_BYTES]
        if len(files) > MAX_FILES or oversized or sum(p.stat().st_size for p in files) > MAX_TOTAL_BYTES:
            return {**base, 'status': 'incomplete', 'errors': [
                f'Review limit exceeded: {len(files)} Python files; oversized: {oversized[:10]}']}
        if not files:
            return {**base, 'status': 'incomplete', 'errors': ['No Python files to scan.']}
        # Explicit files and a trusted cwd avoid loading the workspace's .bandit config.
        # --ignore-nosec prevents source comments from suppressing findings.
        cmd = [python, '-I', '-m', 'bandit', '--ignore-nosec', '-q', '-f', 'json', *map(str, files)]
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1'}
        try:
            version = subprocess.run([python, '-I', '-c',
                                      'from importlib.metadata import version; print(version("bandit"))'],
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=5)
            if version.returncode == 0:
                base['scanner_version'] = version.stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        if 'No module named bandit' in proc.stderr:
            return {**base, 'status': 'unavailable', 'errors': ['Bandit is not installed in the selected Python environment.']}
        if proc.returncode not in (0, 1):
            detail = proc.stderr[-1000:].strip()
            return {**base, 'status': 'error',
                    'errors': [detail or f'Bandit exited with code {proc.returncode}']}
        try:
            report = json.loads(proc.stdout)
            raw_findings = report['results']
            raw_errors = report['errors']
            if not isinstance(raw_findings, list) or not isinstance(raw_errors, list):
                raise ValueError('results and errors must be lists')
            findings = []
            for item in raw_findings:
                location = Path(item['filename']).resolve().relative_to(root)
                message = ('Possible hardcoded credential; inspect the source locally.'
                           if item['test_id'] in {'B105', 'B106', 'B107'} else item['issue_text'])
                findings.append({'file': str(location), 'line': item['line_number'],
                                 'rule': item['test_id'], 'severity': item['issue_severity'],
                                 'confidence': item['issue_confidence'], 'message': message,
                                 'reference': item.get('more_info', '')})
            # Record scan errors, never convert a partial scan into no_findings.
            errors = []
            for item in raw_errors:
                name = item.get('filename', '')
                if name:
                    try:
                        name = str(Path(name).resolve().relative_to(root))
                    except ValueError:
                        name = '<outside workspace>'
                errors.append((name + ': ' + str(item.get('reason', item)))[:500])
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            return {**base, 'status': 'error', 'errors': [f'Invalid Bandit output: {exc}']}
        status = 'incomplete' if errors else ('findings' if findings else 'no_findings')
        return {**base, 'status': status, 'files_considered': len(files), 'findings': findings,
                'errors': errors, 'exit_code': proc.returncode}
    except FileNotFoundError as exc:
        return {**base, 'status': 'unavailable', 'errors': [str(exc)]}
    except subprocess.TimeoutExpired:
        return {**base, 'status': 'error', 'errors': ['Bandit exceeded the 30-second time limit.']}
    except (OSError, ValueError) as exc:
        return {**base, 'status': 'error', 'errors': [str(exc)]}
