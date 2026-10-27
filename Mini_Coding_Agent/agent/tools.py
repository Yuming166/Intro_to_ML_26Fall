"""Bounded file tools, edit checkpoints, and isolated test execution."""
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .security import review

MAX_OUTPUT = 6000


def short(text):
    return text if len(text) <= MAX_OUTPUT else text[:MAX_OUTPUT] + '\n...[output truncated]'


def schema(name, description, props=None, required=None):
    return {'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': props or {},
                           'required': required or [], 'additionalProperties': False}}}


STRING = {'type': 'string'}
SCHEMAS = [
    schema('list_files', 'List the project files.'),
    schema('search_code', 'Search for a literal string in project text files.', {'query': STRING}, ['query']),
    schema('read_file', 'Read a project file with line numbers.', {'path': STRING}, ['path']),
    schema('replace_code', 'Replace exactly one matching code fragment in a Python source file; old must match exactly.',
           {'path': STRING, 'old': STRING, 'new': STRING}, ['path', 'old', 'new']),
    schema('create_file', 'Create a new Python source file in an existing directory outside tests/.',
           {'path': STRING, 'content': STRING}, ['path', 'content']),
    schema('update_plan', 'Record a short, observable task plan. Update it when progress changes.',
           {'plan': {'type': 'array', 'items': STRING}}, ['plan']),
    schema('run_tests', 'Run public unit tests in the isolated workspace copy.'),
    schema('security_review', 'Run read-only Bandit static security review of Python files. Findings need human review.'),
    schema('show_diff', 'Show code changes made in this run.'),
    schema('rollback', 'Undo the most recent successful code edit.'),
]


class Workspace:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.original = {str(p.relative_to(self.root)): p.read_text() for p in self.root.rglob('*.py')
                         if p.is_file() and not p.is_symlink()}
        self.history = []
        self.plan = []

    def path(self, name):
        if not isinstance(name, str) or not name or len(name) > 300:
            raise ValueError('Invalid path')
        path = self.root / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError('Path must be inside workspace and cannot be a symlink')
        if not path.is_file():
            raise ValueError('File does not exist')
        if path.stat().st_size > 100_000:
            raise ValueError('File is too large')
        return path

    def list_files(self):
        return '\n'.join(str(p.relative_to(self.root)) for p in sorted(self.root.rglob('*'))
                         if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts)[:MAX_OUTPUT]

    def search_code(self, query):
        if not query or len(query) > 200:
            raise ValueError('Query must contain 1-200 characters')
        matches = []
        for p in sorted(self.root.rglob('*')):
            if p.is_file() and not p.is_symlink() and p.suffix in ('.py', '.md', '.txt') and p.stat().st_size < 100_000:
                for number, line in enumerate(p.read_text(errors='replace').splitlines(), 1):
                    if query.lower() in line.lower():
                        matches.append(f'{p.relative_to(self.root)}:{number}: {line[:300]}')
                        if len(matches) >= 40:
                            return short('\n'.join(matches))
        return short('\n'.join(matches) or 'No matches.')

    def read_file(self, path):
        p = self.path(path)
        lines = p.read_text().splitlines()
        return short('\n'.join(f'{i:4d} | {line}' for i, line in enumerate(lines, 1)))

    def replace_code(self, path, old, new):
        p = self.path(path)
        if p.suffix != '.py' or 'tests' in p.relative_to(self.root).parts:
            raise ValueError('Only Python source files outside tests/ may be edited')
        if not old or len(old) > 30000 or len(new) > 30000:
            raise ValueError('Invalid edit length')
        content = p.read_text()
        count = content.count(old)
        if count != 1:
            raise ValueError(f'old must match exactly once; matched {count} times. Re-read the file.')
        self.history.append((p, content))
        p.write_text(content.replace(old, new, 1))
        return f'Updated {path}. Run tests to verify.'

    def rollback(self):
        if not self.history:
            return 'No edit to roll back.'
        p, old = self.history.pop()
        if old is None:
            p.unlink()
        else:
            p.write_text(old)
        return f'Rolled back {p.relative_to(self.root)}.'

    def create_file(self, path, content):
        if not isinstance(path, str) or not path or len(path) > 300 or len(content) > 30000:
            raise ValueError('Invalid file name or length')
        p = self.root / path
        if p.suffix != '.py' or 'tests' in Path(path).parts or not p.resolve().is_relative_to(self.root):
            raise ValueError('Only Python source files outside tests/ may be created')
        if p.exists() or p.is_symlink() or not p.parent.is_dir():
            raise ValueError('File must not exist and parent directory must exist')
        p.write_text(content)
        self.history.append((p, None))
        return f'Created {path}. Run tests to verify.'

    def update_plan(self, plan):
        if not isinstance(plan, list) or not 1 <= len(plan) <= 8 or any(not isinstance(x, str) or len(x) > 300 for x in plan):
            raise ValueError('Plan must contain 1-8 short strings')
        self.plan = plan
        return 'Plan updated:\n' + '\n'.join(f'{i}. {item}' for i, item in enumerate(plan, 1))

    def full_diff(self):
        changes = []
        current = {str(p.relative_to(self.root)): p for p in self.root.rglob('*.py') if p.is_file() and not p.is_symlink()}
        for name in sorted(set(self.original) | set(current)):
            before = self.original.get(name, '')
            p = self.root / name
            if p.is_file():
                after = p.read_text()
                if before != after:
                    changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                                       fromfile='before/'+name, tofile='after/'+name))
        return ''.join(changes) or 'No changes.'

    def show_diff(self):
        return short(self.full_diff())

    def run_tests(self):
        with tempfile.TemporaryDirectory(prefix='course-agent-test-') as tmp:
            base = Path(tmp)
            project, scratch = base / 'project', base / 'scratch'
            # Do not follow links from an untrusted workspace.
            for p in self.root.rglob('*'):
                if p.is_symlink():
                    return {'ok': False, 'status': 'invalid_workspace', 'output': 'Symlinks are not allowed.'}
            shutil.copytree(self.root, project, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.git'))
            scratch.mkdir()
            runner = Path(__file__).with_name('sandbox.py').resolve()
            env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'PYTHONNOUSERSITE': '1'}
            try:
                with (base / 'output.log').open('w+') as stream:
                    result = subprocess.run(['/usr/bin/python3', '-I', '-B', str(runner), str(project), str(scratch)],
                                            stdout=stream, stderr=subprocess.STDOUT, timeout=20, env=env,
                                            start_new_session=True)
                    stream.seek(0)
                    output = short(stream.read(MAX_OUTPUT + 1).replace(str(project) + '/', ''))
            except subprocess.TimeoutExpired:
                return {'ok': False, 'status': 'timeout', 'output': 'Tests exceeded 20 seconds.'}
            status = ('passed' if result.returncode == 0 else
                      'sandbox_error' if result.returncode == 125 else
                      'no_tests' if result.returncode == 2 else 'failed')
            return {'ok': result.returncode == 0, 'status': status, 'exit_code': result.returncode, 'output': output}

    def security_review(self):
        return review(self.root)

    def call(self, name, args):
        allowed = {'list_files', 'search_code', 'read_file', 'replace_code', 'create_file', 'update_plan',
                   'run_tests', 'security_review', 'show_diff', 'rollback'}
        if name not in allowed or not isinstance(args, dict):
            raise ValueError('Unknown tool or invalid arguments')
        return getattr(self, name)(**args)
