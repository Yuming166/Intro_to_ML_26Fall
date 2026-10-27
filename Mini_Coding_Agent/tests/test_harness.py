import os
from pathlib import Path
import tempfile
import unittest

from agent.tools import Workspace
from agent.cli import compact_context


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'tests').mkdir()
        (self.root / 'app.py').write_text('value = 1\n')

    def tearDown(self):
        self.tmp.cleanup()

    def test_edits_and_rollback(self):
        ws = Workspace(self.root)
        ws.replace_code('app.py', 'value = 1', 'value = 2')
        self.assertIn('+value = 2', ws.show_diff())
        ws.rollback()
        self.assertEqual((self.root / 'app.py').read_text(), 'value = 1\n')
        ws.create_file('new.py', 'x = 1\n')
        self.assertIn('+x = 1', ws.show_diff())
        ws.rollback()
        self.assertFalse((self.root / 'new.py').exists())
        with self.assertRaises(ValueError):
            ws.replace_code('app.py', 'missing', 'x')

    def test_workspace_boundary(self):
        ws = Workspace(self.root)
        with self.assertRaises(ValueError):
            ws.read_file('/etc/passwd')
        (self.root / 'tests' / 'test_a.py').write_text('test data')
        with self.assertRaises(ValueError):
            ws.replace_code('tests/test_a.py', 'test', 'evil')

    def test_test_runner_sandbox(self):
        (self.root / 'tests' / 'test_boundary.py').write_text('''
import os, socket, subprocess, unittest
class Boundary(unittest.TestCase):
    def test_outside_read(self):
        with self.assertRaises(PermissionError):
            open('/etc/passwd').read()
    def test_network(self):
        with self.assertRaises(PermissionError):
            socket.socket()
    def test_subprocess(self):
        with self.assertRaises(PermissionError):
            subprocess.run(['/bin/true'])
    def test_project_readonly(self):
        with self.assertRaises(PermissionError):
            open('app.py', 'w')
''')
        result = Workspace(self.root).run_tests()
        self.assertTrue(result['ok'], result)
        self.assertEqual((self.root / 'app.py').read_text(), 'value = 1\n')

    def test_empty_test_suite_does_not_pass(self):
        result = Workspace(self.root).run_tests()
        self.assertFalse(result['ok'])
        self.assertIn('No tests collected', result['output'])

    def test_compaction_keeps_tool_pairs(self):
        ws = Workspace(self.root)
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'task'}]
        for i in range(6):
            messages.extend([{'role': 'assistant', 'content': None, 'tool_calls': [{'id': str(i)}]},
                             {'role': 'tool', 'tool_call_id': str(i), 'content': 'x' * 100}])
        compacted, changed = compact_context(messages, ws, limit=1, keep=2)
        self.assertTrue(changed)
        self.assertEqual(compacted[-4]['tool_calls'][0]['id'], compacted[-3]['tool_call_id'])
        self.assertEqual(compacted[-2]['tool_calls'][0]['id'], compacted[-1]['tool_call_id'])
