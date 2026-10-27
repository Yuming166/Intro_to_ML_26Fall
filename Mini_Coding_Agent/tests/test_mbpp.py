"""Offline checks for task isolation and input tampering; no model calls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MBPPTests(unittest.TestCase):
    def test_prepare_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            prepared = Path(tmp) / 'prepared'
            cmd = [sys.executable, '-m', 'benchmarks.mbpp.prepare', '--out', str(prepared), '--limit', '1']
            result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            metadata = json.loads((prepared / 'prepared.json').read_text())
            self.assertEqual([t['task_id'] for t in metadata['tasks']], [92])
            self.assertEqual(metadata['tasks'][0]['reference_preflight'], 'passed')
            source = prepared / 'inputs/92/solution.py'
            self.assertIn('NotImplementedError', source.read_text())
            self.assertFalse(any('heldout' in str(p) for p in (prepared / 'inputs/92').rglob('*')))
            self.assertTrue((prepared / 'private/92/test_heldout.py').exists())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                             metadata['file_sha256']['inputs/92/solution.py'])
            source.write_text(source.read_text() + '# modified\n')
            out = Path(tmp) / 'out'
            run = [sys.executable, '-m', 'benchmarks.mbpp.run', '--prepared', str(prepared), '--out', str(out),
                   '--4b-url', 'http://127.0.0.1:1/v1', '--35b-url', 'http://127.0.0.1:1/v1']
            result = subprocess.run(run, cwd=ROOT, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('changed', result.stderr)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
