import json
from pathlib import Path
import tempfile
import unittest

from agent.security import review
from agent.report import render


class SecurityReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_findings_and_known_issue(self):
        (self.root / 'clean.py').write_text('def double(x):\n    return x * 2\n')
        clean = review(self.root)
        self.assertEqual(clean['status'], 'no_findings', clean)
        (self.root / 'unsafe.py').write_text('def parse(x):\n    return eval(x)  # nosec\n')
        result = review(self.root)
        self.assertEqual(result['status'], 'findings', result)
        finding = next(item for item in result['findings'] if item['rule'] == 'B307')
        self.assertEqual((finding['file'], finding['line']), ('unsafe.py', 2))
        self.assertEqual((finding['severity'], finding['confidence']), ('MEDIUM', 'HIGH'))

    def test_secret_values_not_copied_into_findings(self):
        marker = 'not-a-real-secret-test-value'
        (self.root / 'unsafe.py').write_text('password = ' + repr(marker) + '\n')
        result = review(self.root)
        self.assertTrue(any(f['rule'] == 'B105' for f in result['findings']), result)
        self.assertNotIn(marker, json.dumps(result))

    def test_unavailable_and_incomplete_are_not_clean(self):
        (self.root / 'bad.py').write_text('def broken(:\n')
        missing = review(self.root, python='/definitely/not/a/python')
        self.assertEqual(missing['status'], 'unavailable', missing)
        malformed = review(self.root)
        self.assertEqual(malformed['status'], 'incomplete', malformed)
        self.assertTrue(malformed['errors'])

    def test_symlink_rejected(self):
        (self.root / 'alias.py').symlink_to('/etc/passwd')
        result = review(self.root)
        self.assertEqual(result['status'], 'incomplete')

    def test_report_displays_separate_result_and_escapes_html(self):
        result = {'status': 'verified_public_tests', 'model': 'model', 'initial_tests': {'status': 'failed'},
                  'final_tests': {'status': 'passed'}, 'input_tokens': 1, 'output_tokens': 1, 'final_text': '',
                  'security_review': {'status': 'findings', 'errors': [], 'findings': [
                      {'file': 'app.py', 'line': 2, 'rule': 'B307', 'severity': 'MEDIUM',
                       'confidence': 'HIGH', 'message': '<script>alert(1)</script>'}]}}
        (self.root / 'result.json').write_text(json.dumps(result))
        (self.root / 'trace.jsonl').write_text('')
        (self.root / 'changes.diff').write_text('No changes')
        render(self.root)
        report = (self.root / 'report.html').read_text()
        self.assertIn('findings / 1', report)
        self.assertIn('B307', report)
        self.assertNotIn('<script>alert(1)</script>', report)
        self.assertIn('&lt;script&gt;', report)
