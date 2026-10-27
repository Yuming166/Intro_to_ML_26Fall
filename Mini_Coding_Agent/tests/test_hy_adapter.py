"""Exercise the text-to-tool adapter with an in-process fake model service."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.request

from benchmarks.mbpp.hy_adapter import make_server, parse_answer, transform_messages


class AdapterTests(unittest.TestCase):
    def test_parse_and_history(self):
        allowed = {'list_files'}
        self.assertEqual(parse_answer('{"tool":"list_files","arguments":{}}', allowed)[0], 'tool')
        self.assertEqual(parse_answer('{"final":"done"}', allowed)[0], 'final')
        self.assertEqual(parse_answer('not json', allowed)[0], 'invalid_json')
        self.assertEqual(parse_answer('{"tool":"other","arguments":{}}', allowed)[0], 'invalid_schema')
        messages = [{'role': 'system', 'content': 'original instructions'},
                    {'role': 'assistant', 'tool_calls': [{'function': {'name': 'list_files', 'arguments': '{}'}}]},
                    {'role': 'tool', 'content': 'solution.py'}]
        converted = transform_messages(messages, [])
        self.assertIn('original instructions', [m['content'] for m in converted])
        self.assertEqual(json.loads(converted[2]['content'])['tool'], 'list_files')
        self.assertIn('solution.py', converted[3]['content'])

    def test_proxy_round_trip(self):
        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result = {'model': data['model'], 'choices': [{'message': {'content': '{"tool":"list_files","arguments":{}}'},
                          'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 2, 'completion_tokens': 3}}
                raw = json.dumps(result).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
        upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / 'trace.jsonl'
            adapter = make_server(trace, f'http://127.0.0.1:{upstream.server_port}/v1', port=0, model='fake')
            threading.Thread(target=adapter.serve_forever, daemon=True).start()
            try:
                data = {'model': 'fake', 'messages': [{'role': 'user', 'content': 'list files'}],
                        'tools': [{'type': 'function', 'function': {'name': 'list_files'}}],
                        'temperature': 0, 'max_tokens': 100}
                req = urllib.request.Request(f'http://127.0.0.1:{adapter.server_port}/v1/chat/completions',
                                             data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=5) as response:
                    result = json.load(response)
                self.assertEqual(result['model'], 'fake')
                self.assertEqual(result['choices'][0]['message']['tool_calls'][0]['function']['name'], 'list_files')
                self.assertEqual(json.loads(trace.read_text())['kind'], 'tool')
            finally:
                adapter.shutdown()
                adapter.server_close()
                upstream.shutdown()
                upstream.server_close()


if __name__ == '__main__':
    unittest.main()
