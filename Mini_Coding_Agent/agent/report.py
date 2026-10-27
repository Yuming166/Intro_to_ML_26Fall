"""Render an offline HTML report; no network assets or external service needed."""
import html
import json
from pathlib import Path


def esc(value):
    return html.escape(str(value), quote=True)


def render(run):
    run = Path(run)
    result = json.loads((run / 'result.json').read_text())
    events = [json.loads(line) for line in (run / 'trace.jsonl').read_text().splitlines()]
    cards = []
    for event in events:
        kind = event.get('type')
        if kind == 'tool':
            title = f"Step {event.get('step')} · {event.get('name')} · {event.get('outcome')}"
            body = f"Arguments: {event.get('arguments')}\n\nResult:\n{event.get('result')}"
        elif kind == 'model':
            title = f"Step {event.get('step')} · model · {event.get('elapsed_s')} s"
            message = event.get('message') or {}
            body = json.dumps({'content': message.get('content'), 'tool_calls': message.get('tool_calls'),
                               'usage': event.get('usage')}, ensure_ascii=False, indent=2)
        elif kind in ('verification', 'security_review', 'error', 'context_compacted'):
            title, body = kind, json.dumps(event, ensure_ascii=False, indent=2)
        else:
            continue
        cards.append(f'<details><summary>{esc(title)}</summary><pre>{esc(body)}</pre></details>')
    diff = (run / 'changes.diff').read_text()
    initial = result['initial_tests']['status']
    final = result['final_tests']['status']
    security = result.get('security_review')
    initial_security = result.get('initial_security')
    security_status = security.get('status', 'error') if security else 'not_recorded'
    findings = security.get('findings', []) if security else []
    security_rows = ''.join(
        f"<tr><td>{esc(item.get('file'))}:{esc(item.get('line'))}</td><td>{esc(item.get('rule'))}</td>"
        f"<td>{esc(item.get('severity'))}</td><td>{esc(item.get('confidence'))}</td>"
        f"<td>{esc(item.get('message'))}</td></tr>" for item in findings)
    security_table = ('<table><thead><tr><th>位置</th><th>规则</th><th>严重度</th><th>置信度</th><th>说明</th>'
                      '</tr></thead><tbody>' + security_rows + '</tbody></table>') if security_rows else '<p>没有记录的发现；请结合审查状态判断。</p>'
    security_errors = '\n'.join(security.get('errors', [])) if security else '该运行创建于接入安全审查之前。'
    if initial_security:
        before_findings = initial_security.get('findings', [])
        before_text = '\n'.join(f"{item.get('file')}:{item.get('line')}  {item.get('rule')}  "
                                f"{item.get('severity')} / {item.get('confidence')}  {item.get('message')}"
                                for item in before_findings) or '没有记录的发现；请结合审查状态判断。'
        before_section = (f"<details><summary>修改前：{esc(initial_security.get('status'))} / {len(before_findings)} 条发现</summary>"
                          f"<pre>{esc(before_text)}\n{esc('; '.join(initial_security.get('errors', [])))}</pre></details>")
    else:
        before_section = ''
    output = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini Coding Agent · 运行轨迹</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;background:#f5f7fb;color:#19253b;margin:0}}
main{{max-width:1100px;margin:auto;padding:32px}}h1{{margin:0}}.subtitle{{color:#53627a}}
.stats{{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}}.stat,details{{background:white;border:1px solid #dce3ef;border-radius:12px;padding:14px 18px}}
.stat strong{{display:block;font-size:20px;color:#1755b5}} details{{margin:10px 0}}summary{{cursor:pointer;font-weight:650}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6fa;padding:16px;border-radius:8px;font-size:13px}}
.tag{{font-size:13px;color:#557}}h2{{margin-top:30px}}table{{width:100%;border-collapse:collapse;background:white}}
th,td{{padding:9px;text-align:left;border-bottom:1px solid #dce3ef;vertical-align:top}}</style><main>
<h1>Mini Coding Agent</h1><div class="subtitle">Qwen 工具调用与测试反馈 · 可离线查看的真实运行记录</div>
<div class="stats"><div class="stat"><span>运行状态</span><strong>{esc(result['status'])}</strong></div>
<div class="stat"><span>初始 → 最终测试</span><strong>{esc(initial)} → {esc(final)}</strong></div>
<div class="stat"><span>模型</span><strong>{esc(result['model'])}</strong></div>
<div class="stat"><span>安全审查状态 / 发现数</span><strong>{esc(security_status)} / {len(findings)}</strong></div>
<div class="stat"><span>输入 / 输出 token</span><strong>{esc(result['input_tokens'])} / {esc(result['output_tokens'])}</strong></div></div>
<p class="tag">公开测试通过仅说明这些测试通过；时间与 token 来自该次运行。模型可能失败或给出错误报告。</p>
<h2>Security Review</h2><p class="tag">Bandit 静态扫描：发现需要人工判断；没有发现不等于代码安全，扫描失败或不完整不等于通过。</p>
{before_section}<p>最终审查：{esc(security_status)}</p>{security_table}<p>{esc(security_errors)}</p>
<h2>代码改动</h2><pre>{esc(diff)}</pre><h2>模型最终说明</h2><pre>{esc(result['final_text'])}</pre>
<h2>工具与验证轨迹</h2>{''.join(cards)}</main></html>'''
    (run / 'report.html').write_text(output)


if __name__ == '__main__':
    import sys
    render(sys.argv[1])
