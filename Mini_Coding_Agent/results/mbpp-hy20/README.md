# Hy-MT2-7B 跨模型家族探索测试

在先前固定的同一组 20 道 MBPP sanitized 题上，对 Hy-MT2-7B-FP8 每题运行一次。
原始 4B/35B 结果来自此前独立运行。公开测试为每题前两条断言，保留测试为其余断言。

Hy 服务的原生 auto tool choice 返回 HTTP 400，因此使用 JSON 文本工具适配层，Qwen 使用原生工具调用。
不同的模型家族、模型用途、接口适配、时间及服务部署使结果不是严格受控的模型能力对比。

| 模型 | 通过 | 失败 | 未知 | 任务平均耗时 | 任务耗时中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 17/20 | 3 | 0 | 7.08 s | 5.86 s |
| Qwen3.6-35B-A3B-FP8 | 20/20 | 0 | 0 | 5.49 s | 4.90 s |
| Hy-MT2-7B-FP8 + JSON 适配 | 5/20 | 15 | 0 | 24.82 s | 24.08 s |

三者均通过：5/20。独立保留测试复核：20/20 一致。
任务耗时包含模型请求、排队、agent、公开测试、安全检查和 HTML 报告；不含保留测试重跑。
MBPP 是公开数据，保留断言不代表模型未见过；样本量小、每题仅一次，不适合推断总体性能。

| 题号 | 4B | 35B | Hy-MT2 | Hy 耗时（秒） |
| ---: | --- | --- | --- | ---: |
| 92 | passed | passed | failed | 22.77 |
| 305 | failed | passed | failed | 20.72 |
| 472 | passed | passed | failed | 30.45 |
| 290 | passed | passed | passed | 30.19 |
| 430 | failed | passed | failed | 33.95 |
| 309 | passed | passed | failed | 22.47 |
| 426 | passed | passed | passed | 14.25 |
| 260 | failed | passed | failed | 41.52 |
| 408 | passed | passed | failed | 33.40 |
| 269 | passed | passed | passed | 5.88 |
| 166 | passed | passed | failed | 26.64 |
| 299 | passed | passed | failed | 18.51 |
| 448 | passed | passed | failed | 24.88 |
| 111 | passed | passed | failed | 27.54 |
| 132 | passed | passed | passed | 18.41 |
| 390 | passed | passed | passed | 20.37 |
| 103 | passed | passed | failed | 33.80 |
| 273 | passed | passed | failed | 28.49 |
| 477 | passed | passed | failed | 18.76 |
| 424 | passed | passed | failed | 23.27 |

## 失败和安全检查

Hy 共 16 次达到步数上限，正常结束 4 次。测试通过但达到上限的题目：[132]。
最终代码存在语法错误的题号：[430, 309, 260, 408, 273, 477, 424]；最终仍含原始未实现存根的题号：[92, 472, 166, 299, 448, 111, 103]。
发生 29 次工具调用错误；以上运行中错误均为 replace_code 的原文本未匹配。
Bandit 状态：{'findings': 13, 'incomplete': 7}。7 次 incomplete 均由于代码语法错误，不能当作安全检查通过。
题目 305 返回列表但测试要求元组；题目 430 生成缩进错误；题目 92 多次修改后回滚到原始存根。

适配器输出：{"tool": 217, "invalid_schema": 3, "final": 7, "invalid_json": 3}。
逐题对照见 [`comparison.csv`](comparison.csv)，Hy 的完整记录见 [`episodes.csv`](episodes.csv) 与 [`audit.json`](audit.json)。示例轨迹：[92](../../examples/trajectories/hy-92.html)、[132](../../examples/trajectories/hy-132.html)、[290](../../examples/trajectories/hy-290.html)、[430](../../examples/trajectories/hy-430.html)。
