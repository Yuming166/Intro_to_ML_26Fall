# Mini Coding Agent：本地模型与 MBPP 教学示例

这是一个面向本科高年级课程的可读、可修改的 coding agent demo。它使用 OpenAI 兼容接口调用本地模型，提供文件查看和编辑、测试反馈、失败后调试、静态安全审查和 HTML 轨迹。本项目直接实现 agent 循环，**不依赖 LangChain**；学生也可以自行选择 LangChain、其他框架或完全不同的设计。

本仓库包括原始 MBPP sanitized 数据（427 条，约 249 KiB）、两次实际模型实验的记录和可重新运行的评测脚本。模型权重和推理服务需要自行准备；仓库不会自动下载或启动模型。

## 工作过程

```text
任务 + 工作区 -> 模型选择工具 -> 读取/修改代码 -> 测试反馈 -> 继续修改
                                          |               |
                                          +-> 静态安全审查 +-> 最终测试与 HTML 报告
```

示例工具包括 `list_files`、`search_code`、`read_file`、`replace_code`、`create_file`、`run_tests`、`security_review`、`rollback` 和 `update_plan`。测试失败时 agent 可以继续修改；模型能否最终解决问题依赖任务和模型。报告区分测试结果和 Bandit 审查结果。

## 环境与快速开始

- Linux x86-64；Landlock ABI ≥ 3、`libseccomp.so.2`、`/usr/bin/python3`。测试执行使用 Landlock、seccomp 和资源限制；不满足条件时会报错，不会静默降级。建议以普通用户运行。
- Python 3.9 及以上（实际验证使用 Python 3.13）与 `venv`、`pip`。
- 已运行的 OpenAI 兼容 `/v1/chat/completions` 服务，须支持原生 tool calling、`temperature` 和 `chat_template_kwargs`；本 demo 要求返回的 `model` 字段与请求 ID 一致。不同服务的兼容性和具体模型 ID 需要自行核对。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v

# 根据本地服务修改地址和模型名；需要认证时另设 COURSE_AGENT_API_KEY。
export COURSE_AGENT_BASE_URL=http://127.0.0.1:8000/v1
export COURSE_AGENT_MODEL=Qwen3.5-4B
.venv/bin/python -m agent.cli \
  --workspace examples/metrics \
  --task "$(cat examples/metrics/task.txt)"
```

运行后会在 `runs/` 下创建新目录，包含 agent 的工作区副本、`trace.jsonl`、代码差异、测试与安全审查结果、`report.html`。原示例不会被修改。轨迹可能含本机路径、服务地址、题目或代码，公开前请自行检查；不要把 API key 写入仓库。

其他示例：`examples/text_analyzer/`、`examples/security_review/`。可直接打开 `examples/trajectories/` 中保存的实际 HTML 报告，观察失败与修复过程。

## MBPP 双模型实验记录

此前选取 20 道题，每题让两个模型各尝试一次：每题前两条原始断言提供给 agent，其余断言在 agent 结束后另行测试。没有重试或替换；参考解预检 20/20 通过。主结果要求公开测试和保留测试都通过。

| 指标 | Qwen3.5-4B | Qwen3.6-35B-A3B-FP8 |
| --- | ---: | ---: |
| 两组测试均通过 | 17/20 | 20/20 |
| 未通过 | 3/20 | 0/20 |
| 完整任务平均耗时 | 7.08 秒 | 5.49 秒 |
| 完整任务耗时中位数 | 5.86 秒 | 4.90 秒 |

耗时涵盖 agent 进程启动、服务请求与排队、公开测试、静态扫描和报告生成，不含结束后的保留测试。模型版本、服务和量化方式不同，硬件没有受控；这些数字不能用来推断纯模型速度或参数规模的因果影响。只有 20 道探索性题目，且每题只有一条保留断言，测试通过不代表全面正确。

详细结果见 [20 题结果说明](results/mbpp20/README.md)、[逐题 CSV](results/mbpp20/episodes.csv)、[实验协议](results/mbpp20/protocol.json) 和 [汇总](results/mbpp20/summary.json)。另有较早的 [5 题试运行](results/mbpp5/README.md)，单独保存，不计入上述指标。

MBPP 是公开数据：虽然保留断言未放进 agent 工作区，仓库内的数据集包含原始答案和全部断言，不能当作保密评测集，也无法排除模型预训练中见过它们。建议课堂互评准备另外的、自行设计的测试任务。

## 复跑两模型评测

先配置好两个服务。下面只准备原 20 道题中的第 1 道，适合快速试运行。去掉 `--limit 1` 可准备完整的 20 道。

```bash
.venv/bin/python -m benchmarks.mbpp.prepare --out runs/mbpp-prepared --limit 1
.venv/bin/python -m benchmarks.mbpp.run \
  --prepared runs/mbpp-prepared --out runs/mbpp-comparison \
  --4b-url http://127.0.0.1:8000/v1 --4b-model Qwen3.5-4B \
  --35b-url http://127.0.0.1:8001/v1 --35b-model Qwen3.6-35B-A3B
```

两个输出目录必须是新目录且互不包含；如需再次运行，请使用新的目录名。运行串行、交替模型先后顺序，默认最多 12 步，每次输出最多 1024 tokens，每题每模型 240 秒。地址和模型名应改为实际值。`COURSE_AGENT_API_KEY` 供两个服务共用；不同认证配置需要自行扩展。结果记录在 `episodes.csv`、`episodes.jsonl` 和 `summary.json`；超时、模型故障和验证不可用归为 `unknown`，不会算作通过。具体实现、依赖版本或服务配置变化可能导致新结果不同于历史结果。

## 课堂使用和安全边界

可以让学生观察一条实际轨迹，比较不同工具、反馈策略和测试设计，再自主选择目标、框架、评测方式和展示内容；不必照搬本 demo。项目可以互评，但应公开评测规则并区分正确性、耗时、错误处理和安全性。

Bandit 是基于规则的静态检查：发现需要人工判断，`no_findings` 不证明代码安全。MBPP 公共测试中的 `assert` 可能触发 B101，不能据此说生成代码有漏洞。测试 runner 有文件、系统调用及资源限制，但整个 agent 及模型服务不是经过安全审计的多租户隔离系统；不应在含敏感文件的账户下执行不可信代码或直接作为公网服务。生产或多人部署需要独立审核和更强隔离。

## 目录与许可

- `agent/`：工具、控制循环、沙箱、安全审查、报告。
- `examples/`：示例任务和部分真实轨迹。
- `benchmarks/mbpp/`：题目准备和双模型评测。
- `data/mbpp/`：完整的 sanitized MBPP，来源、署名和许可证见 [数据说明](data/mbpp/README.md)。
- `results/`：历史协议、汇总和逐题结果；`tests/`：项目测试。

本仓库示例代码按 [MIT License](LICENSE) 提供；MBPP 数据按其原许可 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 提供，见 [数据说明](data/mbpp/README.md)。
