# Mini Coding Agent

本项目是《机器学习导论》课程Final Project的一个参考实现示例：让本地大语言模型使用工具阅读和修改 Python 代码，运行测试，并根据测试结果继续调试。项目中还加入了静态安全检查，运行结束后会生成一份 HTML 报告，方便查看模型的操作过程，供大家参考。

示例使用 Python 实现 agent 的基本循环，没有依赖 LangChain。大家可以在此基础上修改，也可以使用 LangChain 或其他工具，从自己感兴趣的方向设计 coding agent。

## Agent 做了什么

给定一个代码目录和一段任务描述，agent 会在工作区副本中执行以下操作：

1. 查看文件和代码，确定需要修改的位置；
2. 使用工具修改代码、运行测试，根据测试反馈继续调整；
3. 使用 Bandit 检查 Python 文件，并记录检查结果；
4. 输出代码差异、测试结果和运行报告。

可用工具包括文件查看、代码搜索、代码修改、运行测试、安全检查和撤销修改等，具体实现见 [`agent/tools.py`](agent/tools.py)。模型可能无法完成任务，也可能反复修改后仍然失败；可以结合报告观察它在什么地方遇到了问题。

## 运行环境

运行示例需要：

- Linux x86-64，支持 Landlock ABI 3 或以上，并安装 `libseccomp.so.2` 和 `/usr/bin/python3`；
- Python 3.9 或以上，以及 `venv`、`pip`；
- 一个已启动的 OpenAI 兼容模型服务，支持 tool calling 和 `chat_template_kwargs`。模型服务返回的 `model` 字段需要与请求的模型名称一致。

项目不包含模型权重，也不会自动启动模型服务。以下是本次实验所用模型的部署情况，供参考：

| 模型 | 实验中的部署情况 |
| --- | --- |
| Qwen3.5-4B | 本机通过 vLLM 服务调用，运行在 1 张 NVIDIA GeForce RTX 4090 上；该卡显存容量约 22.5 GiB。 |
| Qwen3.6-35B-A3B-FP8 | 调用另一台服务器上的 vLLM 服务，运行在 1 张 NVIDIA A100 GPU 上，显存 80 GB。 |
| Hy-MT2-7B-FP8 | 本机通过 vLLM 服务调用，运行在 1 张 NVIDIA GeForce RTX 4090 上；该卡显存容量约 22.5 GiB。 |


## 运行示例

进入本目录，创建环境并安装依赖：

```bash
cd Mini_Coding_Agent
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

启动模型服务后，按实际情况设置服务地址和模型名称，再运行一个示例：

```bash
export COURSE_AGENT_BASE_URL=http://127.0.0.1:8000/v1
export COURSE_AGENT_MODEL=Qwen3.5-4B
# 如果服务需要认证，还需要设置 COURSE_AGENT_API_KEY。

.venv/bin/python -m agent.cli \
  --workspace examples/metrics \
  --task "$(cat examples/metrics/task.txt)"
```

每次运行都会在 `runs/` 下新建目录，保存工作区副本、模型和工具调用记录、代码差异、测试结果、安全检查结果以及 `report.html`。原始示例目录不会被修改。发布自己的运行记录前，请检查其中是否包含本机路径、服务地址或其他不适合公开的内容。

`examples/` 中还有文本处理和安全检查示例。也可以直接打开 [`examples/trajectories/`](examples/trajectories/) 中已有的 HTML 报告，查看模型修改代码和处理测试反馈的过程。

## MBPP 测试

项目附带了 [MBPP sanitized 数据集](data/mbpp/README.md)和三种模型的测试记录。测试从适配当前 agent 的题目中选取了 20 道；每题将前两条原始断言交给 agent，剩余断言在运行结束后单独测试。每个模型每题各运行一次，没有重试或替换题目。

| 指标 | Qwen3.5-4B | Qwen3.6-35B-A3B-FP8 | Hy-MT2-7B-FP8 |
| --- | ---: | ---: | ---: |
| 公开测试和保留测试均通过 | 17/20 | 20/20 | 5/20 |
| 至少一组测试未通过 | 3/20 | 0/20 | 15/20 |
| 正常结束 | 17/20 | 20/20 | 4/20 |
| 平均任务耗时 | 7.08 秒 | 5.49 秒 | 24.82 秒 |
| 任务耗时中位数 | 5.86 秒 | 4.90 秒 | 24.08 秒 |

耗时从 agent 进程启动计至退出，包括模型请求及排队、公开测试、安全检查和报告生成，不包括结束后的保留测试。三次实验的模型、部署环境和运行时间不同，不宜据此比较模型的纯推理速度。每题只有一条保留断言，通过测试也不能保证对所有输入都正确。

Hy-MT2 是翻译模型。实验时其服务没有启用原生自动工具调用，因此使用 JSON 文本适配层转发工具请求；两款 Qwen 使用原生工具调用。这些结果可用来观察不同模型和接口的表现，但不是严格控制条件的模型能力排名。Hy 有 16 次达到步数上限，其中题目 132 的最终测试通过，但 agent 没有正常结束。

三模型逐题对照和 Hy 的失败案例见 [`results/mbpp-hy20/`](results/mbpp-hy20/)，原始 Qwen 双模型结果见 [`results/mbpp20/`](results/mbpp20/)。更早的 5 题试运行记录单独保存在 [`results/mbpp5/`](results/mbpp5/)，没有计入上表。

MBPP 数据集是公开的，仓库中的文件包含参考答案和全部测试；即使保留断言没有提供给 agent，也不能把它当作保密题库。课程互评时可以另行设计测试题和评价方式。

## 重新运行 MBPP 测试

先准备两个模型服务。以下命令只运行原实验的第 1 道题；去掉 `--limit 1` 可以准备全部 20 道。

```bash
.venv/bin/python -m benchmarks.mbpp.prepare --out runs/mbpp-prepared --limit 1
.venv/bin/python -m benchmarks.mbpp.run \
  --prepared runs/mbpp-prepared --out runs/mbpp-comparison \
  --4b-url http://127.0.0.1:8000/v1 --4b-model Qwen3.5-4B \
  --35b-url http://127.0.0.1:8001/v1 --35b-model Qwen3.6-35B-A3B
```

请将地址和模型名称改成实际值。两个输出目录必须是尚不存在的目录，且不能互相包含；再次运行时请使用新的目录名。脚本串行执行任务，并交替两个模型的先后顺序；结果保存在 `episodes.csv`、`episodes.jsonl` 和 `summary.json`。超时、模型错误或测试环境不可用会记录为 `unknown`。如果两个服务都需要认证，当前脚本使用同一个 `COURSE_AGENT_API_KEY`。

### 用 Hy-MT2 重新运行

如果模型服务没有启用原生工具调用，可以启动本地适配层，将模型输出的 JSON 转换为工具调用。先在一个终端运行：

```bash
.venv/bin/python -m benchmarks.mbpp.hy_adapter \
  --upstream http://127.0.0.1:8002/v1 --model Hy-MT2-7B \
  --port 8765 --trace runs/hy-adapter-trace.jsonl
```

将 `--upstream` 和模型名称改为实际服务配置。在另一个终端准备题目并运行；以下只测试 1 道，去掉 `--limit 1` 可准备 20 道：

```bash
.venv/bin/python -m benchmarks.mbpp.prepare --out runs/hy-prepared --limit 1
.venv/bin/python -m benchmarks.mbpp.run_single \
  --prepared runs/hy-prepared --out runs/hy-results \
  --base-url http://127.0.0.1:8765/v1 --model Hy-MT2-7B
```

运行后可在 `runs/hy-results/` 查看结果；结束时在第一个终端按 Ctrl+C 关闭适配层。输出目录和适配器轨迹文件必须使用新路径。轨迹可能包含题目和代码，公开前请检查。历史测试使用的原始 harness 与本仓库版本有少量配置差异，重新运行的结果可能不同。

## 安全说明

Bandit 的扫描结果需要人工判断：没有发现问题不等于代码安全，测试文件中的 `assert` 也可能触发 B101 提示。项目使用 Landlock、seccomp 和资源限制运行测试，但整个 agent 不是经过安全审计的多用户隔离系统。请使用普通用户运行，不要将其直接用于执行来源不明的代码或部署为公网服务。

## 目录

```text
agent/               agent 循环、工具、测试运行器和报告
examples/            示例任务和运行轨迹
benchmarks/mbpp/     MBPP 题目准备、模型测试和工具调用适配
data/mbpp/           MBPP sanitized 数据集
results/             既有测试结果
tests/               项目测试
```

项目代码采用 [MIT License](LICENSE)；MBPP 数据集采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，数据来源和署名见[数据说明](data/mbpp/README.md)。
