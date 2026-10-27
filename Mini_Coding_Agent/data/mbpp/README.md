# MBPP sanitized 数据集及署名

`sanitized-mbpp.json` 是 Google Research 发布的 MBPP sanitized 数据文件，原样包含 427 条记录、题目、参考实现和测试。它的大小为 255,053 bytes，SHA-256：

```text
ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9
```

- 作者：Jacob Austin 等，Google Research。
- 数据来源：[google-research/google-research 的 MBPP](https://github.com/google-research/google-research/tree/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp)，固定版本 `f82046ba5aabbbb427dbfd38a254d26bff08b533`；[原始数据文件](https://raw.githubusercontent.com/google-research/google-research/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json)。
- 许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，见 [Google Research 发布的数据集卡](https://huggingface.co/datasets/google-research-datasets/mbpp)。再分发或改编应保留适当署名、许可链接并说明修改。
- 论文：Austin et al., *Program Synthesis with Large Language Models*, 2021. [arXiv:2108.07732](https://arxiv.org/abs/2108.07732)。
- 修改说明：此 JSON 文件未修改；评测脚本从中选择题目，将前两条原始断言转为公开 unittest，其余断言转为结束后执行的 unittest，并创建函数存根。

这是公开数据，包含答案和全部测试。它不能充当隐藏题库或独立的保密测试集。
