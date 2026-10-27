# 第一次实验：Sign Language MNIST 全连接网络实验

本实验全部计算使用 CPU，无需 GPU 或深度学习框架。macOS / Linux 通常使用 python3，Windows 可将示例命令中的 python3 改为所安装解释器的 python。依赖安装与运行应使用同一个解释器。

建议提交四个代码文件、实验报告和所选实验的输出目录；不重复提交数据集、arrays_cache.npz 或 __pycache__。报告中注明运行命令、训练轮数、网络结构、学习率、激活函数、L2 系数，以及是否开启增强。测试集只用于最终评价，调试和配置选择使用训练／验证数据。

数据来源：https://www.kaggle.com/datasets/datamunge/sign-language-mnist


核心代码只有四个文件：`main.py`、`network.py`、`sign_mnist_loader.py`、`augmentation.py`。推荐目录如下：

```text
Task1-静态手势分类/
├── main.py
├── network.py
├── sign_mnist_loader.py
├── augmentation.py
├── README.md
├── requirements.txt
└── data/
    ├── sign_mnist_train.csv
    └── sign_mnist_test.csv
```

加载器从官方 train 的 27,455 张图片中按类别分层划出 24,709 张训练、2,746 张验证，官方 test 的 7,172 张保持独立。保持 `--split-seed 42 --validation-fraction 0.1` 可复现固定划分。

先安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

## 必做：network.py 中的七处 TODO

保留原有函数接口、循环、缓存及返回结构，将相应 `raise NotImplementedError(...)` 替换为实现。

| 编号 | 需要实现的内容 |
|---|---|
| TODO 1 | 数值稳定的多分类平均交叉熵 |
| TODO 2 | 各层线性变换、隐藏层激活；训练和预测共用 |
| TODO 3 | 输出层误差，使用当前批次真实大小进行平均 |
| TODO 4 | 权重梯度，包含 L2 正则项 |
| TODO 5 | 偏置梯度，沿样本维求和，偏置不正则化 |
| TODO 6 | 隐藏层误差传播及 ReLU / Sigmoid 的导数 |
| TODO 7 | 用已计算的梯度进行 SGD 参数更新 |

形状约定：输入为 `[m,d]`，权重为 `[d,h]`，偏置为 `[h]`，整数标签为 `[m]`。梯度只平均一次；本模板在 TODO 3 处理平均，因此后续步骤不要再次除以批次大小。最后一个批次可能不足设置的 batch-size。

基础运行：

```bash
python3 main.py
```

此时默认关闭训练增强和平移验证，不会调用尚未完成的附加题。默认网络为 784→64→24、ReLU、80 轮、学习率 0.01、批次大小 128、L2=0。每次运行默认在脚本旁 output/ 下创建独立目录并打印路径，便于反复调试。若显式指定 --output，则已有实验目录仍不会被覆盖，请为新实验指定不同目录。

## 附加题：augmentation.py 中的两处大空

**A1：随机平移（shift_batch）。** 每张图片分别随机生成上下、左右整数位移，范围均为 −2 至 2 像素，复制最近边界像素填补露出部分；返回形状及样本顺序不变。代码中给出了正方向、随机数顺序及输入输出要求。

**A2：随机亮度与对比度（augment_batch）。** 使用 A1 平移后的图片，每张图片共享一个随机对比度系数和一个随机亮度偏移，围绕该图片平均像素改变对比度，再加亮度偏移并截断像素。默认亮度偏移为 −0.08 至 0.08，对比度系数为 0.85 至 1.15。A2 依赖 A1；输入归一化方式为 `pixel/255-0.5`。

完成 A1、A2 后：

```bash
python3 main.py --augment mild
```

比较增强收益时，应保持训练轮数、网络结构、学习率、划分种子等其他设置一致。报告训练／验证／测试准确率及曲线，并讨论增强为何可能有帮助、是否存在没有改善的类别。增强在训练批次上动态生成，不修改原始 CSV，不修改标签，测试图片保持原样。

仅完成 A1 时，可以额外检查固定平移后的验证表现：

```bash
python3 main.py --stress-validation
```

只有显式设置 `--stress-validation` 才计算这项指标。它是同一验证集的固定扰动视图，不是新增独立验证集。若希望按正常／平移验证交叉熵均值选择保存轮次，须同时设置 `--selection stress-mean`；常规实验默认按正常验证集交叉熵选择。

## 数据与运行说明

- 训练 24,709 张，验证 2,746 张，官方测试 7,172 张；划分种子默认固定为 42。
- 这是图片级分层划分，不是按手语者分组；缺少 J、Z，原始标签由加载器映射为连续 24 类。
- `main.py` 和 `sign_mnist_loader.py` 已完整提供，无须挖空补写。
- 默认自动识别 data 子目录；显式指定时可用 `python3 main.py --data-dir data`，绝对路径也可使用。
- 开发时可加 `--validation-only`，只输出训练／验证评价；完成后再做最终测试。
- 代码包不含数据，完整版学生包附带 data/ 下的两个原始 CSV；两种包中的代码完全相同。
