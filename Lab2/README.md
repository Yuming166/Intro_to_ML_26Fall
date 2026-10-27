# Task2：使用线性支持向量机区分两种水稻

先完成 svm.py 的六处 TODO，调通无扩展基线，再进行固定对照并解释结果。A1、A2 为选做。

使用 Python 3 和 CPU，无需深度学习框架。进入项目目录，在终端执行：

```bash
python3 -m pip install -r requirements.txt
```

安装与运行使用同一个解释器。Windows 可将 python3 换为本机对应的 python。

## 必做与运行

| 编号 | 需要实现 |
|---|---|
| TODO 1 | 批量线性分数 X@w+b |
| TODO 2 | hinge 损失和 margin 子梯度 |
| TODO 3 | 平均数据损失加 L2 的总目标 |
| TODO 4 | 权重及偏置的批量（子）梯度 |
| TODO 5 | lr/sqrt(epoch) 学习率 |
| TODO 6 | 一次 SGD 参数更新 |

只替换对应 NotImplementedError；保留函数接口、返回形状和训练框架。直接运行未填模板会明确提示未完成题目，这是预期行为。

默认：7 维输入，hinge，标准化，λ=0.01，lr=0.05，150 轮，batch=64，seed=42。采用小批量随机子梯度下降，不含投影步骤，不称为原始 Pegasos。

```bash
# 开发时只看训练、验证，不评价测试集
python3 main.py --validation-only
# 标准化对照；其他设置相同
python3 main.py --scaling none --validation-only
# 正则化对照：依次使用 0.0001、0.001、0.01、0.1
python3 main.py --regularization 0.1 --validation-only
# 同一目标函数的成熟库线性 SVM 参考
python3 main.py --model library-linear --validation-only
# 教师提供的二维线性分类演示
python3 geometry_demo.py
```

每次运行自动建立独立 output 子目录，不覆盖已有实验。数据和输出默认相对脚本位置解析，也可指定 --output。

超参数和预先声明的对照方案确定后，去掉 --validation-only，进行该配置的最终测试；geometry_demo.py 使用 --final-test 开启其测试评价。不得根据测试结果选择学习率、正则化或保存轮次。

## 选做

A1：logistic_terms 的稳定损失与导数。完成后：

```bash
python3 main.py --loss logistic --validation-only
```

A2：向量化 RBF 核，K[i,j]=exp(-gamma*||x[i]-z[j]||²)。训练器和绘图已提供，完成后：

```bash
python3 geometry_demo.py --rbf
python3 main.py --model library-rbf --C 1 --gamma 0.14285714285714285 --validation-only
```

在二维演示中，两种核使用相同 C。水稻库 RBF 的固定设置为 C=1、gamma=1/7，它与默认线性参考的 C 不同，属于扩展示例。改变 C、gamma 时只用验证集选择，不要求 RBF 必须提升。

## 数据、评价和提交

官方数据共 3,810 行，7 个形态特征，Cammeo=−1、Osmancik=+1。固定分层划分为训练 2,286、验证 762、测试 762；split-seed=42。均值和标准差只从训练行拟合。没有缺失值或完全重复的特征行。

目标：λ/2×权重平方和＋批次平均数据损失；偏置不正则化。梯度只平均一次，使用最后一批的真实大小。这里的 λ 越大正则化越强；对应 0.5||w||²+C∑hinge 的 C=1/(n_train×λ)。

每轮由验证集平均数据损失选取模型；不把正则项加入验证选轮损失。metrics.json 的 selected_checkpoint_metrics 是选中轮次的指标；history.csv 是逐轮记录。准确率在 JSON 中为 0–1，报告中可乘 100 写为百分比。测试指标包含准确率、Balanced Accuracy、Macro-F1、各类 Precision/Recall/F1 与混淆矩阵。

报告解释：已分对的样本为什么仍可能更新？margin≥1 时权重为何也可能变化？标准化有什么影响？λ 增大是否必然改善测试表现？如实保留无改善、未完成或失败的结果。不要直接比较 hinge 和 logistic 目标值的大小来判断优劣。

提交 Task2-学号-姓名-班级 文件夹：代码、Word 报告及代表性运行结果。保留 config.json、data_metadata.json、split_indices.npz、metrics.json、history.csv 和相关图。无需重复提交 data 或缓存。

数据来源：UCI Rice (Cammeo and Osmancik), 2019, https://doi.org/10.24432/C5MW4Z 。许可证 CC BY 4.0。本包将官方 ARFF 转为 CSV，保留原始记录与特征值。详细任务见《Task2-实验任务说明-水稻分类.docx》。
