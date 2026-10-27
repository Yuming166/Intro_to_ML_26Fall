"""附加题：NumPy 数据增强。完成必做题时，无须补全本文件。

只有 --augment mild 会在训练中调用增强；--stress-validation 会调用平移。
函数接收形状 [m,784]、按 pixel/255-0.5 归一化的 float32 数组。
不得修改输入数组，不改变样本顺序及标签。使用传入 rng，保证种子可复现。
"""
import numpy as np


def shift_batch(x, rng, pixels=2):
    """附加题 A1：独立实现一批图片的随机平移。

    要求：
    1. 把输入恢复为 [m,28,28] 图片。
    2. 为每张图片独立抽取上下、左右两个整数位移，均匀分布于
       [-pixels,pixels]，包括两端。先抽取整批 dy，再抽取整批 dx。
    3. 正 dy 表示向下移动，正 dx 表示向右移动。
       超出边界的部分裁去，露出的部分复制最近的边界像素填补。
       可以使用 np.pad(mode='edge') 配合循环切片或索引。
    4. pixels=0 时输出与输入逐元素相同；返回 [m,784]、float32。
    5. 每张图片可以有不同位移，不使用循环回绕 np.roll 作为最终结果。
    """
    # BEGIN OPTIONAL A1
    # 选做：完成随机位移生成、边缘填充和平移采样，返回结果。
    raise NotImplementedError("请完成 OPTIONAL A1")
    # END OPTIONAL A1


def augment_batch(x, rng, pixels=2, brightness=.08, contrast=.15):
    """附加题 A2：平移后，独立实现每张图片的亮度与对比度变化。

    要求：
    1. 平移调用和恢复到 [0,1] 像素的步骤已给出。
    2. 每张图片抽取一个对比度系数，均匀分布于
       [1-contrast,1+contrast]，围绕该图片的平均像素调整对比度。
    3. 每张图片再抽取一个亮度偏移，均匀分布于
       [-brightness,brightness]，加到整张图片上。
       先抽取整批对比度，再抽取整批亮度；同一图片内共享参数。
    4. 将像素截断到 [0,1]，再减去 0.5，恢复中心化范围。
       不用逐像素随机噪声代替整张图片的亮度变化。
    5. 默认亮度幅度为 0.08，对比度系数为 0.85 至 1.15。
       下方已给出展平和 float32 转换。
    """
    images = shift_batch(x, rng, pixels).reshape(-1, 28, 28) + np.float32(.5)
    # BEGIN OPTIONAL A2
    # 选做：抽取逐图片参数、调整对比度和亮度、截断并恢复中心化范围。
    raise NotImplementedError("请完成 OPTIONAL A2")
    # END OPTIONAL A2
    return images.reshape(-1, 784).astype(np.float32)
