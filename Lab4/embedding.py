import mat73
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import tqdm


def to_tensor(a: np.ndarray, dtype=None) -> ms.Tensor:
    tensor = ms.Tensor.from_numpy(a.astype(np.float32))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def desc_postprocess(x, desc_mean):
    """
    数据预处理
    """
    # 取平方根，中心化
    # x = np.sqrt(x) - desc_mean
    x = ops.sqrt(x) - desc_mean
    # L2正则化
    # x /= np.linalg.norm(x, axis=1, keepdims=True)
    x /= ops.norm(x, dim=1, keepdim=True)
    return x


def trie_suma(data, centers, x_mean):
    """三角化嵌入"""
    kc = centers.shape[0]
    n, f = data.shape
    h_y = ops.zeros((n, kc * f), dtype=ms.float32)

    # TODO 补全三角化嵌入
    for i in range(kc):
        # h = ...
        # h_y[...] = ...
    h_y -= x_mean
    return h_y.sum(axis=0)


def run_embedding(kc):
    """程序起点"""
    """加载Holidays数据集和其余数据"""
    print(f"Loading `holidays`...")
    x_data = mat73.loadmat("./data/X.mat")["X"].T
    cndes = mat73.loadmat("./data/cndes.mat")["cndes"].astype("int")
    desc_mean = mat73.loadmat("./data/desc_mean.mat")["desc_mean"]
    # 上一步训练好的数据
    centers = np.loadtxt("./temp/center_{}.csv".format(kc))
    x_mean = np.loadtxt("./temp/x_mean_{}.csv".format(kc))
    p_emb = np.loadtxt("./temp/p_emb_{}.csv".format(kc))

    centers = to_tensor(centers, ms.float32)
    x_mean = to_tensor(x_mean, ms.float32)
    x_data = to_tensor(x_data, ms.float32)
    desc_mean = to_tensor(desc_mean, ms.float32)
    p_emb = to_tensor(p_emb.astype(np.float32))
    print("Load done.")

    """开始计算"""
    n = cndes.shape[0] - 1
    n_samples, n_features = x_data.shape
    psi = ops.zeros(size=(n, kc * n_features), dtype=ms.float32)
    # 逐图片计算
    print("Train starts...")
    for i in tqdm.trange(n, desc="Train", ncols=80):
        # 得到第i张图片的sift特征
        l, r = cndes[i].tolist(), cndes[i + 1].tolist()
        x = x_data[l : max(r, l + 1)]
        # 进行预处理
        x = desc_postprocess(x, desc_mean)
        # 三角化嵌入
        psi[i] = trie_suma(x, centers, x_mean)
    psi = psi @ p_emb.T

    """保存"""
    np.savetxt("./temp/psi_{}.csv".format(kc), psi.asnumpy())
