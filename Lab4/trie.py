import mat73
import numpy as np
import scipy

import mindspore as ms
import mindspore.ops as ops

import tqdm


def data_group_avg(group_ids: ms.Tensor, data: ms.Tensor, k: int = 4):
    """计算每个簇的聚类中心
    Args:
        group_ids (ms.Tensor): shape=(num_pts,)
        data (ms.Tensor): shape=(num_pts, 128)
        k (int): 划分成的簇的数量
    Returns:
        avg_by_group (ms.Tensor): (k, 128)
    """
    # UnsortedSegmentSum: 沿分段计算输入Tensor元素的和。
    # 参见 https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.unsorted_segment_sum.html
    # 用法：unsorted_segment_sum(input_x, segment_ids, num_segments)
    unsorted_segment_sum = ops.UnsortedSegmentSum()
    ones_like = ops.OnesLike()

    # TODO: 计算新的聚类中心，建议使用 unsorted_segment_sum，或可自行编写代码求解

    return avg_by_group


def calc_clusters(data_points: ms.Tensor, centroids: ms.Tensor, k: int) -> ms.Tensor:
    """计算每个点与聚类中心之间的距离，并重新划分聚类
    Args:
        data_points (Tensor): shape=(num_pts, num_feats)
        centroids (Tensor): shape=(num_clusters, num_feats)
            current cluster centers
        k (int): num_clusters

    Returns:
        centroid_group (ms.Tensor): shape=(num_pts,)
    """
    reshape = ops.Reshape()
    # tile(tensor, *args) 对 tensor 的各个维度进行复制
    tile = ops.Tile()
    reduce_sum = ops.ReduceSum(keep_dims=False)
    square = ops.Square()
    argmin = ops.Argmin()

    num_pts, num_feats = data_points.shape

    centroid_matrix = ops.repeat_elements(
        reshape(centroids, (1, k, num_feats)), num_pts, axis=0
    )
    # TODO 计算每个点与聚类中心之间的距离，并重新划分簇
    # 分配时，以每个数据点最小距离为最接近的中心点
    # 提示：centroid_groups 的形状应当为 (num_pts,)

    return centroid_groups





def k_means(data_x: ms.Tensor, k: int, iterations: int):

    num_pts, num_feats = data_x.shape

    cluster_labels = np.zeros(num_pts)
    rand_starts = np.array([data_x[np.random.choice(num_pts)] for _ in range(k)])
    centroids = ms.Tensor(rand_starts.astype(np.float32))

    data_points = ms.Tensor.from_numpy(data_x.astype(np.float32))
    for i in tqdm.trange(iterations, desc="K-Means", ncols=80):
        centroid_group = calc_clusters(data_points, centroids, k)
        means = data_group_avg(centroid_group, data_points, k)
        centroids = means
        cluster_labels = centroid_group
        centroid_group_count = centroid_group.asnumpy()
        group_count = []
        for ix in range(k):
            group_count.append(np.sum(centroid_group_count == ix))

    centers, assignments = centroids, cluster_labels.asnumpy()

    return centers


def trie_learn(data: ms.Tensor, centers: ms.Tensor):
    """计算特征
    Args:
        data (ms.Tensor):
        centers (ms.Tensor): 聚类中心
    Returns:
        x_mean ()
        eigvec ()
        eigval ()
    """
    data = ms.Tensor.from_numpy(data)
    n_samples, n_features = data.shape
    k = centers.shape[0]
    x_mean = ops.zeros(k * n_features, dtype=ms.float32)
    for i in range(k):
        # TODO 求取特征均值 x_mean (k * n_features)
        # i_mean = ...
        # x_mean[...] = ...
    x_mean /= n_samples

    # 求取协方差矩阵cov_d (k * n_features, k * n_features)
    cov_d = ops.zeros((k * n_features, k * n_features), dtype=data.dtype)
    # 根据自己的内存/显存 大小调整 step
    for i in range(0, n_samples, step):
        end_i = min(i + step, n_samples)
        h_d = ops.zeros((end_i - i, n_features * k), dtype=data.dtype)
        for j in range(k):
            h_d_h = data[i:end_i] - centers[[j]]
            h_d[:, j * n_features : (j + 1) * n_features] = h_d_h / ops.norm(
                h_d_h, dim=1, keepdim=True
            )
        h_d -= x_mean
        cov_d += ops.dot(h_d.T, h_d)
    # 这里依据数学公式应是除以n_samples，但实际上除以多少都不影响结果，因为不会改变相对大小
    cov_d /= n_samples
    """计算特征值和特征向量"""
    eigval, eigvec = np.linalg.eig(cov_d.asnumpy())
    idx = eigval.argsort()[::-1]  # descending sort
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    return x_mean, eigvec, eigval


def run_trie(kc, iterations):
    """程序起点"""
    """加载Flickr60k数据"""
    # v_train = mat73.loadmat("./data/vtrain.mat")["vtrain"].T  # 5000404*128
    # 加载经过处理后的 vtrain 数据, 2**19 x 128
    v_train = scipy.io.loadmat(f"./data/vtrain_{2**19}.mat")["vtrain"].T
    v_train = v_train.astype(np.float32)
    # v_train = ms.Tensor.from_numpy(v_train)
    print(f"load Flickr60k train data: {v_train.shape[0]} items.")

    """使用K-means聚类算法，得到对应的k个聚类中心，表现形式为k*128维向量"""
    c = k_means(v_train, kc, iterations)
    # 保存计算出的聚类中心
    np.savetxt("./temp/center_{}.csv".format(kc), c.asnumpy())
    print("K-means finish")
    # 节省时间时可以注释掉k_means计算，直接加载计算好的中心
    # c = np.loadtxt('./temp/center_{}.csv'.format(kc))

    """计算三角化嵌入所需参数"""
    x_mean, eigvec, eigval = trie_learn(v_train, c)
    print("trie finish")

    """计算投影矩阵"""
    eigval[-128:] = eigval[-129]
    p_emb = np.diag(np.power(eigval, -0.5)) @ eigvec.T

    """保存"""
    np.savetxt("./temp/x_mean_{}.csv".format(kc), x_mean.asnumpy())
    np.savetxt("./temp/eigval_{}.csv".format(kc), eigval)
    np.savetxt("./temp/eigvec_{}.csv".format(kc), eigvec)
    np.savetxt("./temp/p_emb_{}.csv".format(kc), p_emb)
