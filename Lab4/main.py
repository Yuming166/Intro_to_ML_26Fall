import argparse
import numpy as np

import embedding
import eval_
import trie
import os
import mindspore as ms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100, help="k-means 迭代次数")
    parser.add_argument("--device", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("-k", default=4, type=int, help="聚类中心个数")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists("./temp"):
        os.makedirs("./temp")

    ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device)

    np.seterr(divide="ignore", invalid="ignore")

    # 聚类中心个数
    k = args.k

    # 处理Flickr60k数据并得到聚类中心和对应的特征值
    trie.run_trie(k, iterations=args.iters)

    # 处理Holidays数据并进行特征表示
    embedding.run_embedding(k)

    # 计算mAP
    eval_.mAP(k)
