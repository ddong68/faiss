import time
import sys
sys.path.insert(0, '/home/wanghongya/hanhan/faiss-1.5.0/python')
sys.path.insert(0, '/home/wanghongya/hanhan/faiss-1.5.0/benchs')
import faiss
import numpy as np
from datasets import load_sift10K
from datasets import load_sift1M
from datasets import load_bigann
from datasets import index_add_bigann
from datasets import load_deep
from datasets import index_add_deep
from datasets import load_audio
from datasets import load_glove

# 加载数据
print("load data")
# xb, xq, xt, gt = load_sift10K()
xb, xq, xt, gt = load_sift1M()
# nb, xq, xt, gt = load_bigann()
# nb, xq, xt, gt = load_deep()
# xb, xq, xt, gt = load_audio()
# xb, xq, xt, gt = load_glove()
nq, d = xq.shape
nb = xb.shape[0]
print("nq: [%d], d: [%d]" % (nq, d))

# sys.exit()

# 参数：k、m、efc、maxInDegree、splitRate、r1、r2、nb1、nb2
para = sys.argv
INF = int(1e9 + 7)
# para = [None, 500, 5, 50, INF, 1, 0.01, 0.03, 4, 2, 16]
re = [1, 10, 100, 500] # recall的输出值
ef = [#4, 8, 12, 16, 32, 64, 128, 256, 
      500, 800, 1000, 2000, 3000, 4000, 
      5000, 7000, 9000, 11000, 13000, 15000, 
      17000, 20000] # efsearch的输出值

# 加载输入参数
k = int(para[1])
m = int(para[2])
efConstruction = int(para[3])
maxInDegree = int(para[4])
splitRate = float(para[5])
r1 = float(para[6])
r2 = float(para[7])
nb1 = int(para[8])
nb2 = int(para[9])
pq_m = int(para[10])
# python benchs/bench_hnswflat.py 1 5 40 0.01 0.03 4 2 32 1

def evaluate_hnswpq(index):
    faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    recalls = {i: (I[:, :i] == gt[:, :1]).sum() / float(nq) for i in re}
    computer_count = index.hnsw.get_computer_count()
    missing_rate = (I == -1).sum() / float(k * nq)
    # print(f"\t {((t1 - t0) * 1000.0 / nq):.4f} ms per query, "
    #       f"R@: {recalls}, dis per count:[{computer_count // nq}], "
    #       f"missing rate:[{missing_rate:.4f}]")
    print(f"{((t1 - t0) * 1000.0 / nq):.4f};{recalls[1]};"
          f"{recalls[10]};{recalls[100]};{recalls[500]};"
          f"{computer_count // nq};{missing_rate:.4f}")
        
def evaluate_hnswpq_hot(index):
    faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)
    t0 = time.time()
    index.search_with_hot_hubs_enhence(xq.shape[0], 
            faiss.swig_ptr(xq), k, faiss.swig_ptr(D), 
            faiss.swig_ptr(I),nb)
    t1 = time.time()
    recalls = {i: (I[:, :i] == gt[:, :1]).sum() / float(nq) for i in re}
    computer_count = index.hnsw.get_computer_count()
    missing_rate = (I == -1).sum() / float(k * nq)
    # print(f"\t {((t1 - t0) * 1000.0 / nq):.4f} ms per query, "
    #       f"R@: {recalls}, dis per count:[{computer_count // nq}], "
    #       f"missing rate:[{missing_rate:.4f}]")
    print(f"{((t1 - t0) * 1000.0 / nq):.4f};{recalls[1]};"
          f"{recalls[10]};{recalls[100]};{recalls[500]};"
          f"{computer_count // nq};{missing_rate:.4f}")

def search_without_hot(index):
    print("evaluate_hnswpq")
    print("efsearch;per query;recall@1;recall@10;recall@100;recall@500;clac count;missing rate")
    for efSearch in ef:
        # print("efSearch", efSearch, end=' ')
        print(efSearch, end=';')
        index.hnsw.efSearch = efSearch
        evaluate_hnswpq(index)

def search_hot(index, ratios, nb_nbors_per_level):
    # index.hnsw.statichotpercent(nb)
    index.combine_index_with_hot_hubs_enhence(nb, 2, faiss.swig_ptr(ratios), 0, 0,
        faiss.swig_ptr(nb_nbors_per_level))
    # index.hnsw.statcihotlinknums(nb)

    print("evaluate_hnswpq_hot")
    print("efsearch;per query;recall@1;recall@10;recall@100;recall@500;clac count;missing rate")
    for efSearch in ef:
        # print("efSearch", efSearch, end=' ')
        print(efSearch, end=';')
        index.hnsw.efSearch = efSearch
        evaluate_hnswpq_hot(index)
        
def main():
    # 构造
    print("Testing HNSWPQ")
    index = faiss.IndexHNSWPQ(d, pq_m, m)#三个参数分别代表（向量维度，pq的分块数，图的边数）
    index.verbose = False
    index.storage.verbose = False
    index.hnsw.efConstruction = efConstruction
    index.hnsw.maxInDegree = maxInDegree
    index.hnsw.splitRate = splitRate
    index.hnsw.search_bounded_queue = False
    index.hnsw.search_mode = 0
    print("efConstruction: [%d], maxInDegree: [%d]" % (efConstruction, maxInDegree))

    # 训练质心
    print("train, size", xt.shape)
    t0 = time.time()
    index.train(xt)
    print("  train in %.3f s" % (time.time() - t0))

    # 添加节点
    print("add, size", nb)
    t1 = time.time()
    index.add(xb)
    # index_add_bigann(index, nb, step=int(1e7))
    # index_add_deep(index, nb, step=int(1e6))
    print("  add in total %.2f min" % ((time.time() - t1) / 60))

    # 计算、获取热点
    ratios = np.empty((1, 2), dtype=np.float32)
    nb_nbors_per_level = np.empty((1, 2), dtype=np.int32)
    # ratios.fill(0.0001)
    # nb_nbors_per_level.fill(0)
    ratios[0][0] = r1
    ratios[0][1] = r2
    nb_nbors_per_level[0][0] = nb1
    nb_nbors_per_level[0][1] = nb2
    index.hnsw.getHotSet(nb, 2, faiss.swig_ptr(ratios))

    search_without_hot(index)
    search_hot(index, ratios, nb_nbors_per_level)
    
if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"(efc={efConstruction}, maxInDegree={maxInDegree}) "
            f"use time {((time.time() - t0) / 60):.2f} min\n")