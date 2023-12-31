import time
import sys
import faiss
import numpy as np
from datasets import load_sift1M, load_sift10K, load_audio

# 加载数据
print("load data")
# xb, xq, xt, gt = load_sift10K()
xb, xq, xt, gt = load_sift1M()
# xb, xq, xt, gt = load_audio()
nq, d = xq.shape
nb = xb.shape[0]
print("nq:%d d:%d" %(nq,d))

# 参数：k、m、efc、maxInDegree、splitRate、r1、r2、nb1、nb2
# para = sys.argv
para = [None, 10, 5, 100, 64, 1, 0.01, 0.03, 4, 2]
ef = [12, 16, 24, 32, 48, 64, 128, 256, 512,
      1000, 3000, 5000, 7000, 9000, 11000]

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
# python benchs/bench_hnswflat.py 1 5 40 0.01 0.03 4 2 32 1

def evaluate_hnswflat(index):
    faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    t0 = time.time()
    D, I = index.search(xq, k)
    t1 = time.time()
    computer_count = index.hnsw.get_computer_count()
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1: [%.4f], dis count:[%d]" % 
          ((t1 - t0) * 1000.0 / nq, recall_at_1, computer_count))
        
def evaluate_hnswflat_hot(index):
    faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)
    t0 = time.time()
    index.search_with_hot_hubs_enhence(xq.shape[0], faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    t1 = time.time()
    computer_count = index.hnsw.get_computer_count()
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f ms per query, R@1: [%.4f], dis count:[%d]" % 
          ((t1 - t0) * 1000.0 / nq, recall_at_1, computer_count))


# 测试
print("Testing HNSW Flat")
index = faiss.IndexHNSWFlat(d, m)

# 添加节点
print("add")
index.hnsw.efConstruction = efConstruction
index.hnsw.maxInDegree = maxInDegree
index.hnsw.splitRate = splitRate
index.verbose = True
index.hnsw.search_bounded_queue = False
index.hnsw.search_mode = 0
index.add(xb)

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

# print("evaluate_hnswflat")
# for efSearch in ef:
#     print("efSearch", efSearch, end=' ')
#     index.hnsw.efSearch = efSearch
#     evaluate_hnswflat(index)

# index.hnsw.statichotpercent(nb)
index.combine_index_with_hot_hubs_enhence(nb, 2, faiss.swig_ptr(ratios), 0, 0,
    faiss.swig_ptr(nb_nbors_per_level))
# index.hnsw.statcihotlinknums(nb)

print("evaluate_hnswflat_hot")
for efSearch in ef:
    print("efSearch", efSearch, end=' ')
    index.hnsw.efSearch = efSearch
    evaluate_hnswflat_hot(index)