import time
import sys
import numpy as np
sys.path.insert(0,'/home/wanghongya/dongdong/faiss-1.5.0/python')
sys.path.insert(0,'/home/wanghongya/dongdong/faiss-1.5.0/benchs')
import faiss
# import util
import os
import random
from faiss import normalize_L2


from datasets import load_sift1M
from datasets import load_sift10K
from datasets import load_audio
from datasets import load_glove
from datasets import load_glove2m
from datasets import load_imageNet
from datasets import load_random_gaussian
from datasets import load_word2vec
from datasets import load_tiny80m
from datasets import load_deep
from datasets import load_bigann
from datasets import load_random_hypercube
from datasets import load_random_hypersphere
from datasets import load_random_hypercube200
from datasets import load_random_hypersphere200
from datasets import load_random_hypersphere50


print(sys.path)
INF = int(1e9 + 7)
m = int(sys.argv[1])
efcon = int(sys.argv[2])
k = int(sys.argv[3])
dataset = str(sys.argv[4])
r1 = 0.01
r2 = 0.03

nb1 = 4
nb2 = 2
maxInDegree=INF
splitRate=1.0
# dataName=sys.argv[8]
todo = sys.argv[2:]
# print('K',k,"m",m,'efcon',efcon,"r1",r1,"r2",r2)


# 加载数据
print(f"load data: dataset[{dataset}],m[{m}],efc[{efcon}],k[{k}],maxIn[{maxInDegree}]")
if dataset == "sift10K":
    xb, xq, xt, gt = load_sift10K()
elif dataset == "audio":
    xb, xq, xt, gt = load_audio()
elif dataset == "sift1M":
    xb, xq, xt, gt = load_sift1M()
elif dataset == "glove1M":
    xb, xq, xt, gt = load_glove()
elif dataset == "glove2M":
    xb, xq, xt, gt = load_glove2m()
elif dataset == "random_gaussian":
    xb, xq, xt, gt = load_random_gaussian()
elif dataset == "imageNet":
    xb, xq, xt, gt = load_imageNet()
elif dataset == "word2vec":
    xb, xq, xt, gt = load_word2vec()
elif dataset == "tiny80m":
    xb, xq, xt, gt = load_tiny80m()
elif dataset == "deep":
    xb, xq, xt, gt = load_deep()
elif dataset == "bigann":
    xb, xq, xt, gt = load_bigann()
elif dataset == "random_hypercube":
    xb, xq, xt, gt = load_random_hypercube()
elif dataset == "random_hypersphere":
    xb, xq, xt, gt = load_random_hypersphere()
elif dataset == "random_hypercube200":
    xb, xq, gt = load_random_hypercube200()
elif dataset == "random_hypersphere200":
    xb, xq, gt = load_random_hypersphere200()
elif dataset == "random_hypersphere50":
    xb, xq, xt, gt = load_random_hypersphere50()
else:
    print("dataset not exist")
    exit()


# normalize_L2(xb)
# normalize_L2(xq)



nq, d = xq.shape
print("nq:%d d:%d" %(nq,d))
n=xb.shape[0]

# np.savetxt('../glove.tsv',xb,delimiter='\t')

print("Testing HNSW Flat")
print("xq:")
print(xq.shape[0])

_type1="no"
_type2="shink"
_type3="shink_hot"
_type4="shink_hot_limit"
# 创建标准索引

index = faiss.IndexHNSWFlat(d, m)

index.hnsw.efConstruction = efcon

index.hnsw.maxInDegree=maxInDegree
index.hnsw.splitRate=splitRate
# index.hnsw.upper_beam=50
index.verbose = True
index.hnsw.search_bounded_queue = False
index.hnsw.search_mode = 0
# index.hnsw.init_in_vector(n)
index.add(xb)
# index.shrink_level_0_neighbors(32)
#KNN构图
# index.hnsw.static_in_degree_by_construction(n)



# 热点参数
ratios = np.empty((1, 2), dtype=np.float32)
nb_nbors_per_level = np.empty((1, 2), dtype=np.int32)
# ratios.fill(0.0001)
# nb_nbors_per_level.fill(0)
ratios[0][0]=r1
ratios[0][1]=r2

nb_nbors_per_level[0][0]=nb1
nb_nbors_per_level[0][1]=nb2

# #统计入度
index.hnsw.find_inNode_Hnsw(n,2,faiss.swig_ptr(ratios),str(m)+'m_'+str(maxInDegree)+'indegree_'+dataset)


#获取热点
# print("输出knn热点")
index.hnsw.getHotSet(n,2,faiss.swig_ptr(ratios))
# index.hnsw.print_time(n)
# index.hnsw.print_nb_in_degree(n,m,dataName,_type2)


# index.static_in_degree_by_direct(n)

# index.createKnn(n,11)
# #求二者交集占比
# index.hnsw.staticKNNHot(n,2,faiss.swig_ptr(ratios))

# index.hnsw.search_bounded_queue = False
# index.hnsw.efSearch = 50

#原版搜索
# def evaluate2(index):
#     # for timing with a single core
    
#     faiss.omp_set_num_threads(1)
#     index.hnsw.resetCount()

#     t0 = time.time()
#     D, I = index.search(xq, k)
#     t1 = time.time()
#     #统计距离
#     #index.hnsw.getSearchDisFlat()
#     #统计距离
#     missing_rate = (I == -1).sum() / float(k * nq)
#     recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
#     print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
#         (t1 - t0) * 1000.0 / nq, recall_at_1, missing_rate))
#     return recall_at_1
# for efSearch in 50,100 ,200,300,400,500,600,700,800,900,1000,2000,3000,4000:
#     index.hnsw.efSearch = efSearch
#     if(evaluate2(index) == 1):
#         break

def evaluate_flat(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)
    index.hnsw.resetCount()
    t0 = time.time()
    
    D, I = index.search(xq, k)
    # print(I)
    t1 = time.time()
    # for i in range(len(D)):
    #     print(D[i,0],I[i,0],gt[i,0])
    computer_count=index.hnsw.get_computer_count()
    #index.hnsw.getSearchDis()
   
    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    print("\t %7.3f\t%.4f\t%d\t%d" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1,computer_count,computer_count/nq))
    return recall_at_1
# for efSearch in [500]:
# for efSearch in 10,20,30,40,50,60,70,80,100,110,120,130,140,150,160,200,300,400,500,600,1000,2000,3000,5000,6000,12000,20000,30000,33000,35000,38000,40000:
for efSearch in 10,20,30,40,50,70,80,100,120,130,140,150 ,200,300,400,500,600,1000,2000,3000,5000,8000,9000,11000,20000:
    index.hnsw.efSearch = efSearch
    print(efSearch, end=' ')
    # evaluate(index)
    if(evaluate_flat(index) == 1):
        break
#从边的角度统计热点占比：
# index.hnsw.statichotpercent(n)
# 增强索引
# 2，1 表示使用聚类方法选择/随机分类，添加热点
# 0，0 表示使用全局方法选择，添加热点

hot_t0 = time.time()

index.combine_index_with_hot_hubs_enhence(n,2,faiss.swig_ptr(ratios),0,0,
    faiss.swig_ptr(nb_nbors_per_level))
print("第零层邻居")
print(index.hnsw.cum_nb_neighbors(0))
print("第1层")
print(index.hnsw.cum_nb_neighbors(1))
hot_t1 = time.time()
print((hot_t1 - hot_t0) * 1000.0)
index.hnsw.find_inNode_Hot(n,2,faiss.swig_ptr(ratios),str(m)+'m_'+str(maxInDegree)+'indegree_'+dataset)
# index.static_in_degree_by_direct(n)



# index.hnsw.print_nb_in_degree_hot(n,m,dataName,_type3)

index.hnsw.statcihotlinknums(n)

# 数据收集
ef= []
r_all = []
t = []  


def evaluate_one(index, j, efs, flag):
    # faiss.omp_set_num_threads(1)
    D = np.empty((1, k), dtype=np.float32)
    I = np.empty((1, k), dtype=np.int64)
    t0 = time.time()
    if(flag):
        xq1=np.array([xq[j]])
    else:
        xq1=np.array([xb[j]])
    #统计距离
    index.search_with_hot_hubs_enhence(xq1.shape[0], faiss.swig_ptr(xq1), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    t1 = time.time()
    missing_rate = (I == -1).sum() / float(k)
    if(flag):
        recall_at_1 = (I == gt[j, :1]).sum()
    else:
        recall_at_1 = I[0]==j
        print(I[0])
        print(j)

    # print("\t %7.3f ms per query, R@1 %.4f, missing rate %.4f" % (
    #     (t1 - t0) * 1000.0, recall_at_1, missing_rate))
    
    if(recall_at_1==1): 
        t.append(round((t1-t0)*1000.0,4))
        ef.append(efs)
        print("\t %.3f ms per query, R@1 %.4f, missing rate %d" % (
         (t1 - t0) * 1000.0, recall_at_1, efs))
   
    if(recall_at_1!=1 and (t1 - t0) * 1000.0>30.0):
        # t.append(round((t1-t0)*1000.0,4) )
        # ef.append(efs)
        print("未找到 真实：%d，查找%d"%(gt[j,0],I[0]))
        print(I[0])
        print(j)
        return -1
    return int(recall_at_1)

# # r=evaluate_one(index,0,10)
# find_fail=0
# for i in range(xq.shape[0]):
#     efs=1
#     index.hnsw.efSearch = efs
#     r=evaluate_one(index,i,efs,True)
#     while r!=1 and r!=-1:
#         efs=efs+2
#         index.hnsw.efSearch = efs
#         r=evaluate_one(index,i,efs,True)
#     if(r==1):
#         r_all.append(r)
#     else:
#         find_fail+=1
# index.hnsw.efSearch=1000
# r=evaluate_one(index,53386,1000,False)
# print(t)
# print(ef)
# print(r_all)
# print("查询大于15ms的数量")
# print(find_fail)
# nd=np.array([t,ef,r_all])
# nd.T

# np.savetxt(r'/home/wanghongya/lc/efsearch_time/glove/hot.csv', nd.T, fmt="%f,%d,%d",delimiter=",")


def evaluate(index):
    # for timing with a single core
    # faiss.omp_set_num_threads(1)
    D = np.empty((xq.shape[0], k), dtype=np.float32)
    I = np.empty((xq.shape[0], k), dtype=np.int64)
    #统计距离
    # index.hnsw.resetCount()
    # index.hnsw.initHothubSearchmp()
    index.hnsw.resetCount()
    t0 = time.time()
    
    #统计距离
    index.search_with_hot_hubs_enhence(xq.shape[0], faiss.swig_ptr(xq), k, faiss.swig_ptr(D), faiss.swig_ptr(I),xb.shape[0])
    t1 = time.time()
    computer_count=index.hnsw.get_computer_count()
    #统计距离
    # index.hnsw.getSearchDis()
    # index.hnsw.getHothubsSearchCount()
    # 统计距离
    missing_rate = (I == -1).sum() / float(k * nq)
    recall_at_1 = (I == gt[:, :1]).sum() / float(nq)
    
    ef.append(index.hnsw.efSearch)
    # r.append(float(format(recall_at_1, '.4f')))
    # t.append(float(format((t1-t0)*1000.0/nq, '.4f')))
    print("\t %7.3f\t%.4f\t%d\t%d" % (
        (t1 - t0) * 1000.0 / nq, recall_at_1,computer_count,computer_count/nq))

    return recall_at_1


# for i in 

# 搜索
# index.hnsw.efSearch = 40
# evaluate(index) 
# for efSearch in [5000]:

for efSearch in 10,20,30,40,50,100 ,200,300,400,500,600,700,800,900,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,11000,20000:
# for efSearch in 10,20,30,40,50,100,110,120,130,140,150,160,200,300,400,500,600,1000,2000,3000,5000,6000,12000,20000,30000,33000,35000,40000:
    index.hnsw.efSearch = efSearch
    print(efSearch, end=' ')
    # evaluate(index)
    if(evaluate(index) == 1):
        break
# for efSearch in 100,200,2000:
#     index.hnsw.efSearch = efSearch
#     print(efSearch, end=' ')
#     # evaluate(index)
#     if(evaluate(index) == 1):
#         break
# print(ef)
# print(r)
# print(t)
    