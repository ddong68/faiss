import time
import sys
import numpy as np
sys.path.insert(0,'/home/wanghongya/dongdong/faiss150/python')
sys.path.insert(0,'/home/wanghongya/dongdong/faiss150/benchs')
import faiss
# import util
import os
import random

from faiss import normalize_L2

from datasets import load_random
from datasets import load_glove
from datasets import load_audio
from datasets import load_sift1M
from datasets import load_deep
from datasets import load_gist
from datasets import load_imageNet
from datasets import load_sift10K
from datasets import load_bigann
from datasets import load_sun
from datasets import load_msong
from datasets import load_movielens
from datasets import load_yahoomusic
from datasets import load_tiny5m
from datasets import load_cifar60k
from datasets import load_word2vec
from datasets import load_netflix
from datasets import load_glove2m
from datasets import load_msong
from datasets import load_sun
from datasets import load_movielens
from datasets import load_random_gaussian
from datasets import load_random_gaussian_1
from datasets import load_nuswide
from datasets import load_nuswide_230
from datasets import load_Tiny80m
from datasets import load_trevi
from datasets import load_random_gaussian_center
from datasets import load_text2image1B
from datasets import load_spacev
from datasets import load_enron
import queue
import struct

# xb, xq, gt = load_text2image1B()
# xb, xq, xt, gt = load_glove()
# xb, xq, xt, gt = load_glove()
# xb, xq, xt, gt = load_sift1M()
# xb, xq, xt, gt = load_gist() 
# xb ,xq=load_spacev()
xb ,xq, xt, gt=load_enron()
print(xb)
normalize_L2(xb)
normalize_L2(xq)
print(xb)
print("结束")
sum=0

nq, d = xq.shape
index=faiss.IndexFlat(d,faiss.METRIC_INNER_PRODUCT)
# index=faiss.IndexFlatL2(d)
index.add(xb)
print("开始")
t0 = time.time()
D, I = index.search(xq, 100)
print(I)
# print(D)
print(D[0])
t1 = time.time()
print("结束",t1-t0)
print(len(I))

def to_ivecs(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('I', x)
                fp.write(a)
                
to_ivecs('/home/wanghongya/dataset/enron/enron_cos_groundtruth.ivecs',I)
