import time
import sys
import struct
sys.path.insert(0, '/home/wanghongya/dongdong/faiss-1.5.0/python')
sys.path.insert(0, '/home/wanghongya/dongdong/faiss-1.5.0/benchs')
import faiss
import numpy as np
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
from datasets import load_spacev

dataset = "deep"


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
elif dataset == "spacev":
    xb, xq, gt = load_spacev()
else:
    print("dataset not exist")
    exit()



def write_fvec(filename, data):
    with open(filename, 'wb') as fp:
        for y in data:
            d = struct.pack('I', len(y))
            fp.write(d)
            for x in y:
                a = struct.pack('f', x)
                fp.write(a)

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

# print(xb)
# print(xb.shape)
print(xq.shape)
print(gt.shape)

# write_fvec("/home/wanghongya/dataset/SPACEV//spacev10M_base.fvecs", xb)
# write_fvec("/home/wanghongya/sift1B/sift10M_query.fvecs", xq)

# print(mmap_fvecs("/home/wanghongya/dataset/SPACEV//spacev10M_base.fvecs"))
# print(mmap_fvecs("/home/wanghongya/sift1B/sift10M_query.fvecs").shape)
