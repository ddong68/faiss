# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
'''
from __future__ import print_function
import sys
import time
import numpy as np


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    xt = fvecs_read("sift1M/sift_learn.fvecs")
    xb = fvecs_read("sift1M/sift_base.fvecs")
    xq = fvecs_read("sift1M/sift_query.fvecs")
    gt = ivecs_read("sift1M/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt


def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls
'''
from __future__ import print_function
import sys
import time
import numpy as np

# 文件读取
def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def read_i8bin(filename, start_idx=0, chunk_size=None):
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int8, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def ivecs_mmap(fname):
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]

def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')

def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]
    
def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

def mmap_fvecs(fname):
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def sanitize(x):
    """ convert array to a c-contiguous float array """
    return np.ascontiguousarray(x.astype('float32'))

# 数据加载
simdir = '/home/wanghongya/dataset/'

def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    basedir = simdir + 'sift1M/'
    xt = fvecs_read(basedir + "sift_learn.fvecs")
    xb = fvecs_read(basedir + "sift_base.fvecs")
    xq = fvecs_read(basedir + "sift_query.fvecs")
    gt = ivecs_read(basedir + "sift_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt

def load_sift10K():
    print("Loading sift10K...", end='', file=sys.stderr)
    basedir = simdir + 'sift10k/'
    xt = fvecs_read(basedir + "siftsmall_learn.fvecs")
    xb = fvecs_read(basedir + "siftsmall_base.fvecs")
    xq = fvecs_read(basedir + "siftsmall_query.fvecs")
    gt = ivecs_read(basedir + "siftsmall_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt

def load_bigann():
    print("load sift.....")
    # basedir = simdir + 'sift1B/'
    basedir = '/home/wanghongya/sift1B/'

    dbsize = 10
    # nb = dbsize * 1000 * 1000
    xb_map = bvecs_mmap(basedir + '1milliard.p1.siftbin')
    # xb = fvecs_read(basedir + 'sift10M_base.fvecs')
    xq = bvecs_mmap(basedir + 'queries.bvecs')
    xt = bvecs_mmap(basedir + 'learn.bvecs')
    # trim xb to correct size
    xb = sanitize(xb_map[:dbsize * 1000 * 1000])
    xt = sanitize(xt[:250000])
    xq = sanitize(xq)
    gt = ivecs_read(basedir + 'gnd/idx_%dM.ivecs' % dbsize)

    return xb, xq, xt, gt

def load_random_hypercube():
    print("load random_hypercube.....",end = '', file = sys.stderr)
    basedir = '/home/wanghongya/dataset/random_hypercube/'
    xb = mmap_fvecs(basedir + 'random_hypercube_base.fvecs')
    xq = mmap_fvecs(basedir + 'random_hypercube_query.fvecs')
    xt = mmap_fvecs(basedir + 'random_hypercube_base.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'random_hypercube_groundtruth.ivecs')
    print("done", file = sys.stderr)
    return xb, xq, xt, gt

def load_random_hypersphere():
    print("load random_hypersphere.....",end = '', file = sys.stderr)
    basedir = '/home/wanghongya/dataset/random_hypersphere/'
    xb = mmap_fvecs(basedir + 'random_hypersphere_base.fvecs')
    xq = mmap_fvecs(basedir + 'random_hypersphere_query.fvecs')
    xt = mmap_fvecs(basedir + 'random_hypersphere_base.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'random_hypersphere_groundtruth.ivecs')
    print("done", file = sys.stderr)
    return xb, xq, xt, gt

def load_random_hypersphere50():
    print("load random_hypersphere50.....",end = '', file = sys.stderr)
    basedir = '/home/wanghongya/dataset/random_hypersphere50/'
    xb = mmap_fvecs(basedir + 'random_hypersphere50_base.fvecs')
    xq = mmap_fvecs(basedir + 'random_hypersphere50_query.fvecs')
    xt = mmap_fvecs(basedir + 'random_hypersphere50_base.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'random_hypersphere50_groundtruth.ivecs')
    print("done", file = sys.stderr)
    return xb, xq, xt, gt


# random_hypercube200 1M
def load_random_hypercube200():
    # simdir = '/home/wanghongya/'
    print("load random_hypercube200.....", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/random_hypercube200/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'random_hypercube200_base.fvecs')
    xq = mmap_fvecs(basedir + 'random_hypercube200_query.fvecs')
    # xt = mmap_fvecs(basedir + 'gist_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'random_hypercube200_groundtruth.ivecs')
    print("done", file=sys.stderr)
    return xb, xq, gt

# random_hypersphere200 1M
def load_random_hypersphere200():
    # simdir = '/home/wanghongya/'
    print("load random_hypersphere200.....", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/random_hypersphere200/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'random_hypersphere200_base.fvecs')
    xq = mmap_fvecs(basedir + 'random_hypersphere200_query.fvecs')
    # xt = mmap_fvecs(basedir + 'gist_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'random_hypersphere200_groundtruth.ivecs')
    print("done", file=sys.stderr)
    return xb, xq, gt

# random_laplace100 1M
def load_random_laplace100():
    # simdir = '/home/wanghongya/'
    print("load random_laplace100.....", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/random_laplace100/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'random_laplace100_base.fvecs')
    xq = mmap_fvecs(basedir + 'random_laplace100_query.fvecs')
    # xt = mmap_fvecs(basedir + 'gist_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    # xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'random_laplace100_groundtruth.ivecs')
    print("done", file=sys.stderr)
    return xb, xq, gt


def load_spacev():
    print("Loading spacev...", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/SPACEV/'
    # xt = fvecs_read(basedir + "learn.fvecs")
    # xt = read_i8bin(basedir + "spacev100m_base.i8bin")
    # xt = xt.astype('float32')
    dbsize=10
    xb = read_i8bin(basedir + "spacev100m_base.i8bin")
    xb = xb.astype('float32')
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    xt = sanitize(xb[:25*10000])

    xq = read_i8bin(basedir + "query.i8bin")
    xq = xq.astype('float32')
    gt = ivecs_read(basedir + 'spacev%dm_groundtruth.ivecs'% dbsize)
    print("done", file=sys.stderr)
    return xb, xq, gt


def load_deep(dbsize):
    print("Loading deep...", end='', file=sys.stderr)
    # basedir = simdir + 'deep1b/'
    basedir = '/data/wanghongyadatasets/deep1b/'
    
    # dbsize = 10
    # nb = dbsize * 1000 * 1000
    xb_map = mmap_fvecs(basedir + 'deep1B_base.fvecs')
    xq = mmap_fvecs(basedir + 'deep1B_queries.fvecs')
    xt = mmap_fvecs(basedir + 'deep1B_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb_map[:dbsize * 1000 * 1000])
    # xt = sanitize(xt[:500000])
    xt = xb
    xq = sanitize(xq[:10000])
    gt = ivecs_read(basedir + 'deep%dM_groundtruth.ivecs' % dbsize)
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def index_add_part(index, xb_map, nb, step=int(1e7)):
    start = 0
    end = nb
    sp = nb // step
    for i in range(start, end, step):
        t0 = time.time()
        index.add(sanitize(xb_map[i: min(end, i + step)]))
        t1 = time.time()
        print(f"add {int(i/step+1)}/{sp} in {(t1 - t0):.3f} s")

# gist 1M
def load_gist():
    # simdir = '/home/wanghongya/'
    print("load gist.....")
    basedir = '/home/wanghongya/gist/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'gist_base.fvecs')
    xq = mmap_fvecs(basedir + 'gist_query.fvecs')
    xt = mmap_fvecs(basedir + 'gist_learn.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:dbsize * 1000 * 1000])
    xt = sanitize(xt[:100000])
    xq = sanitize(xq[:1000])
    gt = ivecs_read(basedir + 'gist_groundtruth.ivecs')

    return xb, xq, xt, gt

# glove 1M
def load_glove():
    print("Loading glove1M...", end='', file=sys.stderr)

    basedir = simdir + 'glove/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'glove_base.fvecs')
    xq = mmap_fvecs(basedir + 'glove_query.fvecs')
    xb = sanitize(xb[:xb.shape[0]+1])
    xq = sanitize(xq[:10000])
    xt = xb
    gt = ivecs_read(basedir + 'glove_groundtruth.ivecs')
    print("done", file=sys.stderr)
    
    return xb, xq, xt, gt
    
def load_glove2m():
    print("Loading glove2M...", end='', file=sys.stderr)
    simdir = '/data/wanghongyadatasets/'
    basedir = simdir + 'glove2.2m/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'glove2.2m_base.fvecs')
    xq = mmap_fvecs(basedir + 'glove2.2m_query.fvecs')
    xb = sanitize(xb[:xb.shape[0]+1])
    xq = sanitize(xq[:10000])
    xt = xb
    gt = ivecs_read(basedir + 'glove2.2m_groundtruth.ivecs')
    print("done", file=sys.stderr)
    
    return xb, xq, xt, gt
    
def load_audio():
    print("Loading audio...", end='', file=sys.stderr)
    basedir = simdir + 'audio/'
    xt = fvecs_read(basedir + "audio_base.fvecs")
    xb = fvecs_read(basedir + "audio_base.fvecs")
    xq = fvecs_read(basedir + "audio_query.fvecs")
    gt = ivecs_read(basedir + "audio_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_sun():
    print("Loading sun...", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/sun/'
    xt = fvecs_read(basedir + "sun_base.fvecs")
    xb = fvecs_read(basedir + "sun_base.fvecs")
    xq = fvecs_read(basedir + "sun_query.fvecs")
    gt = ivecs_read(basedir + "sun_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_random():
    print("Loading random...", end='', file=sys.stderr)
    basedir = '/home/wanghongya/dataset/random/'
    xt = fvecs_read(basedir + "random_base.fvecs")
    xb = fvecs_read(basedir + "random_base.fvecs")
    xq = fvecs_read(basedir + "random_query.fvecs")
    gt = ivecs_read(basedir + "random_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_imageNet():
    print("Loading imageNet...", end='', file=sys.stderr)
    basedir = simdir + 'imageNet/'
    xt = fvecs_read(basedir + "imageNet_base.fvecs")
    xb = fvecs_read(basedir + "imageNet_base.fvecs")
    xq = fvecs_read(basedir + "imageNet_query.fvecs")
    gt = ivecs_read(basedir + "imageNet_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_word2vec():
    print("Loading word2vec...", end='', file=sys.stderr)
    basedir = simdir + 'word2vec/'
    xb = mmap_fvecs(basedir + 'word2vec_base.fvecs')
    xt = np.zeros((0, 0))
    xq = mmap_fvecs(basedir + 'word2vec_query.fvecs')
    xb = sanitize(xb[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + "word2vec_groundtruth.ivecs")
    print("done", file=sys.stderr)
    return xb, xq, xt, gt

def load_random_gaussian():
    print('load_random_gaussian')
    basedir = simdir + 'random_gaussian/'
    xb = mmap_fvecs(basedir + 'gaussian_base.fvecs')
    xt = mmap_fvecs(basedir + 'gaussian_base.fvecs')
    xq = mmap_fvecs(basedir + 'gaussian_query.fvecs')
    xb = sanitize(xb[:])
    xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + "gaussian_groundtruth.ivecs")
    return xb, xq, xt, gt

def load_trevi():
    print("load trevi.....")
    basedir = simdir + 'trevi/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'trevi_base.fvecs')
    xq = mmap_fvecs(basedir + 'trevi_query.fvecs')
    xt = mmap_fvecs(basedir + 'trevi_base.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    xt = sanitize(xt[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'trevi_groundtruth.ivecs')
    
    return xb, xq, xt, gt

def load_tiny80m():
    print("load Tiny80m.....",end='',file=sys.stderr)
    basedir = '/data/wanghongyadatasets/Tiny80M/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'tiny80M_base.fvecs')
    xq = mmap_fvecs(basedir + 'tiny80M_query.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    xt = sanitize(xb[:500000])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'tiny80M_groundtruth.ivecs')
    print("done",file=sys.stderr)
    return xb, xq, xt, gt

def load_tiny5m():
    print("load Tiny5m.....",end='',file=sys.stderr)
    basedir = '/data/wanghongyadatasets/tiny5m/'
    
    dbsize = 1
    xb = mmap_fvecs(basedir + 'tiny5m_base.fvecs')
    xq = mmap_fvecs(basedir + 'tiny5m_query.fvecs')
    # trim xb to correct size
    xb = sanitize(xb[:])
    xt = sanitize(xb[:])
    xq = sanitize(xq[:])
    gt = ivecs_read(basedir + 'tiny5m_groundtruth.ivecs')
    print("done",file=sys.stderr)
    return xb, xq, xt, gt

def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls
