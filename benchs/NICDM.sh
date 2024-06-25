export PATH=/home/wanghongya/anaconda3/bin:$PATH
faiss=/home/wanghongya/dongdong/faiss-1.5.0/benchs
log_path=/home/wanghongya/dongdong/faiss-1.5.0/log

make && make py


#  python -u $faiss/bench_hnswflat.py random_hypercube200 5 300 10 0.01 1.0                2>&1 | tee $log_path/random_hypercube200_m5_efc300_k10_0.01_1.0.txt
#  python -u $faiss/bench_hnswflat.py random_hypercube200 8 300 10 0.01 1.0                2>&1 | tee $log_path/random_hypercube200_m8_efc300_k10_0.01_1.0.txt
# python -u $faiss/bench_hnswflat.py random_hypercube200 16 300 10 0.01 1.0               2>&1 | tee $log_path/random_hypercube200_m16_efc300_k10_0.01_1.0.txt

python -u $faiss/hot_hubs_random.py 5 300 10 random_hypersphere                 2>&1 | tee $log_path/random_hypersphere/hnsw_random_hypersphere_m5_efc300_k10.txt
python -u $faiss/hot_hubs_random.py 16 300 10 random_hypersphere                2>&1 | tee $log_path/random_hypersphere/hnsw_random_hypersphere_m16_efc300_k10.txt

python -u $faiss/hot_hubs_random.py 5 300 10 random_hypercube              2>&1 | tee $log_path/random_hypercube/hnsw_random_hypercube_m5_efc300_k10.txt
python -u $faiss/hot_hubs_random.py 16 300 10 random_hypercube              2>&1 | tee $log_path/random_hypercube/hnsw_random_hypercube_m16_efc300_k10.txt


python -u $faiss/hot_hubs_random.py 5 300 10 random_hypercube200              2>&1 | tee $log_path/random_hypercube200/hnsw_random_hypercube200_m5_efc300_k10.txt
python -u $faiss/hot_hubs_random.py 16 300 10 random_hypercube200              2>&1 | tee $log_path/random_hypercube200/hnsw_random_hypercube200_m16_efc300_k10.txt


# python -u $faiss/bench_hnswflat.py random_hypersphere200 5 300 10 0.01 1.0                2>&1 | tee $log_path/random_hypersphere200_m5_efc300_k10_0.01_1.0.txt
# python -u $faiss/bench_hnswflat.py random_hypersphere200 8 300 10 0.01 1.0                2>&1 | tee $log_path/random_hypersphere200_m8_efc300_k10_0.01_1.0.txt
# python -u $faiss/bench_hnswflat.py random_hypersphere200 16 300 10 0.01 1.0               2>&1 | tee $log_path/random_hypersphere200_m16_efc300_k10_0.01_1.0.txt

python -u $faiss/hot_hubs_random.py 5 300 10 random_hypersphere200                 2>&1 | tee $log_path/random_hypersphere200/hnsw_random_hypersphere200_m5_efc300_k10.txt
python -u $faiss/hot_hubs_random.py 16 300 10 random_hypersphere200                2>&1 | tee $log_path/random_hypersphere200/hnsw_random_hypersphere200_m16_efc300_k10.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.1 1.0               2>&1 | tee $log_path/glove2M/glove2M_m5_efc300_k10_0.1_1.0.txt
# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.05 1.0               2>&1 | tee $log_path/glove2M/glove2M_m5_efc300_k10_0.05_1.0.txt
# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.03 1.0               2>&1 | tee $log_path/glove2M/glove2M_m5_efc300_k10_0.03_1.0.txt

# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.1 1.0               2>&1 | tee $log_path/glove2M/glove2M_m16_efc300_k10_0.1_1.0.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.05 1.0               2>&1 | tee $log_path/glove2M/glove2M_m16_efc300_k10_0.05_1.0.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.03 1.0               2>&1 | tee $log_path/glove2M/glove2M_m16_efc300_k10_0.03_1.0.txt

# python -u $faiss/bench_hnswflat.py random_hypersphere50 5 300 10 0.01 1.0               2>&1 | tee $log_path/random_hypersphere50/random_hypersphere50_m5_efc300_k10_0.01_1.0.txt
# python -u $faiss/bench_hnswflat.py random_hypersphere50 16 300 10 0.01 1.0               2>&1 | tee $log_path/random_hypersphere50/random_hypersphere50_m16_efc300_k10_0.01_1.0.txt

# python -u $faiss/bench_hnswflat.py random_hypersphere50 16 300 10 0.01 1.0 32               2>&1 | tee $log_path/random_hypersphere50/random_hypersphere50_m16_efc300_k10_0.01_1.0_mIn32.txt
# python -u $faiss/bench_hnswflat.py random_hypersphere50 16 300 10 0.01 1.0 64               2>&1 | tee $log_path/random_hypersphere50/random_hypersphere50_m16_efc300_k10_0.01_1.0_mIn64.txt


python -u $faiss/hot_hubs_random.py 5 300 10 random_hypersphere50                 2>&1 | tee $log_path/random_hypersphere50/hnsw_random_hypersphere50_m5_efc300_k10.txt
python -u $faiss/hot_hubs_random.py 16 300 10 random_hypersphere50                2>&1 | tee $log_path/random_hypersphere50/hnsw_random_hypersphere50_m16_efc300_k10.txt

