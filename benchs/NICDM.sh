faiss=/home/wanghongya/hanhan/faiss-1.5.0/benchs
log_path=/home/wanghongya/hanhan/faiss-1.5.0/log

make && make py

python $faiss/bench_hnswpq.py sift1M 5 300 16 0.01
# python $faiss/bench_hnswpq.py glove1M 5 300 20 0.01
# python -u $faiss/bench_hnswpq.py glove1M 5 300 10 0.01                  2>&1 | tee $log_path/hnsw_glove1M_m5_efc300_pqm10_sr0.01.txt
# python -u $faiss/bench_hnswpq.py sift1M 5 300 16 0.01                   2>&1 | tee $log_path/hnsw_sift1M_m5_efc300_pqm16_sr0.01.txt
# python -u $faiss/bench_hnswpq.py bigann 5 300 16 0.001 10               2>&1 | tee $log_path/hnsw_sift10M_m5_efc300_pqm16_sr0.001.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.3                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr0.3.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 0.3                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr0.3.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.3                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr0.3.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.5                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr0.5.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 0.5                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr0.5.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.5                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr0.5.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.7                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr0.7.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 0.7                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr0.7.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.7                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr0.7.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.9                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr0.9.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 0.9                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr0.9.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.9                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr0.9.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1                   2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1                   2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1                  2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1.1                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.1.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1.1                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.1.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1.1                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.1.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1.3                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.3.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1.3                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.3.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1.3                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.3.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1.5                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.5.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1.5                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.5.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1.5                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.5.txt

# python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1.7                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.7.txt
# python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1.7                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.7.txt
# python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1.7                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.7.txt