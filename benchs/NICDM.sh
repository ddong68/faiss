export PATH=/home/CPC/hanhan/anaconda3/bin:$PATH
faiss=/home/CPC/hanhan/faiss-1.5.0/benchs
log_path=/home/CPC/hanhan/faiss-1.5.0/log

make && make py

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.3                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr0.3.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 0.3                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr0.3.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.3                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr0.3.txt

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.5                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr0.5.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 0.5                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr0.5.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.5                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr0.5.txt

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.7                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr0.7.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 0.7                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr0.7.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.7                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr0.7.txt

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 0.9                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr0.9.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 0.9                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr0.9.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 0.9                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr0.9.txt

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1                   2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1                   2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1                  2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.txt

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1.1                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.1.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1.1                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.1.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1.1                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.1.txt

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1.3                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.3.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1.3                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.3.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1.3                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.3.txt

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1.5                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.5.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1.5                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.5.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1.5                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.5.txt

python -u $faiss/bench_hnswflat.py glove2M 5 300 10 1.7                 2>&1 | tee $log_path/hnsw_glove2M_m5_efc300_k10_dr1.7.txt
python -u $faiss/bench_hnswflat.py glove2M 8 300 10 1.7                 2>&1 | tee $log_path/hnsw_glove2M_m8_efc300_k10_dr1.7.txt
python -u $faiss/bench_hnswflat.py glove2M 16 300 10 1.7                2>&1 | tee $log_path/hnsw_glove2M_m16_efc300_k10_dr1.7.txt