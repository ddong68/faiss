export PATH=/home/wanghongya/anaconda3/bin:$PATH
faiss=/home/wanghongya/dongdong/faiss-1.5.0/benchs
log_path=/home/wanghongya/dongdong/faiss-1.5.0/log/hnswpq

make && make py



# python -u $faiss/bench_hnswpq.py random 5 300 10                2>&1 | tee $log_path/random_m5_efc300_pq_m10.txt
# python -u $faiss/bench_hnswpq.py random 8 300 10                2>&1 | tee $log_path/random_m8_efc300_pq_m10.txt
# python -u $faiss/bench_hnswpq.py random 16 300 10               2>&1 | tee $log_path/random_m16_efc300_pq_m10.txt

# python -u $faiss/bench_hnswpq.py random_gaussian 5 300 10                2>&1 | tee $log_path/random_gaussian_m5_efc300_pq_m10.txt
# python -u $faiss/bench_hnswpq.py random_gaussian 8 300 10                2>&1 | tee $log_path/random_gaussian_m8_efc300_pq_m10.txt
# python -u $faiss/bench_hnswpq.py random_gaussian 16 300 10               2>&1 | tee $log_path/random_gaussian_m16_efc300_pq_m10.txt


# python -u $faiss/bench_hnswpq.py deep1M 5 300 4 1               2>&1 | tee $log_path/deep1M_m5_efc300_pq_m4.txt
# python -u $faiss/bench_hnswpq.py deep1M 8 300 4 1               2>&1 | tee $log_path/deep1M_m8_efc300_pq_m4.txt
# python -u $faiss/bench_hnswpq.py deep1M 16 300 4 1              2>&1 | tee $log_path/deep1M_m16_efc300_pq_m4.txt

# python -u $faiss/bench_hnswpq.py glove1M 5 300 10                2>&1 | tee $log_path/glove1M_m5_efc300_pq_m10.txt
# python -u $faiss/bench_hnswpq.py glove1M 8 300 10                2>&1 | tee $log_path/glove1M_m8_efc300_pq_m10.txt
# python -u $faiss/bench_hnswpq.py glove1M 16 300 10               2>&1 | tee $log_path/glove1M_m16_efc300_pq_m10.txt

# python -u $faiss/bench_hnswpq.py glove2M 5 300 10                2>&1 | tee $log_path/glove2M_m5_efc300_pq_m10.txt
# python -u $faiss/bench_hnswpq.py glove2M 8 300 10                2>&1 | tee $log_path/glove2M_m8_efc300_pq_m10.txt
# python -u $faiss/bench_hnswpq.py glove2M 16 300 10               2>&1 | tee $log_path/glove2M_m16_efc300_pq_m10.txt

python -u $faiss/bench_hnswpq.py tiny5m 5 300 16                2>&1 | tee $log_path/tiny5m_m5_efc300_pq_m16.txt
python -u $faiss/bench_hnswpq.py tiny5m 8 300 16                2>&1 | tee $log_path/tiny5m_m8_efc300_pq_m16.txt
python -u $faiss/bench_hnswpq.py tiny5m 16 300 16               2>&1 | tee $log_path/tiny5m_m16_efc300_pq_m16.txt

python -u $faiss/bench_hnswpq.py deep 5 300 4 10               2>&1 | tee $log_path/deep10M_m5_efc300_pq_m4.txt
python -u $faiss/bench_hnswpq.py deep 8 300 4 10               2>&1 | tee $log_path/deep10M_m8_efc300_pq_m4.txt
python -u $faiss/bench_hnswpq.py deep 16 300 4 10              2>&1 | tee $log_path/deep10M_m16_efc300_pq_m4.txt