export PATH=/home/CPC/hanhan/anaconda3/bin:$PATH
faiss=/home/CPC/hanhan/faiss-1.5.0/benchs
log_path=/home/CPC/hanhan/faiss-1.5.0/log

make && make py
python -u $faiss/bench_hnswflat.py audio 5 300 10 30                2>&1 | tee $log_path/hnsw_audio_m5_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py audio 8 300 10 30                2>&1 | tee $log_path/hnsw_audio_m8_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py audio 16 300 10 30               2>&1 | tee $log_path/hnsw_audio_m16_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py audio 5 300 10 45                2>&1 | tee $log_path/hnsw_audio_m5_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py audio 8 300 10 45                2>&1 | tee $log_path/hnsw_audio_m8_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py audio 16 300 10 45               2>&1 | tee $log_path/hnsw_audio_m16_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py audio 5 300 10 60                2>&1 | tee $log_path/hnsw_audio_m5_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py audio 8 300 10 60                2>&1 | tee $log_path/hnsw_audio_m8_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py audio 16 300 10 60               2>&1 | tee $log_path/hnsw_audio_m16_efc300_k10_angle60.txt

python -u $faiss/bench_hnswflat.py glove1M 5 300 10 30              2>&1 | tee $log_path/hnsw_glove1M_m5_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py glove1M 8 300 10 30              2>&1 | tee $log_path/hnsw_glove1M_m8_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py glove1M 16 300 10 30             2>&1 | tee $log_path/hnsw_glove1M_m16_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py glove1M 5 300 10 45              2>&1 | tee $log_path/hnsw_glove1M_m5_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py glove1M 8 300 10 45              2>&1 | tee $log_path/hnsw_glove1M_m8_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py glove1M 16 300 10 45             2>&1 | tee $log_path/hnsw_glove1M_m16_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py glove1M 5 300 10 60              2>&1 | tee $log_path/hnsw_glove1M_m5_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py glove1M 8 300 10 60              2>&1 | tee $log_path/hnsw_glove1M_m8_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py glove1M 16 300 10 60             2>&1 | tee $log_path/hnsw_glove1M_m16_efc300_k10_angle60.txt

python -u $faiss/bench_hnswflat.py sift1M 5 300 10 30               2>&1 | tee $log_path/hnsw_sift1M_m5_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py sift1M 8 300 10 30               2>&1 | tee $log_path/hnsw_sift1M_m8_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py sift1M 16 300 10 30              2>&1 | tee $log_path/hnsw_sift1M_m16_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py sift1M 5 300 10 45               2>&1 | tee $log_path/hnsw_sift1M_m5_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py sift1M 8 300 10 45               2>&1 | tee $log_path/hnsw_sift1M_m8_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py sift1M 16 300 10 45              2>&1 | tee $log_path/hnsw_sift1M_m16_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py sift1M 5 300 10 60               2>&1 | tee $log_path/hnsw_sift1M_m5_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py sift1M 8 300 10 60               2>&1 | tee $log_path/hnsw_sift1M_m8_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py sift1M 16 300 10 60              2>&1 | tee $log_path/hnsw_sift1M_m16_efc300_k10_angle60.txt

python -u $faiss/bench_hnswflat.py random_gaussian 5 300 10 30      2>&1 | tee $log_path/hnsw_random_gaussian_m5_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py random_gaussian 8 300 10 30      2>&1 | tee $log_path/hnsw_random_gaussian_m8_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py random_gaussian 16 300 10 30     2>&1 | tee $log_path/hnsw_random_gaussian_m16_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py random_gaussian 5 300 10 45      2>&1 | tee $log_path/hnsw_random_gaussian_m5_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py random_gaussian 8 300 10 45      2>&1 | tee $log_path/hnsw_random_gaussian_m8_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py random_gaussian 16 300 10 45     2>&1 | tee $log_path/hnsw_random_gaussian_m16_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py random_gaussian 5 300 10 60      2>&1 | tee $log_path/hnsw_random_gaussian_m5_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py random_gaussian 8 300 10 60      2>&1 | tee $log_path/hnsw_random_gaussian_m8_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py random_gaussian 16 300 10 60     2>&1 | tee $log_path/hnsw_random_gaussian_m16_efc300_k10_angle60.txt

python -u $faiss/bench_hnswflat.py imageNet 5 300 10 30             2>&1 | tee $log_path/hnsw_imageNet_m5_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py imageNet 8 300 10 30             2>&1 | tee $log_path/hnsw_imageNet_m8_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py imageNet 16 300 10 30            2>&1 | tee $log_path/hnsw_imageNet_m16_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py imageNet 5 300 10 45             2>&1 | tee $log_path/hnsw_imageNet_m5_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py imageNet 8 300 10 45             2>&1 | tee $log_path/hnsw_imageNet_m8_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py imageNet 16 300 10 45            2>&1 | tee $log_path/hnsw_imageNet_m16_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py imageNet 5 300 10 60             2>&1 | tee $log_path/hnsw_imageNet_m5_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py imageNet 8 300 10 60             2>&1 | tee $log_path/hnsw_imageNet_m8_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py imageNet 16 300 10 60            2>&1 | tee $log_path/hnsw_imageNet_m16_efc300_k10_angle60.txt

python -u $faiss/bench_hnswflat.py word2vec 5 300 10 30             2>&1 | tee $log_path/hnsw_word2vec_m5_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py word2vec 8 300 10 30             2>&1 | tee $log_path/hnsw_word2vec_m8_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py word2vec 16 300 10 30            2>&1 | tee $log_path/hnsw_word2vec_m16_efc300_k10_angle30.txt
python -u $faiss/bench_hnswflat.py word2vec 5 300 10 45             2>&1 | tee $log_path/hnsw_word2vec_m5_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py word2vec 8 300 10 45             2>&1 | tee $log_path/hnsw_word2vec_m8_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py word2vec 16 300 10 45            2>&1 | tee $log_path/hnsw_word2vec_m16_efc300_k10_angle45.txt
python -u $faiss/bench_hnswflat.py word2vec 5 300 10 60             2>&1 | tee $log_path/hnsw_word2vec_m5_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py word2vec 8 300 10 60             2>&1 | tee $log_path/hnsw_word2vec_m8_efc300_k10_angle60.txt
python -u $faiss/bench_hnswflat.py word2vec 16 300 10 60            2>&1 | tee $log_path/hnsw_word2vec_m16_efc300_k10_angle60.txt