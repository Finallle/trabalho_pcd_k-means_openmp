kmeans_1d_naive: kmeans_1d_naive.c
	gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive.c -lm

naive: kmeans_1d_naive
	./kmeans_1d_naive dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
