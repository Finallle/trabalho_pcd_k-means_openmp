kmeans_1d_naive: kmeans_1d_naive.c
	gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm

naive: kmeans_1d_naive
	./kmeans_1d_naive dados.csv centroides_iniciais.csv 50 0.000001 assign.csv cent

kmeans_1d_omp: kmeans_1d_omp.c
	gcc -O2 -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm -fopenmp

omp: kmeans_1d_omp
	./kmeans_1d_omp dados.csv centroides_iniciais.csv 50 0.000001 assign.csv cent 2
