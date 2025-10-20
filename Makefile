kmeans_1d_naive: kmeans_1d_naive.c
	gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm

naive: kmeans_1d_naive
	./ dados.csv centroides_iniciais.csv 50 0.000001 assign.csv cent

reduction: reduction.c
	gcc -O2 -std=c99 reduction.c -o reduction -lm -fopenmp

red: reduction
	./reduction dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent 2
