all: kmeans_1d_mpi single

kmeans_1d_mpi: kmeans_1d_mpi.c
	mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm

single: kmeans_1d_mpi
	mpirun -np 2 ./kmeans_1d_mpi dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent

hosts: kmeans_1d_mpi
	mpirun -np 8 --hostfile hosts ./kmeans_1d_mpi dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
