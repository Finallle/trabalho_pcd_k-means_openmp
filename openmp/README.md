## Executar e compilar K-means 1-d com OpenMP

```bash
make omp
```

ou

```bash
gcc -O2 -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm -fopenmp
./kmeans_1d_omp dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
```
