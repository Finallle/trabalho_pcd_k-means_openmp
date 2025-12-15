# Executar e compilar K-means 1-d de forma serial

## Execução

```bash
make naive
```

ou diretamente pelo compilador GCC.

```bash
gcc -O2 -std=c99 kmeans_1d_naive.c -o kmeans_1d_naive -lm
./kmeans_1d_naive dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
```
