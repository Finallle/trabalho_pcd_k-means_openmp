## Executar e compilar K-means 1-d com MPI

### Apenas em um host
```bash
make single
```

ou com o comando `mpi`

```bash
mpirun -np 2 ./kmeans_1d_mpi dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
```
