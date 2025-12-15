# Executar e compilar K-means 1-d com OpenMP

## Configuração e Preparação
Antes de executar, você precisa ter a biblioteca do OpenMP instalado em seu host. Em Ubuntu/Debian já vem pré-instalado como uma <i>feature</i> do compilador.

### Arch Linux
```bash
sudo pacman -S openmp
```

## Execução

```bash
make omp
```

ou diretamente pelo compilador GCC.

```bash
gcc -O2 -std=c99 kmeans_1d_omp.c -o kmeans_1d_omp -lm -fopenmp
./kmeans_1d_omp dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
```
