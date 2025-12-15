# Executar e compilar K-means 1-d com CUDA

## Configuração e Preparação
Antes de executar, você precisa ter a biblioteca do CUDA instalado em seu host.

### Ubuntu/Debian
```bash
sudo apt install cuda
```

### Arch Linux
```bash
sudo pacman -S cuda
```

## Execução
```bash
make
```

ou diretamente pelo comando `nvcc`.

```bash
nvcc -O2 kmeans_1d_cuda.cu -o kmeans_1d_cuda
./kmeans_1d_cuda dados_big.csv centroides_k64.csv 50 0.000001 assign.csv cent
```
