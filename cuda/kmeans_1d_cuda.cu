#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h> // CUDA API library

#define MAX_CENTROID_INPUT 100

__constant__ double C_C[MAX_CENTROID_INPUT]; // Let's declare centroids into constant memory

/* ---------- util CSV 1D: cada linha tem 1 número ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }

    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;

        /* aceita vírgula/ponto-e-vírgula/espaco/tab, pega o primeiro token numérico */
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = atof(tok);
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- k-means 1D ---------- */
/* assignment: para cada X[i], encontra c com menor (X[i]-C[c])^2 */
__global__ void assignment_step_1d(const double *X, int *assign, int N, int K, double* sse_per_point){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int stride = blockDim.x * gridDim.x;

    if(index >= N) return;

    int best = -1;
    double bestd = 1e300;
    for(int c = 0; c < K; c++){
        double diff = X[index] - C_C[c];
        double d = diff*diff;
        if(d < bestd){ bestd = d; best = c; }
    }

    __syncthreads();

    assign[index] = best;
    sse_per_point[index] += bestd;
}

/* update: média dos pontos de cada cluster (1D)
   se cluster vazio, copia X[0] (estratégia naive) */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K){
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    for(int i=0;i<N;i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else C[c] = X[0]; /* simples: cluster vazio recebe o primeiro ponto */
    }
    free(sum); free(cnt);
}

static void kmeans_1d(double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out, int blockSize, double* sse_per_point_h,
                      float* t_h2d, float* t_d2h, float* t_kernel)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;

    double *X_device, *sse_per_point_d;
    int *assign_device;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMalloc((void**)&X_device, sizeof(double) * N);
    cudaMalloc((void**)&assign_device, (size_t)N * sizeof(int));
    cudaMalloc((void**)&sse_per_point_d, sizeof(double) * N);

    cudaMemcpy(X, X_device, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(assign, assign_device, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(C_C, C, N * sizeof(double));
    cudaMemcpy(sse_per_point_h, sse_per_point_d, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(t_h2d, start, stop);

    cudaPeekAtLastError();
    cudaDeviceSynchronize();

    int numBlocks = (N + blockSize - 1) / blockSize; // Out of the Blocks
    
    float kernel = 0, d2h = 0;
    for(it=0; it<max_iter; it++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        assignment_step_1d<<<numBlocks, blockSize>>>(X_device, assign_device, N, K, sse_per_point_d);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&kernel, start, stop);
        *t_kernel += kernel;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaMemcpy(sse_per_point_d, sse_per_point_h, N * sizeof(double), cudaMemcpyDeviceToHost);

        for(int i = 0; i < N; i++)
            sse += sse_per_point_h[i];

        /* parada por variação relativa do SSE */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }

        cudaMemcpy(X_device, X, N * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(assign_device, assign, N * sizeof(double), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&d2h, start, stop);
        *t_d2h += d2h;

        update_step_1d(X, C_C, assign, N, K);

        prev_sse = sse;
    }

    cudaFree(assign_device);
    cudaFree(X_device);
    cudaFree(sse_per_point_d);

    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [blockSize=1] \n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;
    int blockSize = (argc>7) ? atoi(argv[7]) : 1;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    double *X = read_csv_1col(pathX, &N);
    double *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    double *sse_per_point_h = (double*)malloc((size_t)N * sizeof(double));

    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }
    if(!sse_per_point_h){ fprintf(stderr,"Sem memoria para sse_per_point_h\n"); free(X); free(C); return 1; }

    int iters = 0; double sse = 0.0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    float t_h2d = 0.0, t_d2h = 0.0, t_kernel = 0.0;
    kmeans_1d(
        X, C, assign, N, K, max_iter, eps, &iters, &sse, blockSize, sse_per_point_h,
        &t_h2d, &t_d2h, &t_kernel
    );
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("K-means 1D (CUDA)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);
    printf("Tempo kernel: %.2f ms\n", t_kernel);
    printf("Tempo H2D: %.2f ms\n", t_h2d);
    printf("Tempo D2H: %.2f ms\n", t_d2h);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}
