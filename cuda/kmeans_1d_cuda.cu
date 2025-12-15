#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_CENTROID_INPUT 100

__constant__ float C_C[MAX_CENTROID_INPUT];

static void cuda_check(cudaError_t e, const char *msg){
    if(e != cudaSuccess){
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

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

static float *read_csv_1col_float(const char *path, int *n_out){
    
    int R = count_rows(path);
    if(R<=0){
        fprintf(stderr,"Arquivo vazio: %s\n", path);
        exit(1);
     }
    float *A = (float*)malloc((size_t)R * sizeof(float));
    
    if(!A){
        fprintf(stderr,"Sem memoria para %d linhas\n", R); 
        exit(1); 
    }

    FILE *f = fopen(path, "r");
    if(!f){ 
        fprintf(stderr,"Erro ao abrir %s\n", path);
        free(A);
        exit(1); 
    }

    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;

        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) 
            continue;

        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        
        if(!tok){
             fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); 
             free(A); 
             fclose(f); 
             exit(1); }
        
             A[r] = (float)atof(tok);
        r++;
        
        if(r>=R) 
            break;
    }
    fclose(f);
    *n_out = r;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) 
        return;
    
    FILE *f = fopen(path, "w");
    
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    
    for(int i=0;i<N;i++) 
    fprintf(f, "%d\n", assign[i]);
    
    fclose(f);
}

static void write_centroids_csv_f(const char *path, const float *C, int K){
    
    if(!path) 
        return;
    FILE *f = fopen(path, "w");
    
    if(!f){
         fprintf(stderr,"Erro ao abrir %s para escrita\n", path); 
            return; }
    for(int c=0;c<K;c++) 
        fprintf(f, "%.6f\n", (double)C[c]);
    fclose(f);
}

__global__ void assignment_step_1d(const float *X, int *assign, int N, int K, float *sse_per_point){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= N) return;

    int best = 0;
    float bestd = 1e30f;

    for(int c = 0; c < K; c++){
        float diff = X[index] - C_C[c];
        float d = diff * diff;
        if(d < bestd){ bestd = d; best = c; }
    }

    assign[index] = best;
    sse_per_point[index] = bestd;
}

static void update_step_1d(const float *X, float *C, const int *assign, int N, int K){
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }

    for(int i=0;i<N;i++){
        int a = assign[i];
        if(a < 0 || a >= K) continue;
        cnt[a] += 1;
        sum[a] += (double)X[i];
    }
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = (float)(sum[c] / (double)cnt[c]);
        else C[c] = X[0];
    }
    free(sum); free(cnt);
}

static void kmeans_1d(float *X, float *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out, int blockSize,
                      float *sse_per_point_h,
                      float *t_h2d, float *t_d2h, float *t_kernel)
{
    if(K <= 0 || N <= 0){
        *iters_out = 0;
        *sse_out = 0.0;
        return;
    }
    if(K > MAX_CENTROID_INPUT){
        fprintf(stderr, "ERRO: K=%d > MAX_CENTROID_INPUT=%d\n", K, MAX_CENTROID_INPUT);
        exit(1);
    }
    if(blockSize <= 0) blockSize = 256;

    double prev_sse = 1e300;
    double sse = 0.0;
    int it = 0;

    float *X_device = NULL;
    int *assign_device = NULL;
    float *sse_per_point_d = NULL;

    cudaEvent_t e0, e1;
    cuda_check(cudaEventCreate(&e0), "cudaEventCreate e0");
    cuda_check(cudaEventCreate(&e1), "cudaEventCreate e1");

    cuda_check(cudaEventRecord(e0), "event record e0");

    cuda_check(cudaMalloc((void**)&X_device, (size_t)N * sizeof(float)), "cudaMalloc X_device");
    cuda_check(cudaMalloc((void**)&assign_device, (size_t)N * sizeof(int)), "cudaMalloc assign_device");
    cuda_check(cudaMalloc((void**)&sse_per_point_d, (size_t)N * sizeof(float)), "cudaMalloc sse_per_point_d");

    cuda_check(cudaMemcpy(X_device, X, (size_t)N * sizeof(float), cudaMemcpyHostToDevice), "H2D X");
    cuda_check(cudaMemcpy(assign_device, assign, (size_t)N * sizeof(int), cudaMemcpyHostToDevice), "H2D assign");
    cuda_check(cudaMemcpyToSymbol(C_C, C, (size_t)K * sizeof(float)), "H2D centroids const");
    cuda_check(cudaMemset(sse_per_point_d, 0, (size_t)N * sizeof(float)), "memset sse_per_point_d");

    cuda_check(cudaEventRecord(e1), "event record e1");
    cuda_check(cudaEventSynchronize(e1), "event sync e1");
    cuda_check(cudaEventElapsedTime(t_h2d, e0, e1), "elapsed h2d");

    int numBlocks = (N + blockSize - 1) / blockSize;

    for(it = 0; it < max_iter; it++){
        sse = 0.0;

        cuda_check(cudaEventRecord(e0), "kernel event start");

        assignment_step_1d<<<numBlocks, blockSize>>>(X_device, assign_device, N, K, sse_per_point_d);
        cuda_check(cudaGetLastError(), "kernel launch");
        cuda_check(cudaDeviceSynchronize(), "kernel sync");

        cuda_check(cudaEventRecord(e1), "kernel event stop");
        cuda_check(cudaEventSynchronize(e1), "kernel event sync");
        float ms_kernel = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms_kernel, e0, e1), "elapsed kernel");
        *t_kernel += ms_kernel;

        cuda_check(cudaEventRecord(e0), "d2h event start");

        cuda_check(cudaMemcpy(assign, assign_device, (size_t)N * sizeof(int), cudaMemcpyDeviceToHost), "D2H assign");
        cuda_check(cudaMemcpy(sse_per_point_h, sse_per_point_d, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost), "D2H sse_per_point");

        cuda_check(cudaEventRecord(e1), "d2h event stop");
        cuda_check(cudaEventSynchronize(e1), "d2h event sync");
        float ms_d2h = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms_d2h, e0, e1), "elapsed d2h");
        *t_d2h += ms_d2h;

        for(int i = 0; i < N; i++) sse += (double)sse_per_point_h[i];

        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }

        update_step_1d(X, C, assign, N, K);

        cuda_check(cudaEventRecord(e0), "h2d iter event start");

        cuda_check(cudaMemcpyToSymbol(C_C, C, (size_t)K * sizeof(float)), "H2D centroids const iter");
        cuda_check(cudaMemset(sse_per_point_d, 0, (size_t)N * sizeof(float)), "memset sse_per_point_d iter");

        cuda_check(cudaEventRecord(e1), "h2d iter event stop");
        cuda_check(cudaEventSynchronize(e1), "h2d iter event sync");
        float ms_h2d_iter = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms_h2d_iter, e0, e1), "elapsed h2d iter");
        *t_h2d += ms_h2d_iter;

        prev_sse = sse;
    }

    cuda_check(cudaFree(assign_device), "cudaFree assign_device");
    cuda_check(cudaFree(X_device), "cudaFree X_device");
    cuda_check(cudaFree(sse_per_point_d), "cudaFree sse_per_point_d");

    cuda_check(cudaEventDestroy(e0), "event destroy e0");
    cuda_check(cudaEventDestroy(e1), "event destroy e1");

    *iters_out = it;
    *sse_out = sse;
}

int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv] [blockSize=256]\n", argv[0]);
        printf("Obs: arquivos CSV com 1 coluna (1 valor por linha), sem cabeçalho.\n");
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;
    int blockSize = (argc>7) ? atoi(argv[7]) : 256;

    if(max_iter <= 0 || eps <= 0.0){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    float *X = read_csv_1col_float(pathX, &N);
    float *C = read_csv_1col_float(pathC, &K);

    if(N <= 0 || K <= 0){
        fprintf(stderr,"Entrada inválida: N=%d K=%d\n", N, K);
        free(X); free(C);
        return 1;
    }

    int *assign = (int*)malloc((size_t)N * sizeof(int));
    float *sse_per_point_h = (float*)malloc((size_t)N * sizeof(float));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }
    if(!sse_per_point_h){ fprintf(stderr,"Sem memoria para sse_per_point_h\n"); free(assign); free(X); free(C); return 1; }

    for(int i=0;i<N;i++) assign[i] = 0;

    int iters = 0;
    double sse = 0.0;

    cudaEvent_t start, stop;
    cuda_check(cudaEventCreate(&start), "event create start");
    cuda_check(cudaEventCreate(&stop), "event create stop");

    float t_h2d = 0.0f, t_d2h = 0.0f, t_kernel = 0.0f;

    cuda_check(cudaEventRecord(start), "record start");

    kmeans_1d(
        X, C, assign, N, K, max_iter, eps,
        &iters, &sse, blockSize, sse_per_point_h,
        &t_h2d, &t_d2h, &t_kernel
    );

    cuda_check(cudaEventRecord(stop), "record stop");
    cuda_check(cudaEventSynchronize(stop), "sync stop");

    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed total");

    printf("K-means 1D (CUDA)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.3f ms\n", iters, sse, ms);
    printf("Tempo kernel: %.3f ms\n", t_kernel);
    printf("Tempo H2D: %.3f ms\n", t_h2d);
    printf("Tempo D2H: %.3f ms\n", t_d2h);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv_f(outCentroid, C, K);

    cuda_check(cudaEventDestroy(start), "destroy start");
    cuda_check(cudaEventDestroy(stop), "destroy stop");

    free(sse_per_point_h);
    free(assign);
    free(X);
    free(C);

    return 0;
}
