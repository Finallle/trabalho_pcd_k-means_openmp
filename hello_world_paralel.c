// hello_omp.c
#include <stdio.h>
#include <omp.h>

int main(void) {
    omp_set_num_threads(4);
    #pragma omp parallel
    {
        int tid  = omp_get_thread_num();   // id da thread atual
        int nthreads = omp_get_num_threads(); // total de threads no time
        printf("Olá do thread %d de %d\n", tid, nthreads);
    }
    return 0;
}
