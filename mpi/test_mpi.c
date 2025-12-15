#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size, resultlen;
    char hostname[256];
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Get_processor_name(hostname, &resultlen);

    printf("Rank %d of %d running on %s\n", rank, size, hostname);

    MPI_Finalize();
    return 0;
}

