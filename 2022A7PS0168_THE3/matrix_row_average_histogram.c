#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10  

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    int base = N / P;
    int remainder = N % P;

    int rows = base + (rank < remainder ? 1 : 0);

   
    int *local_matrix = malloc(rows * N * sizeof(int));

    if (rank == 0) {
        int *matrix = malloc(N * N * sizeof(int));
        srand(56); 
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i * N + j] = rand() % 10;
            }
        }

        
        int *sendcounts = malloc(P * sizeof(int));
        int *displs = malloc(P * sizeof(int));
        for (int i = 0; i < P; i++) {
            int rows_i = base + (i < remainder ? 1 : 0);
            sendcounts[i] = rows_i * N;
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i - 1] + sendcounts[i - 1];
            }
        }

    
        MPI_Scatterv(matrix, sendcounts, displs, MPI_INT, local_matrix, rows * N, MPI_INT, 0, MPI_COMM_WORLD);

        free(matrix);
        free(sendcounts);
        free(displs);
    } else {
        MPI_Scatterv(NULL, NULL, NULL, MPI_INT, local_matrix, rows * N, MPI_INT, 0, MPI_COMM_WORLD);
    }

   
    int *local_avgi = malloc(rows * sizeof(int));
    for (int i = 0; i < rows; i++) {
        int sum = 0;
        for (int j = 0; j < N; j++) {
            sum += local_matrix[i * N + j];
        }
        local_avgi[i] = (sum + N - 1) / N;  // Equivalent to ceil(sum / N)
    }

    
    int *rows_per_process = malloc(P * sizeof(int));
    for (int i = 0; i < P; i++) {
        rows_per_process[i] = base + (i < remainder ? 1 : 0);
    }

    int *displs = malloc(P * sizeof(int));
    displs[0] = 0;
    for (int i = 1; i < P; i++) {
        displs[i] = displs[i - 1] + rows_per_process[i - 1];
    }

    int *global_avgi = malloc(N * sizeof(int));

    
    MPI_Allgatherv(local_avgi, rows, MPI_INT, global_avgi, rows_per_process, displs, MPI_INT, MPI_COMM_WORLD);

    // Compute histogram from global_avgi
    int histogram[10] = {0};
    for (int k = 0; k < N; k++) {
        int val = global_avgi[k];
        histogram[val]++;
    }

    char filename[20];
    sprintf(filename, "hit_%d.txt", rank);
    FILE *fp = fopen(filename, "w");
    for (int i = 0; i < 10; i++) {
        fprintf(fp, "%d: %d\n", i, histogram[i]);
    }
    fclose(fp);

    free(local_matrix);
    free(local_avgi);
    free(rows_per_process);
    free(displs);
    free(global_avgi);

    MPI_Finalize();
    return 0;
}