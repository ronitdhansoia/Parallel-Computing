/* File:
 *     pth_mat_vect_rand_split.c
 *
 * Purpose:
 *     Computes a parallel matrix-vector product.  Matrix
 *     is distributed by block rows.  Vectors are distributed by
 *     blocks.  This version uses a random number generator to
 *     generate A and x.  It also makes some small changes to
 *     the multiplication.  These are intended to improve
 *     performance and explicitly use temporary variables.
 *
 * Input:
 *     none
 *
 * Output:
 *     y: the product vector
 *     Elapsed time for the computation
 *
 * Compile:
 *    gcc -g -Wall -o pth_mat_vect_rand pth_mat_vect_rand.c -lpthread
 * Usage:
 *     pth_mat_vect <thread_count> <m> <n>
 *
 * Notes:
 *     1.  Local storage for A, x, y is dynamically allocated.
 *     2.  Number of threads (thread_count) should evenly divide
 *         m.  The program doesn't check for this.
 *     3.  We use a 1-dimensional array for A and compute subscripts
 *         using the formula A[i][j] = A[i*n + j]
 *     4.  Distribution of A, x, and y is logical:  all three are
 *         globally shared.
 *     5.  Compile with -DDEBUG for information on generated data
 *         and product.
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "timer.h"

#define CACHE_LINE 64

/* Structs */
typedef struct {
    double value;
    char padding[CACHE_LINE - sizeof(double)];
} aligned_double;

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int crossing;
} barrier_t;

/* Global Variables */
int thread_count, m, n;
double *A, *x, *y;
aligned_double *y_aligned;
barrier_t barrier;

/* Function Declarations */
void initialize_barrier(barrier_t *barrier, int count);
void destroy_barrier(barrier_t *barrier);
void wait_barrier(barrier_t *barrier);
void generate_matrix(double *A, int m, int n);
void generate_vector(double *x, int n);
void print_results(double time_taken, int m, int n);
void *parallel_matrix_vector(void *rank);
void execute_test(int thread_count, int m, int n);

/* Main Function */
int main() {
    int test_cases[3][2] = {{8000000, 8}, {8000, 8000}, {8, 8000000}};
    int thread_options[] = {1, 2, 4, 8};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            printf("\nMatrix: %dx%d, Threads: %d\n", test_cases[i][0], test_cases[i][1], thread_options[j]);
            execute_test(thread_options[j], test_cases[i][0], test_cases[i][1]);
        }
    }
    return 0;
}

/* Function Implementations */
void execute_test(int thread_count_param, int m_param, int n_param) {
    thread_count = thread_count_param;
    m = m_param;
    n = n_param;

    pthread_t threads[thread_count];
    A = (double *)malloc(m * n * sizeof(double));
    x = (double *)malloc(n * sizeof(double));
    y = (double *)malloc(m * sizeof(double));
    y_aligned = (aligned_double *)malloc(m * sizeof(aligned_double));

    initialize_barrier(&barrier, thread_count);
    generate_matrix(A, m, n);
    generate_vector(x, n);

    double start, finish;
    GET_TIME(start);
    for (long t = 0; t < thread_count; t++)
        pthread_create(&threads[t], NULL, parallel_matrix_vector, (void *)t);
    for (long t = 0; t < thread_count; t++)
        pthread_join(threads[t], NULL);
    GET_TIME(finish);

    print_results(finish - start, m, n);
    
    free(A);
    free(x);
    free(y);
    free(y_aligned);
    destroy_barrier(&barrier);
}

void *parallel_matrix_vector(void *rank) {
    long my_rank = (long)rank;
    int local_m = m / thread_count;
    int start_row = my_rank * local_m;
    int end_row = start_row + local_m;

    for (int i = start_row; i < end_row; i++) {
        double temp = 0.0;
        for (int j = 0; j < n; j++)
            temp += A[i * n + j] * x[j];
        y[i] = temp;
    }
    return NULL;
}

void generate_matrix(double *A, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = random() / ((double)RAND_MAX);
}

void generate_vector(double *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = random() / ((double)RAND_MAX);
}

void print_results(double time_taken, int m, int n) {
    double gflops = (2.0 * m * n) / (time_taken * 1e9);
    printf("GFlops/sec: %f\n", gflops);
}

void initialize_barrier(barrier_t *barrier, int count) {
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->cond, NULL);
    barrier->count = count;
    barrier->crossing = 0;
}

void destroy_barrier(barrier_t *barrier) {
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->cond);
}

void wait_barrier(barrier_t *barrier) {
    pthread_mutex_lock(&barrier->mutex);
    barrier->crossing++;
    if (barrier->crossing < barrier->count)
        pthread_cond_wait(&barrier->cond, &barrier->mutex);
    else {
        barrier->crossing = 0;
        pthread_cond_broadcast(&barrier->cond);
    }
    pthread_mutex_unlock(&barrier->mutex);
}