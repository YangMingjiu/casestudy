/*
 * MAP55672: Case Studies in HPC — Case 1
 * Question 2 & 3: TSQR (Communication-Avoiding QR)
 *
 * Usage:
 *   mpirun -np 4 ./tsqr            -> Q2: run default test and verify correctness
 *   mpirun -np 4 ./tsqr <m> <n>   -> Q3: run with given dimensions and print timing
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* LAPACK Fortran-style interface */
extern void dgeqrf_(int *m, int *n, double *A, int *lda,
                    double *tau, double *work, int *lwork, int *info);

/*
 * local_qr_get_R:
 *   Compute QR of A (m x n, column-major) via LAPACK dgeqrf.
 *   Writes the upper-triangular R (n x n, column-major) into R_out.
 *   A is overwritten in place.
 */
static void local_qr_get_R(double *A, int m, int n, double *R_out)
{
    double *tau  = (double *)malloc(n * sizeof(double));
    int     lwork = -1, info;
    double  wq;

    /* workspace query */
    dgeqrf_(&m, &n, A, &m, tau, &wq, &lwork, &info);
    lwork = (int)wq;
    double *work = (double *)malloc(lwork * sizeof(double));

    /* factorisation */
    dgeqrf_(&m, &n, A, &m, tau, work, &lwork, &info);

    /* extract upper-triangular R from first n rows (column-major) */
    memset(R_out, 0, n * n * sizeof(double));
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j; i++)
            R_out[i + j * n] = A[i + j * m];

    free(tau);
    free(work);
}

/*
 * stack_R:
 *   Vertically stack two n x n column-major matrices Ra, Rb
 *   into a (2n x n) column-major matrix out.
 */
static void stack_R(const double *Ra, const double *Rb, double *out, int n)
{
    int rows = 2 * n;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            out[i       + j * rows] = Ra[i + j * n];
        for (int i = 0; i < n; i++)
            out[(n + i) + j * rows] = Rb[i + j * n];
    }
}

/*
 * TSQR:
 *   Distributed TSQR over exactly 4 MPI processes.
 *   Each process owns m_local rows of a tall-narrow matrix (column-major).
 *
 *   Reduction tree:
 *     Level 0: each proc  i  does local QR -> R_i
 *     Level 1: proc 1 -> proc 0: merge [R0; R1] -> R01
 *              proc 3 -> proc 2: merge [R2; R3] -> R23
 *     Level 2: proc 2 -> proc 0: merge [R01; R23] -> R  (final)
 *
 *   R_final (n x n, column-major) is written on rank 0.
 */
void TSQR(double *A_local, int m_local, int n, double *R_final, int rank)
{
    double *buf     = (double *)malloc(m_local * n * sizeof(double));
    double *R_mine  = (double *)malloc(n * n * sizeof(double));
    double *R_recv  = (double *)malloc(n * n * sizeof(double));
    double *stacked = (double *)malloc(2 * n * n * sizeof(double));

    memcpy(buf, A_local, m_local * n * sizeof(double));

    /* ---- Level 0: local QR ---- */
    local_qr_get_R(buf, m_local, n, R_mine);
    free(buf);

    /* ---- Level 1 ---- */
    if (rank == 1)
        MPI_Send(R_mine, n * n, MPI_DOUBLE, 0, 10, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(R_recv, n * n, MPI_DOUBLE, 1, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        stack_R(R_mine, R_recv, stacked, n);
        local_qr_get_R(stacked, 2 * n, n, R_mine);   /* R_mine <- R01 */
    }
    if (rank == 3)
        MPI_Send(R_mine, n * n, MPI_DOUBLE, 2, 11, MPI_COMM_WORLD);
    if (rank == 2) {
        MPI_Recv(R_recv, n * n, MPI_DOUBLE, 3, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        stack_R(R_mine, R_recv, stacked, n);
        local_qr_get_R(stacked, 2 * n, n, R_mine);   /* R_mine <- R23 */
    }

    /* ---- Level 2 ---- */
    if (rank == 2)
        MPI_Send(R_mine, n * n, MPI_DOUBLE, 0, 12, MPI_COMM_WORLD);
    if (rank == 0) {
        MPI_Recv(R_recv, n * n, MPI_DOUBLE, 2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        stack_R(R_mine, R_recv, stacked, n);
        local_qr_get_R(stacked, 2 * n, n, R_final);  /* final R */
    }

    free(R_mine);
    free(R_recv);
    free(stacked);
}

/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) fprintf(stderr, "Error: must run with exactly 4 processes.\n");
        MPI_Finalize();
        return 1;
    }

    /* m and n: default (Q2 verification) or command-line (Q3 scaling) */
    int m = 1200, n = 6;
    int scaling_mode = (argc >= 3);
    if (scaling_mode) { m = atoi(argv[1]); n = atoi(argv[2]); }

    if (m % 4 != 0) {
        if (rank == 0) fprintf(stderr, "Error: m must be divisible by 4.\n");
        MPI_Finalize();
        return 1;
    }

    int m_local = m / 4;

    /* Rank 0 generates the full matrix (row-major) */
    double *A_rowmaj = NULL;
    if (rank == 0) {
        A_rowmaj = (double *)malloc(m * n * sizeof(double));
        srand(42);
        for (int i = 0; i < m * n; i++)
            A_rowmaj[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }

    /* Scatter rows to all processes */
    double *local_rowmaj = (double *)malloc(m_local * n * sizeof(double));
    MPI_Scatter(A_rowmaj, m_local * n, MPI_DOUBLE,
                local_rowmaj, m_local * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Convert to column-major for LAPACK */
    double *A_local = (double *)malloc(m_local * n * sizeof(double));
    for (int i = 0; i < m_local; i++)
        for (int j = 0; j < n; j++)
            A_local[i + j * m_local] = local_rowmaj[i * n + j];
    free(local_rowmaj);

    /* Time TSQR */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    double *R_tsqr = (double *)calloc(n * n, sizeof(double));
    TSQR(A_local, m_local, n, R_tsqr, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - t_start;

    /* ---- Output ---- */
    if (rank == 0) {
        if (scaling_mode) {
            /* Q3: print "m n time" for the benchmark script to collect */
            printf("%d %d %.6f\n", m, n, elapsed);
        } else {
            /* Q2: verify correctness against a direct LAPACK QR */
            double *A_colmaj = (double *)malloc(m * n * sizeof(double));
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    A_colmaj[i + j * m] = A_rowmaj[i * n + j];

            double *R_ref = (double *)calloc(n * n, sizeof(double));
            local_qr_get_R(A_colmaj, m, n, R_ref);

            double num = 0.0, den = 0.0;
            for (int k = 0; k < n * n; k++) {
                double d = fabs(R_tsqr[k]) - fabs(R_ref[k]);
                num += d * d;
                den += R_ref[k] * R_ref[k];
            }
            printf("Test: m=%d, n=%d, procs=4\n", m, n);
            printf("Relative |R| error (TSQR vs LAPACK reference): %.2e\n",
                   sqrt(num / den));
            printf("Wall time: %.6f s\n", elapsed);

            free(R_ref);
            free(A_colmaj);
        }
        free(A_rowmaj);
    }

    free(A_local);
    free(R_tsqr);
    MPI_Finalize();
    return 0;
}
