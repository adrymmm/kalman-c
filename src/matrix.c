#include <stdio.h>
#include "matrix.h"

void mat_print(const double *A, int rows, int cols) {
    /* print matrix living in a double array - (i, j) is at i*cols + j*/
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f ", A[i*cols + j]);
        }
        printf("\n");
    }
}

void mat_add(const double *A, const double *B, double *C, int rows, int cols) {
    /* Elementwise matrix addition - walking through each array*/
    int i, j;

    for (i = 0; i < (rows * cols); i++) {
        C[i] = A[i] + B[i];
    }
}

void mat_mul(const double *A, const double *B, double *C, int rows, int inner, int cols) {
    /* Assumed conformable matrices and sums dot products*/
    int i, j, k;
    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            C[i*cols + j] = 0.0;

            for (k = 0; k < inner; k++) {
                /* Dot product C(i,j) = A(i,k) * B(k, j)*/
                C[i*cols + j] += A[i*inner + k] * B[k*cols + j];
            }
        }
    }
}

void mat_transpose(const double *A, double *B, int rows, int cols) {
    /* Provide output matrix for transpose (i,j) -> (j,i)*/
    int i, j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            B[j*rows + i] = A[i*cols + j];
        }
    }
}

int mat_inv_2x2(const double *A, double *B) {
    /* Standard 2x2 matrix inversion*/
    double det = A[0]*A[3] - A[1]*A[2];

    if (det == 0) {
        fprintf(stderr, "mat_inv_2x2: matrix is singular\n");
        return -1;
    }
    
    B[0] = (1/det) * A[3];
    B[1] = -((1/det) * A[1]);
    B[2] = -((1/det) * A[2]);
    B[3] = (1/det) * A[0];
    return 0;
}

int main() {
    const double X[6] = {1, 2, 3,
                         4, 5, 6};
    const double Y[6] = {1, 2,
                         3, 4,
                        5, 6};
    double Z[6];
    mat_transpose(X,Z,2,3);
    mat_print(Z, 3, 2);
}