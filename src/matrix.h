#ifndef MATRIX_H
#define MATRIX_H

/* Prints a matrix to stdout with the rows and columns specified*/
void mat_print(const double *A, int rows, int cols);

/* Adds two matrices together and mutates C to be their sum*/
void mat_add(const double *A, const double *B, double *C, int rows, int cols);

/* Performs matrix multiplication assuming conformable matrices through rows x inner x cols*/
void mat_mul(const double *A, const double *B, double *C, int rows, int inner, int cols);

/* Transposes a matrix and mutates B to be its transpose */
void mat_transpose(const double *A, double *B, int rows, int cols);

/* Finds the inverse of a 2x2 matrix and mutates B to be the inverse */
int mat_inv_2x2(const double *A, double *B);

#endif