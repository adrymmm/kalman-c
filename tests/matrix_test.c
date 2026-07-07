#include <stdio.h>
#include "../src/matrix.h"

int main() {
    const double X[6] = {1, 2, 3,
                         4, 5, 6};
    const double Y[6] = {1, 2, 3,
                         4, 5, 6};
    double Z[6];

    const double L[4] = {1, 2,
                         3, 4};
    double P[4];

    mat_add(X, Y, Z, 2, 3);
    mat_inv_2x2(L, P);
    mat_print(Z, 2, 3);
    printf("\n");
    mat_print(P, 2, 2);
}