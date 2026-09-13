#include <stdio.h>
#include "kalman.h"
#include "matrix.h"

double Y[] = {1.0, 2.0, 3.0, 4.0, 5.0};
#define N_OBS 5

int main() {
    KalmanState s = {
        .a = {Y[0], 0.0},
        .P = {10.0, 0.0, 0.0, 10.0},
        .H = 1.0,
        .Q = {0.01, 0.0, 0.0, 0.01},
        .Z = {1.0, 0.0},
        .T = {1.0, 1.0, 0.0, 1.0},
        .n = 2
    };

    double a_out[N_OBS * 2];  // n_obs * n, flat — same layout kalman_filter writes to

    double result = kalman_filter(Y, N_OBS, &s, a_out);

    printf("log-likelihood: %f\n", result);

    printf("filtered states (level, slope):\n");
    for (int i = 0; i < N_OBS; i++) {
        printf("  t=%d: level=%f slope=%f\n", i, a_out[i*2 + 0], a_out[i*2 + 1]);
    }

    printf("final P:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%f ", s.P[i*2 + j]);
        }
        printf("\n");
    }

    return 0;
}