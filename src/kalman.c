#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "kalman.h"

void kalman_predict(KalmanState *s) {
    int n = s->n;
    double a_temp[MAXN];
    double TP[MAXN*MAXN], Tt[MAXN*MAXN], TPTt[MAXN*MAXN];

    // a_pred = Ta
    mat_mul(s->T, s->a, a_temp, n, n, 1);

    // T transpose
    mat_transpose(s->T, Tt, n, n);

    //TP = T*P
    mat_mul(s->T, s->P, TP, n, n, n);

    // TPT' = TP*T'
    mat_mul(TP, Tt, TPTt, n, n, n);

    // P = TPT' + Q
    mat_add(TPTt, s->Q, s->P, n, n);

    for (int i = 0; i < n; i++) {
        s->a[i] = a_temp[i];
    }
}

double kalman_update(KalmanState *s, double y) {
    int n = s->n;
    double v_t, F;
    double Za, ZPZt; // scalar
    double Zt[MAXN], PZt[MAXN], PZtF_inv[MAXN], Kv[MAXN], KZ[MAXN*MAXN];
    double KZP[MAXN*MAXN];

    // Z*a -> Scalar Za
    mat_mul(s->Z, s->a, &Za, 1, n, 1);

    v_t = y - Za;

    mat_transpose(s->Z, Zt, 1, n);
    mat_mul(s->P, Zt, PZt, n, n, 1);
    // Z * P * Z' [(1 x n) x (n x n) x (n x 1)]
    mat_mul(s->Z, PZt, &ZPZt, 1, n, 1);

    F = ZPZt + s->H;

    if (F == 0.0) {
        fprintf(stderr, "kalman_update: F is zero, singular\n");
        return NAN;
    }
    // Inverse of a scalar is 1/X
    double F_inv = 1.0 / F;

    // K = PZ'F^-1 (n x 1)
    mat_mul(PZt, &F_inv, PZtF_inv, n, 1, 1);

    // Kv [(n x 1) x (1 x 1)] = n x 1
    mat_mul(PZtF_inv, &v_t, Kv, n, 1, 1);

    // a = a + Kv 
    for (int i = 0; i < n; i++) {
        s->a[i] = s->a[i] + Kv[i];
    }
    
    // KZ (n x 1) x (1 x n)
    mat_mul(PZtF_inv, s->Z, KZ, n, 1, n);

    // P = P + KZP
    // KZP -> (n x 1) x (1 x n) x (n x n)
    mat_mul(KZ, s->P, KZP, n, n, n);

    for (int i = 0; i < n*n; i++) {
        s->P[i] = s->P[i] - KZP[i];
    }
    // Return log likelihood contribution
    return -0.5 * (log(2*M_PI) + log(F) + (v_t*v_t / F));
}

double kalman_filter(double y[], int n_obs, KalmanState *s, double a_out[]) {
    int n = s->n;
    int i, j;
    double Lt;
    double Lt_sum = 0;

    for (i = 0; i < n_obs; i++) {
        kalman_predict(s);
        Lt = kalman_update(s, y[i]);
        Lt_sum += Lt;
        for (j=0; j < n; j++) {
            a_out[i*n+j] = s->a[j];
        }
    }
    return Lt_sum;
}