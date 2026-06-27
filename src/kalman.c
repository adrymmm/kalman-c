#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "kalman.h"

void kalman_predict(KalmanState *s) {
    s->P += s->Q;
}

double kalman_update(KalmanState *s, double y) {
    double v, F, K;

    v = y - s->a;
    F = s->P + s->H;
    K = s->P / F;

    s->a += K * v;
    s->P = (1-K) * s->P;

    // Return log likelihood contribution
    return -0.5 * (log(2*M_PI) + log(F) + (v*v / F));
}

double kalman_filter(double y[], int T, KalmanState *s) {
    int i;
    double Lt;
    double Lt_sum = 0;

    for (i = 0; i < T; i++) {
        kalman_predict(s);
        Lt = kalman_update(s, y[i]);
        Lt_sum += Lt;
    }
    return Lt_sum;
}