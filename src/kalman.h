#ifndef KALMAN_H
#define KALMAN_H

typedef struct {
    double a, P, H, Q;
} KalmanState;

void kalman_predict(KalmanState *s);
double kalman_update(KalmanState *s, double y);
double kalman_filter(double y[], int T, KalmanState *s);

#endif

