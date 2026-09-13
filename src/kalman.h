#ifndef KALMAN_H
#define KALMAN_H
#define MAXN 2

typedef struct {
    double a[MAXN], P[MAXN*MAXN], H, Q[MAXN*MAXN], Z[MAXN], T[MAXN*MAXN];
    int n;
} KalmanState;

void kalman_predict(KalmanState *s);
double kalman_update(KalmanState *s, double y);
double kalman_filter(double y[], int n_obs, KalmanState *s, double a_out[]);

#endif

