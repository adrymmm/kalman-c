#include <stdio.h>
#include "kalman.h"
#include "matrix.h"

double Y[] = {1.0, 2.0, 3.0, 4.0, 5.0};


KalmanState s = {
    0.0, 1.0, 1.0, 0.1
};

int main() {

double result = kalman_filter(Y, 5, &s);
printf("log-likelihood: %f\n", result);
printf("a parameter: %f\n", s.a);
printf("P parameter: %f\n", s.P);

}