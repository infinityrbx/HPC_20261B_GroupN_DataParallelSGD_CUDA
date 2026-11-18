#ifndef TIMER_H
#define TIMER_H
#include <time.h>

/* Wall-clock time in milliseconds (CPU side). For GPU work, synchronize the
 * relevant streams/devices before reading the timer. */
static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static inline int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

#endif
