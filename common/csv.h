#ifndef CSV_H
#define CSV_H
#include <stdio.h>
#include <stddef.h>

/* Every executable writes results with EXACTLY these columns.
 * Use -1 for a column that does not apply to that run
 * (e.g. busbw_GBs for a training run, test_acc for a raw allreduce run). */
#define CSV_HEADER "method,k,msg_bytes,time_ms,busbw_GBs,epoch_s,comm_pct,test_acc\n"

static inline void csv_write(const char *path, const char *method, int k, size_t msg_bytes,
                             double time_ms, double busbw_GBs, double epoch_s,
                             double comm_pct, double test_acc) {
    FILE *f = NULL;
    int need_header = 1;
    if (path) {
        FILE *t = fopen(path, "r");
        if (t) { need_header = fgetc(t) == EOF; fclose(t); }
        f = fopen(path, "a");
    }
    if (!f) f = stdout;
    if (need_header) fputs(CSV_HEADER, f);
    fprintf(f, "%s,%d,%zu,%.6f,%.4f,%.4f,%.3f,%.4f\n",
            method, k, msg_bytes, time_ms, busbw_GBs, epoch_s, comm_pct, test_acc);
    if (f != stdout) fclose(f);
}

#endif
