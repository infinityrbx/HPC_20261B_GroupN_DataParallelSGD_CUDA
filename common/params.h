#ifndef PARAMS_H
#define PARAMS_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Offsets into the flat parameter / gradient buffer (see config.h). */
typedef struct {
    int H;
    size_t W1, b1, W2, b2, total;
} ParamLayout;

ParamLayout param_layout(int H);

/* Xavier-uniform init for W1, W2; zeros for biases. Deterministic in seed,
 * so the CPU baseline and every GPU start from identical weights. */
void init_params(float *p, const ParamLayout *L, uint32_t seed);

/* Simple CLI parsing shared by all executables. */
typedef struct {
    int hidden, epochs, batch, gpus;
    float lr;
    unsigned seed;
    const char *method;    /* allreduce method name            */
    const char *data_dir;
    const char *csv;       /* append results here (optional)   */
    const char *loss_log;  /* per-epoch loss log (optional)    */
    int overlap;           /* 1 = per-layer bucketed allreduce */
} Options;

void parse_options(int argc, char **argv, Options *o);

#ifdef __cplusplus
}
#endif
#endif
