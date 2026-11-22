#include "params.h"
#include "config.h"
#include "mnist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

ParamLayout param_layout(int H) {
    ParamLayout L;
    L.H = H;
    L.W1 = 0;
    L.b1 = L.W1 + (size_t)CFG_INPUT * H;
    L.W2 = L.b1 + H;
    L.b2 = L.W2 + (size_t)H * CFG_CLASSES;
    L.total = L.b2 + CFG_CLASSES;
    return L;
}

void init_params(float *p, const ParamLayout *L, uint32_t seed) {
    uint32_t s = seed;
    float a1 = sqrtf(6.0f / (CFG_INPUT + L->H));
    float a2 = sqrtf(6.0f / (L->H + CFG_CLASSES));
    for (size_t i = 0; i < (size_t)CFG_INPUT * L->H; i++) p[L->W1 + i] = (2.0f * rng_uniform(&s) - 1.0f) * a1;
    for (size_t i = 0; i < (size_t)L->H * CFG_CLASSES; i++) p[L->W2 + i] = (2.0f * rng_uniform(&s) - 1.0f) * a2;
    memset(p + L->b1, 0, L->H * sizeof(float));
    memset(p + L->b2, 0, CFG_CLASSES * sizeof(float));
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s [--hidden H] [--epochs E] [--batch B] [--lr LR] [--seed S]\n"
        "          [--gpus K] [--method NAME] [--overlap] [--data DIR]\n"
        "          [--csv FILE] [--loss-log FILE]\n"
        "Defaults come from common/config.h. Use the defaults for final results.\n", prog);
    exit(1);
}

void parse_options(int argc, char **argv, Options *o) {
    o->hidden = CFG_HIDDEN; o->epochs = CFG_EPOCHS; o->batch = CFG_GLOBAL_BATCH;
    o->lr = CFG_LR; o->seed = CFG_SEED; o->gpus = 1; o->method = "host_ref";
    o->data_dir = CFG_DATA_DIR; o->csv = NULL; o->loss_log = NULL; o->overlap = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int has = i + 1 < argc;
        if      (!strcmp(a, "--hidden")   && has) o->hidden = atoi(argv[++i]);
        else if (!strcmp(a, "--epochs")   && has) o->epochs = atoi(argv[++i]);
        else if (!strcmp(a, "--batch")    && has) o->batch = atoi(argv[++i]);
        else if (!strcmp(a, "--lr")       && has) o->lr = (float)atof(argv[++i]);
        else if (!strcmp(a, "--seed")     && has) o->seed = (unsigned)strtoul(argv[++i], NULL, 10);
        else if (!strcmp(a, "--gpus")     && has) o->gpus = atoi(argv[++i]);
        else if (!strcmp(a, "--method")   && has) o->method = argv[++i];
        else if (!strcmp(a, "--data")     && has) o->data_dir = argv[++i];
        else if (!strcmp(a, "--csv")      && has) o->csv = argv[++i];
        else if (!strcmp(a, "--loss-log") && has) o->loss_log = argv[++i];
        else if (!strcmp(a, "--overlap"))         o->overlap = 1;
        else usage(argv[0]);
    }
}
