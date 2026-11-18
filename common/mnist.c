#include "mnist.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t read_be32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}

int mnist_load(const char *dir, const char *split, Dataset *out) {
    char path[512];
    memset(out, 0, sizeof(*out));

    snprintf(path, sizeof path, "%s/%s-images-idx3-ubyte", dir, split);
    FILE *fi = fopen(path, "rb");
    if (!fi) { fprintf(stderr, "cannot open %s (run scripts/get_mnist.sh)\n", path); return -1; }
    if (read_be32(fi) != 2051) { fprintf(stderr, "bad magic in %s\n", path); fclose(fi); return -1; }
    int n = (int)read_be32(fi), rows = (int)read_be32(fi), cols = (int)read_be32(fi);
    if (rows * cols != CFG_INPUT) { fprintf(stderr, "unexpected image size\n"); fclose(fi); return -1; }

    unsigned char *raw = (unsigned char *)malloc((size_t)n * CFG_INPUT);
    if (fread(raw, 1, (size_t)n * CFG_INPUT, fi) != (size_t)n * CFG_INPUT) {
        fprintf(stderr, "short read %s\n", path); free(raw); fclose(fi); return -1;
    }
    fclose(fi);

    snprintf(path, sizeof path, "%s/%s-labels-idx1-ubyte", dir, split);
    FILE *fl = fopen(path, "rb");
    if (!fl) { fprintf(stderr, "cannot open %s\n", path); free(raw); return -1; }
    if (read_be32(fl) != 2049 || (int)read_be32(fl) != n) {
        fprintf(stderr, "bad label file %s\n", path); free(raw); fclose(fl); return -1;
    }
    unsigned char *lab = (unsigned char *)malloc(n);
    if (fread(lab, 1, n, fl) != (size_t)n) { free(raw); free(lab); fclose(fl); return -1; }
    fclose(fl);

    out->n = n;
    out->images = (float *)malloc((size_t)n * CFG_INPUT * sizeof(float));
    out->labels = (int *)malloc((size_t)n * sizeof(int));
    for (size_t i = 0; i < (size_t)n * CFG_INPUT; i++) out->images[i] = raw[i] / 255.0f;
    for (int i = 0; i < n; i++) out->labels[i] = lab[i];
    free(raw); free(lab);
    return 0;
}

void mnist_free(Dataset *d) {
    free(d->images); free(d->labels);
    memset(d, 0, sizeof(*d));
}

uint32_t rng_next(uint32_t *s) {
    uint32_t x = *s ? *s : 0x9E3779B9u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *s = x;
}

float rng_uniform(uint32_t *s) { return (rng_next(s) >> 8) * (1.0f / 16777216.0f); }

void shuffle_indices(int *idx, int n, uint32_t *s) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_next(s) % (uint32_t)(i + 1));
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

void gather_batch(const Dataset *d, const int *idx, int start, int count, float *X, float *Y) {
    for (int r = 0; r < count; r++) {
        int s = idx[start + r];
        memcpy(X + (size_t)r * CFG_INPUT, d->images + (size_t)s * CFG_INPUT, CFG_INPUT * sizeof(float));
        for (int c = 0; c < CFG_CLASSES; c++) Y[r * CFG_CLASSES + c] = (c == d->labels[s]) ? 1.0f : 0.0f;
    }
}
