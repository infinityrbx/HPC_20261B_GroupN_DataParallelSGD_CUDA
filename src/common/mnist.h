#ifndef MNIST_H
#define MNIST_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int    n;        /* number of samples                     */
    float *images;   /* n x 784, row-major, scaled to [0,1]  */
    int   *labels;   /* n, values 0..9                        */
} Dataset;

/* Load MNIST idx files (uncompressed). split = "train" or "t10k".
 * Returns 0 on success. */
int  mnist_load(const char *dir, const char *split, Dataset *out);
void mnist_free(Dataset *d);

/* Deterministic RNG (xorshift32) shared by all code so results are reproducible. */
uint32_t rng_next(uint32_t *state);
float    rng_uniform(uint32_t *state);                 /* [0,1)  */
void     shuffle_indices(int *idx, int n, uint32_t *state);

/* Copy samples idx[start..start+count) into contiguous X (count x 784) and
 * one-hot Y (count x 10). */
void gather_batch(const Dataset *d, const int *idx, int start, int count,
                  float *X, float *Y);

#ifdef __cplusplus
}
#endif
#endif
