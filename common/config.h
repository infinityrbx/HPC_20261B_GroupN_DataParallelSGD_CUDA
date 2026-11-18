/*
 * config.h - FIXED experiment settings shared by every method.
 *
 * Do NOT change these per method. Every number in the final comparison must
 * come from these defaults (CLI flags exist only for exploring/debugging).
 */
#ifndef CONFIG_H
#define CONFIG_H
#include <stddef.h>

#define CFG_SEED          42u     /* RNG seed: weight init + shuffling          */
#define CFG_GLOBAL_BATCH  256     /* global mini-batch, split evenly over k GPUs */
#define CFG_LR            0.1f    /* SGD learning rate                           */
#define CFG_HIDDEN        512     /* hidden units H (scale up to grow p)         */
#define CFG_EPOCHS        10      /* training epochs                             */

#define CFG_INPUT         784     /* 28x28 MNIST images                          */
#define CFG_CLASSES       10

#define CFG_BENCH_WARMUP  5       /* allreduce benchmark: untimed warm-up runs   */
#define CFG_BENCH_REPEATS 20      /* allreduce benchmark: timed runs (median)    */
#define CFG_CHECK_TOL     1e-5f   /* correctness tolerance (relative)            */

#define CFG_DATA_DIR      "data"  /* where scripts/get_mnist.sh puts the files   */

/* Parameter layout (flat buffer, identical on CPU and every GPU):
 *   [ W1 (INPUT x H) | b1 (H) | W2 (H x CLASSES) | b2 (CLASSES) ]
 * Gradients use the same layout, so allreduce works on ONE contiguous buffer. */
static inline size_t cfg_num_params(int H) {
    return (size_t)CFG_INPUT * H + H + (size_t)H * CFG_CLASSES + CFG_CLASSES;
}

#endif
