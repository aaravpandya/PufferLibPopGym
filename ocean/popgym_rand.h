// Shared RNG and clip helpers for the POPGym cartpole/pendulum cores. Both
// families must draw from identical formulas in the same order so that the
// noisy variants stay comparable (tests/test_popgym_noise_sigma_variants.py
// relies on the shared rand_r consumption order).

#pragma once

#include <math.h>
#include <stdlib.h>

static inline float popgym_randf(unsigned int* rng, float lo, float hi) {
    float t = (float)rand_r(rng) / ((float)RAND_MAX + 1.0f);
    return lo + t * (hi - lo);
}

static inline float popgym_unit_open(unsigned int* rng) {
    return ((float)rand_r(rng) + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static inline float popgym_normal(unsigned int* rng) {
    float u1 = popgym_unit_open(rng);
    float u2 = popgym_unit_open(rng);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static inline float popgym_clipf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}
