// Runtime validation for POPGym env configuration and allocations.
// assert() is deleted by the -DNDEBUG release build, so checks that guard
// user-supplied config or allocation failures must go through POPGYM_CHECK.

#pragma once

#include <stdio.h>
#include <stdlib.h>

#define POPGYM_CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "popgym: "); \
            fprintf(stderr, __VA_ARGS__); \
            fprintf(stderr, "\n"); \
            abort(); \
        } \
    } while (0)
