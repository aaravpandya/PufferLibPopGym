// Env-kwarg helpers for POPGym bindings. Include after vecenv.h so the Dict
// types are available.

#pragma once

#include <string.h>
#include "popgym_check.h"

// Read an optional env kwarg, falling back to the base POPGym default instead
// of crashing on dict_get's NULL when the key is missing.
static inline double kwarg_or(Dict* kwargs, const char* key, double fallback) {
    DictItem* item = kwargs != NULL ? dict_get_unsafe(kwargs, key) : NULL;
    return item != NULL ? item->value : fallback;
}

// Missing kwargs fall back to defaults, so a misspelled key must fail loudly
// or it silently reconfigures the env to its default.
static inline void popgym_require_known_kwargs(
        Dict* kwargs, const char* const* known, int num_known) {
    if (kwargs == NULL) {
        return;
    }
    for (int i = 0; i < kwargs->size; i++) {
        int recognized = 0;
        for (int j = 0; j < num_known; j++) {
            recognized = recognized || strcmp(kwargs->items[i].key, known[j]) == 0;
        }
        POPGYM_CHECK(recognized, "unknown env kwarg: %s", kwargs->items[i].key);
    }
}

// Difficulty-variant envs take no env kwargs: their difficulty is fixed at
// compile time. Warn (once, not per env instance) instead of silently
// discarding anything the caller sent.
static inline void popgym_ignore_kwargs(Dict* kwargs) {
    static int warned = 0;
    if (kwargs != NULL && kwargs->size > 0 && !warned) {
        warned = 1;
        fprintf(stderr,
            "popgym: env kwargs are ignored for this env; "
            "difficulty is fixed at compile time\n");
    }
}
