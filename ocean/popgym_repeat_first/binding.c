#include "popgym_repeat_first.h"

#define OBS_SIZE 1
#define NUM_ATNS 1
#define ACT_SIZES {RF_NUM_SUITS}
#define OBS_TENSOR_T ByteTensor

#define Env RepeatFirst
static inline void puffer_state_refresh(Env* env) { refresh_observations(env); }
#include "vecenv.h"
#include "../popgym_kwargs.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
#ifdef RF_ALIAS_NUM_DECKS
    popgym_ignore_kwargs(kwargs);
    env->num_decks = RF_ALIAS_NUM_DECKS;
#else
    static const char* const known_kwargs[] = {"num_decks"};
    popgym_require_known_kwargs(kwargs, known_kwargs, 1);
    env->num_decks = (int)kwarg_or(kwargs, "num_decks", 1);
#endif
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "accuracy", log->accuracy);
    dict_set(out, "invalid_action_rate", log->invalid_action_rate);
}
