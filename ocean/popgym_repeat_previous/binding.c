#include "popgym_repeat_previous.h"

#define OBS_SIZE 4
#define NUM_ATNS 1
#define ACT_SIZES {NUM_SUITS}
#define OBS_TENSOR_T ByteTensor

#define Env RepeatPrevious
static inline void puffer_state_refresh(Env* env) { refresh_observations(env); }
#include "vecenv.h"
#include "../popgym_kwargs.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
#ifdef RP_ALIAS_NUM_DECKS
    popgym_ignore_kwargs(kwargs);
    env->num_decks = RP_ALIAS_NUM_DECKS;
    env->k = RP_ALIAS_K;
    env->include_prev_action = 0;
    env->include_antialias = 0;
#else
    static const char* const known_kwargs[] = {"num_decks", "k", "include_prev_action", "include_antialias"};
    popgym_require_known_kwargs(kwargs, known_kwargs, 4);
    env->num_decks = (int)kwarg_or(kwargs, "num_decks", 1);
    env->k = (int)kwarg_or(kwargs, "k", 4);
    env->include_prev_action = (int)kwarg_or(kwargs, "include_prev_action", 0);
    env->include_antialias = (int)kwarg_or(kwargs, "include_antialias", 0);
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
