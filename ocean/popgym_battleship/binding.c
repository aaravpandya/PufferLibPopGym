#include "popgym_battleship.h"

#define OBS_SIZE 1
#define NUM_ATNS 2
#define ACT_SIZES {BS_BOARD_SIZE, BS_BOARD_SIZE}
#define OBS_TENSOR_T ByteTensor

#define Env Battleship
static inline void puffer_state_refresh(Env* env) { refresh_observations(env); }
#include "vecenv.h"
#include "../popgym_kwargs.h"

void my_init(Env* env, Dict* kwargs) {
    popgym_ignore_kwargs(kwargs);
    env->num_agents = 1;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "success", log->success);
    dict_set(out, "invalid_action_rate", log->invalid_action_rate);
}
