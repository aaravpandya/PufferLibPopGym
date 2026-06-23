#include "popgym_minesweeper.h"

#define OBS_SIZE 1
#define NUM_ATNS 2
#define ACT_SIZES {MS_ROWS, MS_COLS}
#define OBS_TENSOR_T ByteTensor
#define PUFFER_HAS_STATE 1
#define PUFFER_STATE_REFRESH(env) refresh_observations(env, env->observations[0])

#define Env PopGymMineSweeper
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
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
