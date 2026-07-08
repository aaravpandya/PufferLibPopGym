#include "popgym_concentration_easy.h"

#define OBS_SIZE CONC_NUM_CARDS
#define NUM_ATNS 1
#define ACT_SIZES {CONC_NUM_CARDS}
#define OBS_TENSOR_T ByteTensor
#define PUFFER_HAS_STATE 1
#define PUFFER_STATE_REFRESH(env) refresh_observations(env)

#define Env Concentration
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
