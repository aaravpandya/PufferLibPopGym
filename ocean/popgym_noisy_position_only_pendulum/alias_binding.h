#include "popgym_noisy_position_only_pendulum.h"

#define OBS_SIZE NPOP_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {1}
#define OBS_TENSOR_T FloatTensor
#define PUFFER_HAS_STATE 1
#define PUFFER_STATE_REFRESH(env) refresh_observations(env)

#define Env NoisyPositionOnlyPendulum
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    env->max_episode_length = NPOP_ALIAS_MAX_EPISODE_LENGTH;
    env->noise_sigma = NPOP_ALIAS_NOISE_SIGMA;
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
}
