#include "popgym_noisy_position_only_pendulum.h"

#define OBS_SIZE PENDULUM_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {1}
#define OBS_TENSOR_T FloatTensor

#define Env NoisyPositionOnlyPendulum
static inline void puffer_state_refresh(Env* env) { refresh_observations(env); }
#include "vecenv.h"
#include "../popgym_kwargs.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
#ifdef NPOP_ALIAS_MAX_EPISODE_LENGTH
    popgym_ignore_kwargs(kwargs);
    env->max_episode_length = NPOP_ALIAS_MAX_EPISODE_LENGTH;
    env->noise_sigma = NPOP_ALIAS_NOISE_SIGMA;
#else
    static const char* const known_kwargs[] = {"max_episode_length", "noise_sigma"};
    popgym_require_known_kwargs(kwargs, known_kwargs, 2);
    env->max_episode_length = (int)kwarg_or(kwargs, "max_episode_length", 200);
    env->noise_sigma = (float)kwarg_or(kwargs, "noise_sigma", 0.1);
#endif
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
}
