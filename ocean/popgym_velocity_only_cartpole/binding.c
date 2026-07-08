#include "popgym_velocity_only_cartpole.h"

#define OBS_SIZE CARTPOLE_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor

#define Env VelocityOnlyCartPole
static inline void puffer_state_refresh(Env* env) { refresh_observations(env); }
#include "vecenv.h"
#include "../popgym_kwargs.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
#ifdef VOC_ALIAS_MAX_EPISODE_LENGTH
    popgym_ignore_kwargs(kwargs);
    env->max_episode_length = VOC_ALIAS_MAX_EPISODE_LENGTH;
#else
    static const char* const known_kwargs[] = {"max_episode_length"};
    popgym_require_known_kwargs(kwargs, known_kwargs, 1);
    env->max_episode_length = (int)kwarg_or(kwargs, "max_episode_length", 200);
#endif
    init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "x_threshold_termination", log->x_threshold_termination);
    dict_set(out, "pole_angle_termination", log->pole_angle_termination);
    dict_set(out, "max_steps_termination", log->max_steps_termination);
}
