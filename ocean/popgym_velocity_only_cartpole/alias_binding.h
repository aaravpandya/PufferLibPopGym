#include "popgym_velocity_only_cartpole.h"

#define OBS_SIZE VOC_OBS_SIZE
#define NUM_ATNS 1
#define ACT_SIZES {2}
#define OBS_TENSOR_T FloatTensor
#define PUFFER_HAS_STATE 1
#define PUFFER_STATE_REFRESH(env) refresh_observations(env)

#define Env VelocityOnlyCartPole
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    env->max_episode_length = VOC_ALIAS_MAX_EPISODE_LENGTH;
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
