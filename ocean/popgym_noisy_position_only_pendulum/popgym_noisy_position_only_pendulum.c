#include "popgym_noisy_position_only_pendulum.h"
#include "raylib.h"

#ifndef NPOP_ALIAS_MAX_EPISODE_LENGTH
#define NPOP_ALIAS_MAX_EPISODE_LENGTH 200
#endif
#ifndef NPOP_ALIAS_NOISE_SIGMA
#define NPOP_ALIAS_NOISE_SIGMA 0.1f
#endif

int main(void) {
    NoisyPositionOnlyPendulum env = {0};
    env.num_agents = 1;
    env.max_episode_length = NPOP_ALIAS_MAX_EPISODE_LENGTH;
    env.noise_sigma = NPOP_ALIAS_NOISE_SIGMA;
    env.rng = 42;

    init(&env);
    env.observations = (float*)calloc(PENDULUM_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    while (!WindowShouldClose()) {
        c_render(&env);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsKeyDown(KEY_SPACE)) {
            env.actions[0] = popgym_randf(&env.rng, -PENDULUM_MAX_TORQUE, PENDULUM_MAX_TORQUE);
            c_step(&env);
        }
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
