#include "popgym_position_only_cartpole.h"
#include "raylib.h"

#ifndef POC_ALIAS_MAX_EPISODE_LENGTH
#define POC_ALIAS_MAX_EPISODE_LENGTH 200
#endif

int main(void) {
    PositionOnlyCartPole env = {0};
    env.num_agents = 1;
    env.max_episode_length = POC_ALIAS_MAX_EPISODE_LENGTH;
    env.rng = 42;

    init(&env);
    env.observations = (float*)calloc(CARTPOLE_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    while (!WindowShouldClose()) {
        c_render(&env);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsKeyDown(KEY_SPACE)) {
            env.actions[0] = (float)(rand_r(&env.rng) % 2);
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
