#include "popgym_position_only_pendulum.h"
#include "raylib.h"

int main(void) {
    PositionOnlyPendulum env = {0};
    env.num_agents = 1;
    env.max_episode_length = 200;
    env.rng = 42;

    init(&env);
    env.observations = (float*)calloc(POP_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    while (!WindowShouldClose()) {
        c_render(&env);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsKeyDown(KEY_SPACE)) {
            env.actions[0] = pop_randf(&env, -POP_MAX_TORQUE, POP_MAX_TORQUE);
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
