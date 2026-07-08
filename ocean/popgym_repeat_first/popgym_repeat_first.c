#include "popgym_repeat_first.h"
#include "raylib.h"

int main(void) {
    RepeatFirst env = {0};
    env.num_agents = 1;
    env.num_decks = 1;
    env.rng = 42;

    init(&env);
    env.observations = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    while (!WindowShouldClose()) {
        c_render(&env);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsKeyDown(KEY_SPACE)) {
            env.actions[0] = (float)(rand_r(&env.rng) % RF_NUM_SUITS);
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
