#include "popgym_higher_lower.h"
#include "raylib.h"

#ifndef HL_ALIAS_NUM_DECKS
#define HL_ALIAS_NUM_DECKS 1
#endif

int main(void) {
    HigherLower env = {0};
    env.num_agents = 1;
    env.num_decks = HL_ALIAS_NUM_DECKS;
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
