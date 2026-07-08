#include "popgym_concentration.h"
#include "raylib.h"

int main(void) {
    Concentration env = {0};
    env.num_agents = 1;
    env.rng = 42;

    init(&env);
    env.observations = (unsigned char*)calloc(CONC_NUM_CARDS, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    while (!WindowShouldClose()) {
        c_render(&env);
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsKeyDown(KEY_SPACE)) {
            env.actions[0] = (float)(rand_r(&env.rng) % CONC_NUM_CARDS);
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
