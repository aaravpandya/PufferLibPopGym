#include "popgym_repeat_previous.h"
#include "raylib.h"

int main(void) {
    RepeatPrevious env = {0};
    env.num_agents = 1;
    env.num_decks = 1;
    env.k = 4;
    env.include_prev_action = 1;
    env.include_antialias = 1;
    env.rng = 42;

    init(&env);
    env.observations = (unsigned char*)calloc(OBS_SIZE, sizeof(unsigned char));
    env.actions = (float*)calloc(1, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (float*)calloc(1, sizeof(float));

    c_reset(&env);
    c_render(&env);

    while (!WindowShouldClose()) {
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsKeyDown(KEY_SPACE)) {
            env.actions[0] = (float)(rand_r(&env.rng) % NUM_SUITS);
            c_step(&env);
            c_render(&env);
        }
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    c_close(&env);
    return 0;
}
