// Native POPGym MultiarmedBandit default/Easy semantics.

#include <assert.h>
#include <stdlib.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#ifndef MB_NUM_BANDITS
#define MB_NUM_BANDITS 10
#endif
#ifndef MB_EPISODE_LENGTH
#define MB_EPISODE_LENGTH 200
#endif

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float invalid_action_rate;
    float n;
} Log;

typedef struct {
    int tick;
    int invalid_actions;
    unsigned char obs;
    float bandits[MB_NUM_BANDITS];
    float episode_return;
} State;

typedef struct {
    Log log;
    unsigned char* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    State state;
    unsigned int rng;
} MultiarmedBandit;

static inline float mb_random_float(unsigned int* rng) {
    return (float)rand_r(rng) / ((float)RAND_MAX + 1.0f);
}

void refresh_observations(MultiarmedBandit* env) {
    env->observations[0] = env->state.obs;
}

void add_log(MultiarmedBandit* env) {
    State* s = &env->state;
    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.invalid_action_rate += (float)s->invalid_actions / (float)MB_EPISODE_LENGTH;
    env->log.n += 1.0f;
}

void init(MultiarmedBandit* env) {
    (void)env;
}

void c_reset(MultiarmedBandit* env) {
    State* s = &env->state;
    s->tick = 0;
    s->invalid_actions = 0;
    s->obs = 0;
    s->episode_return = 0.0f;
    for (int i = 0; i < MB_NUM_BANDITS; i++) {
        s->bandits[i] = mb_random_float(&env->rng);
    }
    refresh_observations(env);
}

void c_step(MultiarmedBandit* env) {
    State* s = &env->state;
    int action = (int)env->actions[0];
    if (action < 0 || action >= MB_NUM_BANDITS) {
        s->invalid_actions += 1;
        action = 0;
    }

    s->obs = mb_random_float(&env->rng) < s->bandits[action] ? 1 : 0;
    float reward_scale = 1.0f / (float)MB_EPISODE_LENGTH;
    env->rewards[0] = s->obs ? reward_scale : -reward_scale;
    s->episode_return += env->rewards[0];

    env->terminals[0] = s->tick >= MB_EPISODE_LENGTH - 1 ? 1.0f : 0.0f;
    s->tick += 1;

    if (env->terminals[0]) {
        add_log(env);
        c_reset(env);
        return;
    }

    refresh_observations(env);
}

void c_render(MultiarmedBandit* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(460, 220, "PufferLib MultiarmedBandit");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | obs: %d | reward: %.3f",
        env->state.tick, MB_EPISODE_LENGTH, env->state.obs, env->rewards[0]),
        10, 10, 18, (Color){241, 241, 241, 241});
    for (int i = 0; i < MB_NUM_BANDITS; i++) {
        DrawText(TextFormat("%d: %.2f", i, env->state.bandits[i]),
            10 + (i % 5) * 82, 52 + (i / 5) * 28, 18, RAYWHITE);
    }
    EndDrawing();
#endif
}

void c_close(MultiarmedBandit* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
