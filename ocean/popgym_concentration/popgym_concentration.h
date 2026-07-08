// Native POPGym Concentration default/Hard ranks semantics.

#pragma once

#include <assert.h>
#include <stdlib.h>
#include "../popgym_check.h"
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#ifndef CONC_NUM_CARDS
#define CONC_NUM_CARDS 52
#endif
#ifndef CONC_NUM_VALUES
#define CONC_NUM_VALUES 13
#endif
#ifndef CONC_FACEDOWN
#define CONC_FACEDOWN CONC_NUM_VALUES
#endif
#ifndef CONC_EPISODE_LENGTH
#define CONC_EPISODE_LENGTH 104
#endif
#ifndef CONC_CARD_VALUE
#define CONC_CARD_VALUE(i) ((i) / 4)
#endif
#define CONC_SUCCESS_REWARD (1.0f / (float)(CONC_NUM_CARDS / 2))
#define CONC_FAILURE_REWARD (-1.0f / (float)CONC_EPISODE_LENGTH)

static_assert(CONC_NUM_CARDS >= 2 && CONC_NUM_CARDS % 2 == 0,
    "cards must come in matchable pairs");
static_assert(CONC_EPISODE_LENGTH >= 1, "episode length must be positive");

// Render layout: 13 cards per row; the window must be tall enough for the
// bottom row of the largest (medium, 104-card) board.
#define CONC_RENDER_ROWS ((CONC_NUM_CARDS + 12) / 13)
#define CONC_RENDER_HEIGHT_RAW (48 + CONC_RENDER_ROWS * 36 + 12)
#define CONC_RENDER_HEIGHT (CONC_RENDER_HEIGHT_RAW > 280 ? CONC_RENDER_HEIGHT_RAW : 280)

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float success;
    float invalid_action_rate;
    float n;
} Log;

typedef struct {
    int tick;
    int face_up_count;
    int in_play_count;
    int in_play_idx[2];
    int invalid_actions;
    unsigned char ranks[CONC_NUM_CARDS];
    unsigned char face_up[CONC_NUM_CARDS];
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
} Concentration;

void refresh_observations(Concentration* env) {
    State* s = &env->state;
    for (int i = 0; i < CONC_NUM_CARDS; i++) {
        env->observations[i] = s->face_up[i] ? s->ranks[i] : CONC_FACEDOWN;
    }
    for (int i = 0; i < s->in_play_count; i++) {
        int idx = s->in_play_idx[i];
        env->observations[idx] = s->ranks[idx];
    }
}

void add_log(Concentration* env, int success) {
    State* s = &env->state;
    env->log.score += success ? 1.0f : 0.0f;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.success += success ? 1.0f : 0.0f;
    env->log.invalid_action_rate += (float)s->invalid_actions / (float)CONC_EPISODE_LENGTH;
    env->log.n += 1.0f;
}

void init(Concentration* env) {
    (void)env;
}

void c_reset(Concentration* env) {
    State* s = &env->state;
    s->tick = 0;
    s->face_up_count = 0;
    s->in_play_count = 0;
    s->invalid_actions = 0;
    s->episode_return = 0.0f;

    for (int i = 0; i < CONC_NUM_CARDS; i++) {
        s->ranks[i] = (unsigned char)CONC_CARD_VALUE(i);
        s->face_up[i] = 0;
    }
    for (int i = CONC_NUM_CARDS - 1; i > 0; i--) {
        int j = (int)(rand_r(&env->rng) % (unsigned int)(i + 1));
        unsigned char tmp = s->ranks[i];
        s->ranks[i] = s->ranks[j];
        s->ranks[j] = tmp;
    }

    refresh_observations(env);
}

void c_step(Concentration* env) {
    State* s = &env->state;
    int action = (int)env->actions[0];
    if (action < 0 || action >= CONC_NUM_CARDS) {
        s->invalid_actions += 1;
        action = 0;
    }

    env->rewards[0] = 0.0f;
    env->terminals[0] = s->tick >= CONC_EPISODE_LENGTH - 1 ? 1.0f : 0.0f;

    POPGYM_CHECK(s->in_play_count < 2,
        "in_play_count invariant violated (%d)", s->in_play_count);
    s->in_play_idx[s->in_play_count] = action;
    s->in_play_count += 1;
    refresh_observations(env);

    int trying_face_up = 0;
    for (int i = 0; i < s->in_play_count; i++) {
        trying_face_up = trying_face_up || s->face_up[s->in_play_idx[i]];
    }

    if (trying_face_up) {
        env->rewards[0] = CONC_FAILURE_REWARD * (float)s->in_play_count;
        s->in_play_count = 0;
    } else if (s->in_play_count == 2) {
        int a = s->in_play_idx[0];
        int b = s->in_play_idx[1];
        if (a != b && s->ranks[a] == s->ranks[b]) {
            env->rewards[0] = CONC_SUCCESS_REWARD;
            s->face_up[a] = 1;
            s->face_up[b] = 1;
            s->face_up_count += 2;
            env->terminals[0] = s->face_up_count == CONC_NUM_CARDS ? 1.0f : env->terminals[0];
        } else {
            env->rewards[0] = 2.0f * CONC_FAILURE_REWARD;
        }
        s->in_play_count = 0;
    }

    s->episode_return += env->rewards[0];
    s->tick += 1;

    if (env->terminals[0]) {
        add_log(env, s->face_up_count == CONC_NUM_CARDS);
        c_reset(env);
    }
}

void c_render(Concentration* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(560, CONC_RENDER_HEIGHT, "PufferLib Concentration");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | face up: %d/%d | reward: %.3f",
        env->state.tick, CONC_EPISODE_LENGTH,
        env->state.face_up_count, CONC_NUM_CARDS, env->rewards[0]),
        10, 10, 18, (Color){241, 241, 241, 241});
    for (int i = 0; i < CONC_NUM_CARDS; i++) {
        int x = 10 + (i % 13) * 40;
        int y = 48 + (i / 13) * 36;
        unsigned char obs = env->observations[i];
        const char* text = obs == CONC_FACEDOWN ? "?" : TextFormat("%d", obs);
        DrawText(text, x, y, 20, RAYWHITE);
    }
    EndDrawing();
#endif
}

void c_close(Concentration* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
