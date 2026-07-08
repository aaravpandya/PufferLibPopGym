// Native POPGym HigherLower semantics.
// Action 0 guesses the next rank is higher; action 1 guesses lower.

#pragma once

#include <stdlib.h>
#include "../popgym_check.h"
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define HL_NUM_RANKS 13
#define HL_SUITS_PER_RANK 4
#define HL_DECK_SIZE 52

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float accuracy;
    float invalid_action_rate;
    float n;
} Log;

// State is snapshotted/restored by struct assignment in the GPU state
// curriculum, so it must stay copyable: the owned card buffer lives on the
// env, not in State.
typedef struct {
    int tick;
    int num_cards;
    int episode_length;
    int correct_predictions;
    int non_tie_predictions;
    int invalid_actions;
    unsigned char current_rank;
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
    unsigned char* cards;
    int num_decks;
    unsigned int rng;
} HigherLower;

void refresh_observations(HigherLower* env) {
    env->observations[0] = env->state.current_rank;
}

void add_log(HigherLower* env) {
    State* s = &env->state;
    float denom = (float)(s->non_tie_predictions > 0 ? s->non_tie_predictions : 1);

    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->episode_length;
    env->log.accuracy += (float)s->correct_predictions / denom;
    env->log.invalid_action_rate += (float)s->invalid_actions
        / (float)(s->episode_length > 0 ? s->episode_length : 1);
    env->log.n += 1.0f;
}

void init(HigherLower* env) {
    POPGYM_CHECK(env->num_decks >= 1,
        "num_decks must be >= 1 (got %d)", env->num_decks);

    env->state.num_cards = env->num_decks * HL_DECK_SIZE;
    env->state.episode_length = env->state.num_cards - 1;
    env->cards = (unsigned char*)realloc(
        env->cards,
        (size_t)env->state.num_cards * sizeof(unsigned char)
    );
    POPGYM_CHECK(env->cards != NULL,
        "out of memory allocating %d cards", env->state.num_cards);
}

void c_reset(HigherLower* env) {
    State* s = &env->state;
    s->tick = 0;
    s->correct_predictions = 0;
    s->non_tie_predictions = 0;
    s->invalid_actions = 0;
    s->episode_return = 0.0f;

    for (int i = 0; i < s->num_cards; i++) {
        env->cards[i] = (unsigned char)((i / HL_SUITS_PER_RANK) % HL_NUM_RANKS);
    }
    for (int i = s->num_cards - 1; i > 0; i--) {
        int j = (int)(rand_r(&env->rng) % (unsigned int)(i + 1));
        unsigned char tmp = env->cards[i];
        env->cards[i] = env->cards[j];
        env->cards[j] = tmp;
    }

    s->current_rank = env->cards[0];
    refresh_observations(env);
}

void c_step(HigherLower* env) {
    State* s = &env->state;
    int action = (int)env->actions[0];
    if (action < 0 || action > 1) {
        s->invalid_actions += 1;
        action = 0;
    }

    unsigned char next_rank = env->cards[s->tick + 1];
    float reward = 0.0f;
    if (next_rank != s->current_rank) {
        int correct = (next_rank > s->current_rank) == (action == 0);
        reward = correct ? 1.0f / (float)s->num_cards : -1.0f / (float)s->num_cards;
        s->correct_predictions += correct;
        s->non_tie_predictions += 1;
    }

    env->rewards[0] = reward;
    env->terminals[0] = s->tick >= s->episode_length - 1 ? 1.0f : 0.0f;
    s->episode_return += reward;
    s->tick += 1;

    if (env->terminals[0]) {
        add_log(env);
        c_reset(env);
        return;
    }

    s->current_rank = next_rank;
    refresh_observations(env);
}

void c_render(HigherLower* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(420, 220, "PufferLib HigherLower");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | rank: %d | reward: %.3f",
        env->state.tick, env->state.episode_length,
        env->state.current_rank, env->rewards[0]),
        10, 10, 18, (Color){241, 241, 241, 241});
    EndDrawing();
#endif
}

void c_close(HigherLower* env) {
    free(env->cards);
    env->cards = NULL;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
