// Native POPGym RepeatFirst semantics.
// The agent must repeat the first observed suit for the rest of the deck.

#pragma once

#include <stdlib.h>
#include "../popgym_check.h"
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define RF_NUM_SUITS 4
#define RF_DECK_SIZE 52

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
    int invalid_actions;
    unsigned char first_suit;
    unsigned char current_suit;
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
} RepeatFirst;

void add_log(RepeatFirst* env) {
    State* s = &env->state;
    float denom = (float)(s->episode_length > 0 ? s->episode_length : 1);

    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->episode_length;
    env->log.accuracy += (float)s->correct_predictions / denom;
    env->log.invalid_action_rate += (float)s->invalid_actions / denom;
    env->log.n += 1.0f;
}

void refresh_observations(RepeatFirst* env) {
    env->observations[0] = env->state.current_suit;
}

void init(RepeatFirst* env) {
    POPGYM_CHECK(env->num_decks >= 1,
        "num_decks must be >= 1 (got %d)", env->num_decks);

    env->state.num_cards = env->num_decks * RF_DECK_SIZE;
    env->state.episode_length = env->state.num_cards - 1;
    env->cards = (unsigned char*)realloc(
        env->cards,
        (size_t)env->state.num_cards * sizeof(unsigned char)
    );
    POPGYM_CHECK(env->cards != NULL,
        "out of memory allocating %d cards", env->state.num_cards);
}

void c_reset(RepeatFirst* env) {
    State* s = &env->state;
    s->tick = 0;
    s->correct_predictions = 0;
    s->invalid_actions = 0;
    s->episode_return = 0.0f;

    for (int i = 0; i < s->num_cards; i++) {
        env->cards[i] = (unsigned char)(i % RF_NUM_SUITS);
    }
    for (int i = s->num_cards - 1; i > 0; i--) {
        int j = (int)(rand_r(&env->rng) % (unsigned int)(i + 1));
        unsigned char tmp = env->cards[i];
        env->cards[i] = env->cards[j];
        env->cards[j] = tmp;
    }

    s->first_suit = env->cards[0];
    s->current_suit = s->first_suit;
    refresh_observations(env);
}

void c_step(RepeatFirst* env) {
    State* s = &env->state;
    int action = (int)env->actions[0];
    if (action < 0 || action >= RF_NUM_SUITS) {
        s->invalid_actions += 1;
        action = 0;
    }

    float reward_scale = 1.0f / (float)s->episode_length;
    int correct = action == (int)s->first_suit;
    env->rewards[0] = correct ? reward_scale : -reward_scale;
    env->terminals[0] = 0.0f;
    s->correct_predictions += correct;
    s->episode_return += env->rewards[0];

    s->tick += 1;
    if (s->tick < s->episode_length) {
        s->current_suit = env->cards[s->tick];
        refresh_observations(env);
        return;
    }

    env->terminals[0] = 1.0f;
    add_log(env);
    c_reset(env);
}

void c_render(RepeatFirst* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(420, 220, "PufferLib RepeatFirst");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | suit: %d | first: %d",
        env->state.tick, env->state.episode_length,
        env->state.current_suit, env->state.first_suit),
        10, 10, 18, (Color){241, 241, 241, 241});
    DrawText(TextFormat("reward=%.3f", env->rewards[0]),
        10, 42, 18, (Color){241, 241, 241, 241});
    EndDrawing();
#endif
}

void c_close(RepeatFirst* env) {
    free(env->cards);
    env->cards = NULL;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
