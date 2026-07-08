// Native POPGym Autoencode semantics.
// Watch a shuffled suit sequence, then recite it in reverse.

#include <assert.h>
#include <stdlib.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define AE_NUM_SUITS 4
#define AE_DECK_SIZE 52
#define AE_MODE_PLAY 0
#define AE_MODE_WATCH 1

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float accuracy;
    float invalid_action_rate;
    float n;
} Log;

typedef struct {
    int tick;
    int num_cards;
    int mode;
    int shown_count;
    int remaining;
    int correct_predictions;
    int invalid_actions;
    unsigned char current_suit;
    unsigned char* cards;
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
    int num_decks;
    unsigned int rng;
} Autoencode;

void refresh_observations(Autoencode* env) {
    env->observations[0] = (unsigned char)env->state.mode;
    env->observations[1] = env->state.current_suit;
}

void add_log(Autoencode* env) {
    State* s = &env->state;
    float denom = (float)(s->num_cards > 0 ? s->num_cards : 1);

    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.accuracy += (float)s->correct_predictions / denom;
    env->log.invalid_action_rate += (float)s->invalid_actions / denom;
    env->log.n += 1.0f;
}

void init(Autoencode* env) {
    assert(env->num_decks >= 1);

    env->state.num_cards = env->num_decks * AE_DECK_SIZE;
    env->state.cards = (unsigned char*)realloc(
        env->state.cards,
        (size_t)env->state.num_cards * sizeof(unsigned char)
    );
    assert(env->state.cards != NULL);
}

void c_reset(Autoencode* env) {
    State* s = &env->state;
    s->tick = 0;
    s->mode = AE_MODE_WATCH;
    s->shown_count = 1;
    s->remaining = s->num_cards;
    s->correct_predictions = 0;
    s->invalid_actions = 0;
    s->episode_return = 0.0f;

    for (int i = 0; i < s->num_cards; i++) {
        s->cards[i] = (unsigned char)(i % AE_NUM_SUITS);
    }
    for (int i = s->num_cards - 1; i > 0; i--) {
        int j = (int)(rand_r(&env->rng) % (unsigned int)(i + 1));
        unsigned char tmp = s->cards[i];
        s->cards[i] = s->cards[j];
        s->cards[j] = tmp;
    }

    s->current_suit = s->cards[0];
    refresh_observations(env);
}

void c_step(Autoencode* env) {
    State* s = &env->state;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    if (s->mode == AE_MODE_WATCH) {
        s->current_suit = s->cards[s->shown_count];
        s->shown_count += 1;
        if (s->shown_count == s->num_cards) {
            s->mode = AE_MODE_PLAY;
        }
        s->tick += 1;
        refresh_observations(env);
        return;
    }

    int action = (int)env->actions[0];
    if (action < 0 || action >= AE_NUM_SUITS) {
        s->invalid_actions += 1;
        action = 0;
    }

    int target = (int)s->cards[s->remaining - 1];
    int correct = action == target;
    float reward_scale = 1.0f / (float)s->num_cards;
    env->rewards[0] = correct ? reward_scale : -reward_scale;
    env->terminals[0] = s->remaining == 1 ? 1.0f : 0.0f;
    s->correct_predictions += correct;
    s->episode_return += env->rewards[0];
    s->remaining -= 1;
    s->tick += 1;

    if (env->terminals[0]) {
        add_log(env);
        c_reset(env);
        return;
    }

    s->current_suit = 0;
    refresh_observations(env);
}

void c_render(Autoencode* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(460, 220, "PufferLib Autoencode");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d | mode: %s | suit: %d",
        env->state.tick,
        env->state.mode == AE_MODE_WATCH ? "watch" : "play",
        env->state.current_suit),
        10, 10, 18, (Color){241, 241, 241, 241});
    DrawText(TextFormat("shown=%d remaining=%d reward=%.3f",
        env->state.shown_count, env->state.remaining, env->rewards[0]),
        10, 42, 18, (Color){241, 241, 241, 241});
    EndDrawing();
#endif
}

void c_close(Autoencode* env) {
    free(env->state.cards);
    env->state.cards = NULL;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
