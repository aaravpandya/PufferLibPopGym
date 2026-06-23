// A simple C env for the POPGym RepeatPrevious semantic task.
// The agent sees a stream of suits [0..3] and predicts the suit from k steps ago.

#include <stdlib.h>
#include <assert.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define NUM_SUITS 4
#define DECK_SIZE 52

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float accuracy;
    float invalid_action_rate;
    float n; // number of episodes
} Log;

typedef struct {
    int tick;
    int num_cards;
    int episode_length;
    int k;
    int k_queries;
    int correct_predictions;
    int invalid_actions;
    unsigned char current_suit;
    unsigned char previous_action;
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

    // Env controls
    int num_decks;
    int k;
    int include_prev_action;
    int include_antialias;

    unsigned int rng;
} RepeatPrevious;

static inline int rp_has_query(const State* s) {
    return (s->k == 0) || (s->tick + 1 >= s->k);
}

void add_log(RepeatPrevious* env) {
    env->log.score += env->state.episode_return;
    env->log.episode_return += env->state.episode_return;
    env->log.episode_length += (float)env->state.episode_length;

    float denominator = (float)(env->state.k_queries > 0 ? env->state.k_queries : 1);
    env->log.accuracy += env->state.correct_predictions / denominator;
    env->log.invalid_action_rate += (float)env->state.invalid_actions
        / (float)(env->state.episode_length > 0 ? env->state.episode_length : 1);
    env->log.n += 1.0f;
}

void refresh_observations(RepeatPrevious* env) {
    State* s = &env->state;
    env->observations[0] = env->state.current_suit;                  // Current suit
    env->observations[1] = (unsigned char)rp_has_query(s);           // Query available?
    env->observations[2] = env->include_prev_action
        ? env->state.previous_action
        : (unsigned char)0;                                          // previous action proxy
    env->observations[3] = env->include_antialias
        ? (unsigned char)(s->tick == 0)
        : (unsigned char)0;                                          // reset/t0 flag
}

void init(RepeatPrevious* env) {
    assert(env->num_decks >= 1);
    assert(env->k >= 0);

    env->state.num_cards = env->num_decks * DECK_SIZE;
    assert(env->state.num_cards > env->k);

    env->state.episode_length = env->state.num_cards - 1;
    env->state.cards = (unsigned char*)realloc(
        env->state.cards,
        (size_t)env->state.num_cards * sizeof(unsigned char)
    );
    assert(env->state.cards != NULL);
}

void c_reset(RepeatPrevious* env) {
    State* s = &env->state;
    s->tick = 0;
    s->k = env->k;
    s->k_queries = 0;
    s->correct_predictions = 0;
    s->invalid_actions = 0;
    s->previous_action = 0;
    s->episode_return = 0.0f;

    for (int i = 0; i < s->num_cards; i++) {
        s->cards[i] = (unsigned char)(i % NUM_SUITS);
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

void c_step(RepeatPrevious* env) {
    State* s = &env->state;
    int action = (int)env->actions[0];
    if (action < 0 || action >= NUM_SUITS) {
        s->invalid_actions += 1;
        action = 0;
    }

    float reward = 0.0f;
    if (rp_has_query(s)) {
        int query_idx = (s->k == 0) ? s->tick : s->tick + 1 - s->k;
        int target = (int)s->cards[query_idx];
        float reward_scale = 1.0f / (float)(s->num_cards - s->k);
        reward = (action == target) ? reward_scale : -reward_scale;
        s->correct_predictions += (action == target);
        s->k_queries += 1;
    }

    s->previous_action = (unsigned char)(action + 1);
    env->rewards[0] = reward;
    env->terminals[0] = 0.0f;
    s->episode_return += reward;

    s->tick += 1;
    if (s->tick < s->episode_length) {
        s->current_suit = s->cards[s->tick];
        refresh_observations(env);
        return;
    }

    env->terminals[0] = 1.0f;
    add_log(env);
    c_reset(env);
    refresh_observations(env);
}

void c_render(RepeatPrevious* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(420, 220, "PufferLib RepeatPrevious");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    const char* query_ready = env->state.tick >= env->state.k ? "ready" : "not ready";
    DrawText(TextFormat("tick: %d/%d | suit: %d | query: %s", env->state.tick, env->state.episode_length, env->state.current_suit, query_ready), 10, 10, 18, (Color){241,241,241,241});
    DrawText(TextFormat("k=%d  reward=%.2f", env->state.k, env->rewards[0]), 10, 42, 18, (Color){241,241,241,241});
    EndDrawing();
#endif
}

void c_close(RepeatPrevious* env) {
    free(env->state.cards);
    env->state.cards = NULL;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
