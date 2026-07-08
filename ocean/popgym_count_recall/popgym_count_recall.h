// Native POPGym CountRecall default/Easy colors semantics.

#include <assert.h>
#include <stdlib.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#ifndef CR_NUM_VALUES
#define CR_NUM_VALUES 2
#endif
#ifndef CR_DECK_SIZE
#define CR_DECK_SIZE 52
#endif
#ifndef CR_CARD_VALUE
#define CR_CARD_VALUE(i) ((i) % CR_NUM_VALUES)
#endif
#define CR_MAX_COUNT (CR_DECK_SIZE / CR_NUM_VALUES)
#define CR_NUM_ACTIONS (CR_MAX_COUNT + 1)
#define CR_EPISODE_LENGTH (CR_DECK_SIZE - 1)

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
    int correct_predictions;
    int invalid_actions;
    unsigned char value;
    unsigned char query;
    unsigned char value_cards[CR_DECK_SIZE];
    unsigned char query_cards[CR_DECK_SIZE];
    int counts[CR_NUM_VALUES];
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
} CountRecall;

void refresh_observations(CountRecall* env) {
    env->observations[0] = env->state.value;
    env->observations[1] = env->state.query;
}

void add_log(CountRecall* env) {
    State* s = &env->state;
    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)CR_EPISODE_LENGTH;
    env->log.accuracy += (float)s->correct_predictions / (float)CR_EPISODE_LENGTH;
    env->log.invalid_action_rate += (float)s->invalid_actions / (float)CR_EPISODE_LENGTH;
    env->log.n += 1.0f;
}

static inline void cr_shuffle(unsigned char* cards, unsigned int* rng) {
    for (int i = CR_DECK_SIZE - 1; i > 0; i--) {
        int j = (int)(rand_r(rng) % (unsigned int)(i + 1));
        unsigned char tmp = cards[i];
        cards[i] = cards[j];
        cards[j] = tmp;
    }
}

void init(CountRecall* env) {
    (void)env;
}

void c_reset(CountRecall* env) {
    State* s = &env->state;
    s->tick = 0;
    s->correct_predictions = 0;
    s->invalid_actions = 0;
    s->episode_return = 0.0f;
    for (int i = 0; i < CR_NUM_VALUES; i++) {
        s->counts[i] = 0;
    }

    for (int i = 0; i < CR_DECK_SIZE; i++) {
        s->value_cards[i] = (unsigned char)CR_CARD_VALUE(i);
        s->query_cards[i] = (unsigned char)CR_CARD_VALUE(i);
    }
    cr_shuffle(s->value_cards, &env->rng);
    cr_shuffle(s->query_cards, &env->rng);

    s->value = s->value_cards[0];
    s->query = s->query_cards[0];
    s->counts[s->value] += 1;
    refresh_observations(env);
}

void c_step(CountRecall* env) {
    State* s = &env->state;
    int action = (int)env->actions[0];
    if (action < 0 || action >= CR_NUM_ACTIONS) {
        s->invalid_actions += 1;
        action = 0;
    }

    int prev_count = s->counts[s->query];
    int correct = action == prev_count;
    float reward_scale = 1.0f / (float)CR_EPISODE_LENGTH;
    env->rewards[0] = correct ? reward_scale : -reward_scale;
    env->terminals[0] = s->tick >= CR_EPISODE_LENGTH - 1 ? 1.0f : 0.0f;
    s->correct_predictions += correct;
    s->episode_return += env->rewards[0];

    s->tick += 1;
    s->value = s->value_cards[s->tick];
    s->query = s->query_cards[s->tick];
    s->counts[s->value] += 1;

    if (env->terminals[0]) {
        add_log(env);
        c_reset(env);
        return;
    }

    refresh_observations(env);
}

void c_render(CountRecall* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(440, 220, "PufferLib CountRecall");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | value: %d | query: %d",
        env->state.tick, CR_EPISODE_LENGTH, env->state.value, env->state.query),
        10, 10, 18, (Color){241, 241, 241, 241});
    DrawText(TextFormat("query count: %d | reward=%.3f",
        env->state.counts[env->state.query], env->rewards[0]),
        10, 42, 18, (Color){241, 241, 241, 241});
    EndDrawing();
#endif
}

void c_close(CountRecall* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
