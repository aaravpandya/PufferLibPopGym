// Native POPGym Battleship default/Medium semantics.

#include <assert.h>
#include <stdlib.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#ifndef BS_BOARD_SIZE
#define BS_BOARD_SIZE 10
#endif
#define BS_CELLS (BS_BOARD_SIZE * BS_BOARD_SIZE)
#define BS_NUM_SHIPS 4
#define BS_NEEDED_HITS 12
#define BS_HIT_REWARD (1.0f / (float)BS_NEEDED_HITS)
#define BS_MISS_REWARD (-1.0f / (float)(BS_CELLS - BS_NEEDED_HITS))

static const int BS_SHIP_SIZES[BS_NUM_SHIPS] = {2, 3, 3, 4};

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
    int hits;
    int invalid_actions;
    unsigned char obs;
    unsigned char board[BS_CELLS];
    unsigned char guesses[BS_CELLS];
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
} Battleship;

static inline int bs_index(int row, int col) {
    return row * BS_BOARD_SIZE + col;
}

void refresh_observations(Battleship* env) {
    env->observations[0] = env->state.obs;
}

void add_log(Battleship* env, int success) {
    State* s = &env->state;
    env->log.score += success ? 1.0f : 0.0f;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.success += success ? 1.0f : 0.0f;
    env->log.invalid_action_rate += (float)s->invalid_actions / (float)BS_CELLS;
    env->log.n += 1.0f;
}

int ship_fits(Battleship* env, int row, int col, int axis, int sign, int size) {
    State* s = &env->state;
    int end_row = row + (axis == 0 ? sign * (size - 1) : 0);
    int end_col = col + (axis == 1 ? sign * (size - 1) : 0);
    if (end_row < 0 || end_row >= BS_BOARD_SIZE || end_col < 0 || end_col >= BS_BOARD_SIZE) {
        return 0;
    }

    int min_row = row < end_row ? row : end_row;
    int max_row = row > end_row ? row : end_row;
    int min_col = col < end_col ? col : end_col;
    int max_col = col > end_col ? col : end_col;
    for (int r = min_row; r <= max_row; r++) {
        for (int c = min_col; c <= max_col; c++) {
            if (s->board[bs_index(r, c)]) {
                return 0;
            }
        }
    }
    return 1;
}

void place_ship(Battleship* env, int size) {
    State* s = &env->state;
    while (1) {
        int row = (int)(rand_r(&env->rng) % BS_BOARD_SIZE);
        int col = (int)(rand_r(&env->rng) % BS_BOARD_SIZE);
        int axis = (int)(rand_r(&env->rng) % 2);
        int sign = (rand_r(&env->rng) % 2) ? -1 : 1;
        if (!ship_fits(env, row, col, axis, sign, size)) {
            continue;
        }

        for (int i = 0; i < size; i++) {
            int r = row + (axis == 0 ? sign * i : 0);
            int c = col + (axis == 1 ? sign * i : 0);
            s->board[bs_index(r, c)] = 1;
        }
        return;
    }
}

void init(Battleship* env) {
    (void)env;
}

void c_reset(Battleship* env) {
    State* s = &env->state;
    s->tick = 0;
    s->hits = 0;
    s->invalid_actions = 0;
    s->obs = 0;
    s->episode_return = 0.0f;
    for (int i = 0; i < BS_CELLS; i++) {
        s->board[i] = 0;
        s->guesses[i] = 0;
    }
    for (int i = 0; i < BS_NUM_SHIPS; i++) {
        place_ship(env, BS_SHIP_SIZES[i]);
    }
    refresh_observations(env);
}

void c_step(Battleship* env) {
    State* s = &env->state;
    int row = (int)env->actions[0];
    int col = (int)env->actions[1];
    if (row < 0 || row >= BS_BOARD_SIZE || col < 0 || col >= BS_BOARD_SIZE) {
        s->invalid_actions += 1;
        row = 0;
        col = 0;
    }

    int idx = bs_index(row, col);
    int hit = s->board[idx] && !s->guesses[idx];
    s->guesses[idx] = 1;
    s->hits += hit;
    s->tick += 1;

    s->obs = (unsigned char)hit;
    env->rewards[0] = hit ? BS_HIT_REWARD : BS_MISS_REWARD;
    env->terminals[0] = (s->hits == BS_NEEDED_HITS || s->tick >= BS_CELLS) ? 1.0f : 0.0f;
    s->episode_return += env->rewards[0];

    if (env->terminals[0]) {
        add_log(env, s->hits == BS_NEEDED_HITS);
        c_reset(env);
        return;
    }

    refresh_observations(env);
}

void c_render(Battleship* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(420, 360, "PufferLib Battleship");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | hits: %d/%d | obs: %d",
        env->state.tick, BS_CELLS, env->state.hits, BS_NEEDED_HITS, env->state.obs),
        10, 10, 18, (Color){241, 241, 241, 241});
    for (int r = 0; r < BS_BOARD_SIZE; r++) {
        for (int c = 0; c < BS_BOARD_SIZE; c++) {
            int idx = bs_index(r, c);
            const char* text = ".";
            if (env->state.guesses[idx] && env->state.board[idx]) text = "X";
            else if (env->state.guesses[idx]) text = "o";
            else if (env->state.board[idx]) text = "#";
            DrawText(text, 24 + c * 30, 52 + r * 26, 20, RAYWHITE);
        }
    }
    EndDrawing();
#endif
}

void c_close(Battleship* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
