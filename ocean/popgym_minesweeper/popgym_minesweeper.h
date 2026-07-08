// Native POPGym MineSweeperEasy semantics.
// Observation is the adjacent mine count for the last selected square.

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#ifndef MS_ROWS
#define MS_ROWS 4
#endif
#ifndef MS_COLS
#define MS_COLS 4
#endif
#define MS_CELLS (MS_ROWS * MS_COLS)
#ifndef MS_MINES
#define MS_MINES 2
#endif
#define MS_CLEAR 0
#define MS_MINE 1
#define MS_VIEWED 2
#define MS_MAX_STEPS (MS_CELLS - MS_MINES)
#define MS_SUCCESS_REWARD (1.0f / (float)MS_MAX_STEPS)
#define MS_FAIL_REWARD (-0.5f - MS_SUCCESS_REWARD)
#define MS_BAD_ACTION_REWARD (-0.5f / (float)(MS_MAX_STEPS - 2))

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
    int viewed_clear;
    int invalid_actions;
    unsigned char hidden_grid[MS_CELLS];
    unsigned char neighbor_grid[MS_CELLS];
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
} PopGymMineSweeper;

static inline int ms_index(int row, int col) {
    return row * MS_COLS + col;
}

static inline bool ms_in_bounds(int row, int col) {
    return row >= 0 && row < MS_ROWS && col >= 0 && col < MS_COLS;
}

void refresh_observations(PopGymMineSweeper* env, unsigned char obs) {
    env->observations[0] = obs;
}

void add_log(PopGymMineSweeper* env, bool success) {
    env->log.score += success ? 1.0f : 0.0f;
    env->log.success += success ? 1.0f : 0.0f;
    env->log.episode_return += env->state.episode_return;
    env->log.episode_length += (float)env->state.tick;
    env->log.invalid_action_rate += (float)env->state.invalid_actions
        / (float)(env->state.tick > 0 ? env->state.tick : MS_MAX_STEPS);
    env->log.n += 1.0f;
}

void init(PopGymMineSweeper* env) {
    (void)env;
}

void c_reset(PopGymMineSweeper* env) {
    State* s = &env->state;
    s->tick = 0;
    s->viewed_clear = 0;
    s->invalid_actions = 0;
    s->episode_return = 0.0f;

    for (int i = 0; i < MS_CELLS; i++) {
        s->hidden_grid[i] = MS_CLEAR;
        s->neighbor_grid[i] = 0;
    }

    int placed = 0;
    while (placed < MS_MINES) {
        int idx = (int)(rand_r(&env->rng) % MS_CELLS);
        if (s->hidden_grid[idx] == MS_MINE) {
            continue;
        }
        s->hidden_grid[idx] = MS_MINE;
        placed += 1;
    }

    for (int row = 0; row < MS_ROWS; row++) {
        for (int col = 0; col < MS_COLS; col++) {
            int count = 0;
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    int nr = row + dr;
                    int nc = col + dc;
                    if (ms_in_bounds(nr, nc)
                            && s->hidden_grid[ms_index(nr, nc)] == MS_MINE) {
                        count += 1;
                    }
                }
            }
            s->neighbor_grid[ms_index(row, col)] = (unsigned char)count;
        }
    }

    refresh_observations(env, 0);
}

void c_step(PopGymMineSweeper* env) {
    State* s = &env->state;
    int row = (int)env->actions[0];
    int col = (int)env->actions[1];

    bool terminal = false;
    bool success = false;
    float reward = 0.0f;
    unsigned char obs = 0;

    if (!ms_in_bounds(row, col)) {
        s->invalid_actions += 1;
        reward = MS_BAD_ACTION_REWARD;
    } else {
        int idx = ms_index(row, col);
        obs = s->neighbor_grid[idx];
        if (s->hidden_grid[idx] == MS_MINE) {
            terminal = true;
            reward = MS_FAIL_REWARD;
        } else if (s->hidden_grid[idx] == MS_VIEWED) {
            reward = MS_BAD_ACTION_REWARD;
        } else {
            s->hidden_grid[idx] = MS_VIEWED;
            s->viewed_clear += 1;
            reward = MS_SUCCESS_REWARD;
        }
    }

    s->tick += 1;
    if (s->viewed_clear == MS_MAX_STEPS) {
        terminal = true;
        success = true;
    }
    if (s->tick == MS_MAX_STEPS && !terminal) {
        terminal = true;
    }

    env->rewards[0] = reward;
    env->terminals[0] = terminal ? 1.0f : 0.0f;
    s->episode_return += reward;

    if (terminal) {
        add_log(env, success);
        c_reset(env);
    } else {
        refresh_observations(env, obs);
    }
}

void c_render(PopGymMineSweeper* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(360, 260, "PufferLib MineSweeperEasy");
        SetTargetFPS(20);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d | viewed: %d/%d",
        env->state.tick, env->state.viewed_clear, MS_CELLS - MS_MINES),
        10, 10, 18, (Color){241, 241, 241, 241});
    for (int row = 0; row < MS_ROWS; row++) {
        for (int col = 0; col < MS_COLS; col++) {
            int idx = ms_index(row, col);
            const char* text = ".";
            if (env->state.hidden_grid[idx] == MS_VIEWED) {
                text = TextFormat("%d", env->state.neighbor_grid[idx]);
            }
            DrawText(text, 40 + col * 42, 58 + row * 36, 24, RAYWHITE);
        }
    }
    EndDrawing();
#endif
}

void c_close(PopGymMineSweeper* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
