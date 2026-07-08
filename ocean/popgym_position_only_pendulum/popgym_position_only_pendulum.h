// Native POPGym PositionOnlyPendulum semantics.

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define POP_OBS_SIZE 2
#define POP_MAX_SPEED 8.0f
#define POP_MAX_TORQUE 2.0f
#define POP_DT 0.05f
#define POP_REWARD_LOW -16.2736044f

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float max_steps_termination;
    float n;
} Log;

typedef struct {
    int tick;
    float theta;
    float theta_dot;
    float episode_return;
} State;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    State state;
    int max_episode_length;
    unsigned int rng;
} PositionOnlyPendulum;

static inline float pop_randf(PositionOnlyPendulum* env, float lo, float hi) {
    float t = (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
    return lo + t * (hi - lo);
}

static inline float pop_clip(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

static inline float pop_angle_normalize(float x) {
    while (x > (float)M_PI) x -= 2.0f * (float)M_PI;
    while (x < -(float)M_PI) x += 2.0f * (float)M_PI;
    return x;
}

void refresh_observations(PositionOnlyPendulum* env) {
    env->observations[0] = cosf(env->state.theta);
    env->observations[1] = sinf(env->state.theta);
}

float transform_reward(PositionOnlyPendulum* env, float reward) {
    float half_range = -POP_REWARD_LOW / 2.0f;
    return (reward + half_range) / half_range / (float)env->max_episode_length;
}

void add_log(PositionOnlyPendulum* env, int timeout) {
    State* s = &env->state;
    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void init(PositionOnlyPendulum* env) {
    assert(env->max_episode_length > 0);
}

void c_reset(PositionOnlyPendulum* env) {
    State* s = &env->state;
    s->tick = 0;
    s->theta = pop_randf(env, -(float)M_PI, (float)M_PI);
    s->theta_dot = pop_randf(env, -1.0f, 1.0f);
    s->episode_return = 0.0f;
    refresh_observations(env);
}

void c_step(PositionOnlyPendulum* env) {
    State* s = &env->state;
    float torque = pop_clip(env->actions[0], -POP_MAX_TORQUE, POP_MAX_TORQUE);
    float angle = pop_angle_normalize(s->theta);
    float costs = angle * angle
        + 0.1f * s->theta_dot * s->theta_dot
        + 0.001f * torque * torque;

    float theta_dot = s->theta_dot
        + (15.0f * sinf(s->theta) + 3.0f * torque) * POP_DT;
    s->theta_dot = pop_clip(theta_dot, -POP_MAX_SPEED, POP_MAX_SPEED);
    s->theta += s->theta_dot * POP_DT;
    s->tick += 1;

    int timeout = s->tick >= env->max_episode_length;
    env->rewards[0] = transform_reward(env, -costs);
    env->terminals[0] = timeout ? 1.0f : 0.0f;
    s->episode_return += env->rewards[0];

    if (timeout) {
        add_log(env, timeout);
        c_reset(env);
        return;
    }

    refresh_observations(env);
}

void c_render(PositionOnlyPendulum* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(420, 220, "PufferLib PositionOnlyPendulum");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | obs: %.3f %.3f | reward: %.3f",
        env->state.tick, env->max_episode_length,
        env->observations[0], env->observations[1], env->rewards[0]),
        10, 10, 18, (Color){241, 241, 241, 241});
    EndDrawing();
#endif
}

void c_close(PositionOnlyPendulum* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
