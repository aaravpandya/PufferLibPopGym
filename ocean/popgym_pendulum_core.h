// Shared Pendulum physics for the POPGym pendulum family.
// Include from a family header after defining:
//   PENDULUM_ENV           env struct/typedef name
//   PENDULUM_WINDOW_TITLE  render window title string
// and optionally:
//   PENDULUM_NOISY_OBS     gaussian noise (noise_sigma) on cos/sin obs

#pragma once

#include <math.h>
#include <stdlib.h>
#include "popgym_check.h"
#include "popgym_rand.h"
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define PENDULUM_OBS_SIZE 2
#define PENDULUM_MAX_SPEED 8.0f
#define PENDULUM_MAX_TORQUE 2.0f
#define PENDULUM_DT 0.05f
#define PENDULUM_REWARD_LOW -16.2736044f

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
#ifdef PENDULUM_NOISY_OBS
    float noise_sigma;
#endif
    unsigned int rng;
} PENDULUM_ENV;

static inline float pendulum_angle_normalize(float x) {
    while (x > (float)M_PI) x -= 2.0f * (float)M_PI;
    while (x < -(float)M_PI) x += 2.0f * (float)M_PI;
    return x;
}

void refresh_observations(PENDULUM_ENV* env) {
#ifdef PENDULUM_NOISY_OBS
    env->observations[0] = popgym_clipf(
        cosf(env->state.theta) + env->noise_sigma * popgym_normal(&env->rng),
        -1.0f,
        1.0f
    );
    env->observations[1] = popgym_clipf(
        sinf(env->state.theta) + env->noise_sigma * popgym_normal(&env->rng),
        -1.0f,
        1.0f
    );
#else
    env->observations[0] = cosf(env->state.theta);
    env->observations[1] = sinf(env->state.theta);
#endif
}

float transform_reward(PENDULUM_ENV* env, float reward) {
    float half_range = -PENDULUM_REWARD_LOW / 2.0f;
    return (reward + half_range) / half_range / (float)env->max_episode_length;
}

void add_log(PENDULUM_ENV* env, int timeout) {
    State* s = &env->state;
    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void init(PENDULUM_ENV* env) {
    POPGYM_CHECK(env->max_episode_length > 0,
        "max_episode_length must be > 0 (got %d)", env->max_episode_length);
#ifdef PENDULUM_NOISY_OBS
    POPGYM_CHECK(env->noise_sigma >= 0.0f,
        "noise_sigma must be >= 0 (got %f)", (double)env->noise_sigma);
#endif
}

void c_reset(PENDULUM_ENV* env) {
    State* s = &env->state;
    s->tick = 0;
    s->theta = popgym_randf(&env->rng, -(float)M_PI, (float)M_PI);
    s->theta_dot = popgym_randf(&env->rng, -1.0f, 1.0f);
    s->episode_return = 0.0f;
    refresh_observations(env);
}

void c_step(PENDULUM_ENV* env) {
    State* s = &env->state;
    float torque = popgym_clipf(env->actions[0], -PENDULUM_MAX_TORQUE, PENDULUM_MAX_TORQUE);
    float angle = pendulum_angle_normalize(s->theta);
    float costs = angle * angle
        + 0.1f * s->theta_dot * s->theta_dot
        + 0.001f * torque * torque;

    float theta_dot = s->theta_dot
        + (15.0f * sinf(s->theta) + 3.0f * torque) * PENDULUM_DT;
    s->theta_dot = popgym_clipf(theta_dot, -PENDULUM_MAX_SPEED, PENDULUM_MAX_SPEED);
    s->theta += s->theta_dot * PENDULUM_DT;
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

void c_render(PENDULUM_ENV* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(420, 220, PENDULUM_WINDOW_TITLE);
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

void c_close(PENDULUM_ENV* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
