// Shared CartPole physics for the POPGym cartpole family.
// Include from a family header after defining:
//   CARTPOLE_ENV           env struct/typedef name
//   CARTPOLE_WINDOW_TITLE  render window title string
// and at most one of:
//   CARTPOLE_NOISY_OBS     gaussian noise (noise_sigma) on position obs
//   CARTPOLE_VELOCITY_OBS  velocities as obs (positions on the reset step)

#pragma once

#include <math.h>
#include <stdlib.h>
#include "popgym_check.h"
#include "popgym_rand.h"
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define CARTPOLE_OBS_SIZE 2
#define CARTPOLE_X_THRESHOLD 2.4f
#define CARTPOLE_THETA_THRESHOLD_RADIANS (12.0f * 2.0f * (float)M_PI / 360.0f)
#ifdef CARTPOLE_NOISY_OBS
#define CARTPOLE_X_OBS_LIMIT (2.0f * CARTPOLE_X_THRESHOLD)
#define CARTPOLE_THETA_OBS_LIMIT (2.0f * CARTPOLE_THETA_THRESHOLD_RADIANS)
#endif

typedef struct {
    float score;
    float episode_return;
    float episode_length;
    float x_threshold_termination;
    float pole_angle_termination;
    float max_steps_termination;
    float n;
} Log;

typedef struct {
    int tick;
    float x;
    float x_dot;
    float theta;
    float theta_dot;
    float episode_return;
#ifdef CARTPOLE_VELOCITY_OBS
    int reset_obs;
#endif
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
#ifdef CARTPOLE_NOISY_OBS
    float noise_sigma;
#endif
    unsigned int rng;
} CARTPOLE_ENV;

void refresh_observations(CARTPOLE_ENV* env) {
    State* s = &env->state;
#if defined(CARTPOLE_VELOCITY_OBS)
    if (s->reset_obs) {
        env->observations[0] = s->x;
        env->observations[1] = s->theta;
    } else {
        env->observations[0] = s->x_dot;
        env->observations[1] = s->theta_dot;
    }
#elif defined(CARTPOLE_NOISY_OBS)
    env->observations[0] = popgym_clipf(
        s->x + env->noise_sigma * popgym_normal(&env->rng),
        -CARTPOLE_X_OBS_LIMIT,
        CARTPOLE_X_OBS_LIMIT
    );
    env->observations[1] = popgym_clipf(
        s->theta + env->noise_sigma * popgym_normal(&env->rng),
        -CARTPOLE_THETA_OBS_LIMIT,
        CARTPOLE_THETA_OBS_LIMIT
    );
#else
    env->observations[0] = s->x;
    env->observations[1] = s->theta;
#endif
}

void add_log(CARTPOLE_ENV* env, int x_done, int theta_done, int timeout) {
    State* s = &env->state;
    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.x_threshold_termination += x_done ? 1.0f : 0.0f;
    env->log.pole_angle_termination += theta_done ? 1.0f : 0.0f;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void init(CARTPOLE_ENV* env) {
    POPGYM_CHECK(env->max_episode_length > 0,
        "max_episode_length must be > 0 (got %d)", env->max_episode_length);
#ifdef CARTPOLE_NOISY_OBS
    POPGYM_CHECK(env->noise_sigma >= 0.0f,
        "noise_sigma must be >= 0 (got %f)", (double)env->noise_sigma);
#endif
}

void c_reset(CARTPOLE_ENV* env) {
    State* s = &env->state;
    s->tick = 0;
    s->x = popgym_randf(&env->rng, -0.05f, 0.05f);
    s->x_dot = popgym_randf(&env->rng, -0.05f, 0.05f);
    s->theta = popgym_randf(&env->rng, -0.05f, 0.05f);
    s->theta_dot = popgym_randf(&env->rng, -0.05f, 0.05f);
    s->episode_return = 0.0f;
#ifdef CARTPOLE_VELOCITY_OBS
    s->reset_obs = 1;
#endif
    refresh_observations(env);
}

void c_step(CARTPOLE_ENV* env) {
    State* s = &env->state;
    int action = (int)env->actions[0];
    if (action != 0 && action != 1) {
        action = 0;
    }

    const float gravity = 9.8f;
    const float masscart = 1.0f;
    const float masspole = 0.1f;
    const float total_mass = masspole + masscart;
    const float length = 0.5f;
    const float polemass_length = masspole * length;
    const float force_mag = 10.0f;
    const float tau = 0.02f;

    float force = action == 1 ? force_mag : -force_mag;
    float costheta = cosf(s->theta);
    float sintheta = sinf(s->theta);
    float temp = (force + polemass_length * s->theta_dot * s->theta_dot * sintheta)
        / total_mass;
    float thetaacc = (gravity * sintheta - costheta * temp)
        / (length * (4.0f / 3.0f - masspole * costheta * costheta / total_mass));
    float xacc = temp - polemass_length * thetaacc * costheta / total_mass;

    s->x += tau * s->x_dot;
    s->x_dot += tau * xacc;
    s->theta += tau * s->theta_dot;
    s->theta_dot += tau * thetaacc;
    s->tick += 1;
#ifdef CARTPOLE_VELOCITY_OBS
    s->reset_obs = 0;
#endif

    int x_done = s->x < -CARTPOLE_X_THRESHOLD || s->x > CARTPOLE_X_THRESHOLD;
    int theta_done = s->theta < -CARTPOLE_THETA_THRESHOLD_RADIANS
        || s->theta > CARTPOLE_THETA_THRESHOLD_RADIANS;
    int timeout = s->tick >= env->max_episode_length;
    int done = x_done || theta_done || timeout;

    env->rewards[0] = 1.0f / (float)env->max_episode_length;
    env->terminals[0] = done ? 1.0f : 0.0f;
    s->episode_return += env->rewards[0];

    if (done) {
        add_log(env, x_done, theta_done, timeout);
        c_reset(env);
        return;
    }

    refresh_observations(env);
}

void c_render(CARTPOLE_ENV* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(600, 220, CARTPOLE_WINDOW_TITLE);
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
#if defined(CARTPOLE_VELOCITY_OBS)
    DrawText(TextFormat("tick: %d/%d | xdot: %.3f | thetadot: %.3f",
        env->state.tick, env->max_episode_length,
        env->state.x_dot, env->state.theta_dot),
        10, 10, 18, (Color){241, 241, 241, 241});
#elif defined(CARTPOLE_NOISY_OBS)
    DrawText(TextFormat("tick: %d/%d | obs: %.3f %.3f",
        env->state.tick, env->max_episode_length,
        env->observations[0], env->observations[1]),
        10, 10, 18, (Color){241, 241, 241, 241});
#else
    DrawText(TextFormat("tick: %d/%d | x: %.3f | theta: %.3f",
        env->state.tick, env->max_episode_length, env->state.x, env->state.theta),
        10, 10, 18, (Color){241, 241, 241, 241});
#endif
    EndDrawing();
#endif
}

void c_close(CARTPOLE_ENV* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
