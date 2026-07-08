// Native POPGym NoisyPositionOnlyCartPole semantics.

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define NPOC_OBS_SIZE 2
#define NPOC_X_THRESHOLD 2.4f
#define NPOC_X_OBS_LIMIT (2.0f * NPOC_X_THRESHOLD)
#define NPOC_THETA_THRESHOLD_RADIANS (12.0f * 2.0f * (float)M_PI / 360.0f)
#define NPOC_THETA_OBS_LIMIT (2.0f * NPOC_THETA_THRESHOLD_RADIANS)

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
    float noise_sigma;
    unsigned int rng;
} NoisyPositionOnlyCartPole;

static inline float npoc_randf(NoisyPositionOnlyCartPole* env, float lo, float hi) {
    float t = (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
    return lo + t * (hi - lo);
}

static inline float npoc_unit_open(NoisyPositionOnlyCartPole* env) {
    return ((float)rand_r(&env->rng) + 1.0f) / ((float)RAND_MAX + 2.0f);
}

static inline float npoc_normal(NoisyPositionOnlyCartPole* env) {
    float u1 = npoc_unit_open(env);
    float u2 = npoc_unit_open(env);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static inline float npoc_clip(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

void refresh_observations(NoisyPositionOnlyCartPole* env) {
    env->observations[0] = npoc_clip(
        env->state.x + env->noise_sigma * npoc_normal(env),
        -NPOC_X_OBS_LIMIT,
        NPOC_X_OBS_LIMIT
    );
    env->observations[1] = npoc_clip(
        env->state.theta + env->noise_sigma * npoc_normal(env),
        -NPOC_THETA_OBS_LIMIT,
        NPOC_THETA_OBS_LIMIT
    );
}

void add_log(NoisyPositionOnlyCartPole* env, int x_done, int theta_done, int timeout) {
    State* s = &env->state;
    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.x_threshold_termination += x_done ? 1.0f : 0.0f;
    env->log.pole_angle_termination += theta_done ? 1.0f : 0.0f;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void init(NoisyPositionOnlyCartPole* env) {
    assert(env->max_episode_length > 0);
    assert(env->noise_sigma >= 0.0f);
}

void c_reset(NoisyPositionOnlyCartPole* env) {
    State* s = &env->state;
    s->tick = 0;
    s->x = npoc_randf(env, -0.05f, 0.05f);
    s->x_dot = npoc_randf(env, -0.05f, 0.05f);
    s->theta = npoc_randf(env, -0.05f, 0.05f);
    s->theta_dot = npoc_randf(env, -0.05f, 0.05f);
    s->episode_return = 0.0f;
    refresh_observations(env);
}

void c_step(NoisyPositionOnlyCartPole* env) {
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

    int x_done = s->x < -NPOC_X_THRESHOLD || s->x > NPOC_X_THRESHOLD;
    int theta_done = s->theta < -NPOC_THETA_THRESHOLD_RADIANS
        || s->theta > NPOC_THETA_THRESHOLD_RADIANS;
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

void c_render(NoisyPositionOnlyCartPole* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(600, 220, "PufferLib NoisyPositionOnlyCartPole");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | obs: %.3f %.3f",
        env->state.tick, env->max_episode_length,
        env->observations[0], env->observations[1]),
        10, 10, 18, (Color){241, 241, 241, 241});
    EndDrawing();
#endif
}

void c_close(NoisyPositionOnlyCartPole* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
