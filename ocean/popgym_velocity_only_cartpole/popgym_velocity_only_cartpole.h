// Native POPGym VelocityOnlyCartPole semantics.

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#ifndef PUFFER_PYTHON_EXTENSION
#include "raylib.h"
#endif

#define VOC_OBS_SIZE 2
#define VOC_X_THRESHOLD 2.4f
#define VOC_THETA_THRESHOLD_RADIANS (12.0f * 2.0f * (float)M_PI / 360.0f)

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
    int reset_obs;
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
} VelocityOnlyCartPole;

static inline float voc_randf(VelocityOnlyCartPole* env, float lo, float hi) {
    float t = (float)rand_r(&env->rng) / ((float)RAND_MAX + 1.0f);
    return lo + t * (hi - lo);
}

void refresh_observations(VelocityOnlyCartPole* env) {
    State* s = &env->state;
    if (s->reset_obs) {
        env->observations[0] = s->x;
        env->observations[1] = s->theta;
    } else {
        env->observations[0] = s->x_dot;
        env->observations[1] = s->theta_dot;
    }
}

void add_log(VelocityOnlyCartPole* env, int x_done, int theta_done, int timeout) {
    State* s = &env->state;
    env->log.score += s->episode_return;
    env->log.episode_return += s->episode_return;
    env->log.episode_length += (float)s->tick;
    env->log.x_threshold_termination += x_done ? 1.0f : 0.0f;
    env->log.pole_angle_termination += theta_done ? 1.0f : 0.0f;
    env->log.max_steps_termination += timeout ? 1.0f : 0.0f;
    env->log.n += 1.0f;
}

void init(VelocityOnlyCartPole* env) {
    assert(env->max_episode_length > 0);
}

void c_reset(VelocityOnlyCartPole* env) {
    State* s = &env->state;
    s->tick = 0;
    s->x = voc_randf(env, -0.05f, 0.05f);
    s->x_dot = voc_randf(env, -0.05f, 0.05f);
    s->theta = voc_randf(env, -0.05f, 0.05f);
    s->theta_dot = voc_randf(env, -0.05f, 0.05f);
    s->episode_return = 0.0f;
    s->reset_obs = 1;
    refresh_observations(env);
}

void c_step(VelocityOnlyCartPole* env) {
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
    s->reset_obs = 0;

    int x_done = s->x < -VOC_X_THRESHOLD || s->x > VOC_X_THRESHOLD;
    int theta_done = s->theta < -VOC_THETA_THRESHOLD_RADIANS
        || s->theta > VOC_THETA_THRESHOLD_RADIANS;
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

void c_render(VelocityOnlyCartPole* env) {
#ifdef PUFFER_PYTHON_EXTENSION
    (void)env;
#else
    if (!IsWindowReady()) {
        InitWindow(600, 220, "PufferLib VelocityOnlyCartPole");
        SetTargetFPS(30);
    }
    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){6, 24, 24, 255});
    DrawText(TextFormat("tick: %d/%d | xdot: %.3f | thetadot: %.3f",
        env->state.tick, env->max_episode_length,
        env->state.x_dot, env->state.theta_dot),
        10, 10, 18, (Color){241, 241, 241, 241});
    EndDrawing();
#endif
}

void c_close(VelocityOnlyCartPole* env) {
    (void)env;
#ifndef PUFFER_PYTHON_EXTENSION
    if (IsWindowReady()) {
        CloseWindow();
    }
#endif
}
