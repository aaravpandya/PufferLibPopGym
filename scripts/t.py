from popgym.envs import RepeatPrevious

env = RepeatPrevious(num_decks=1)

obs = env.reset()
print(obs)

for _ in range(10):
    sampled_action = env.action_space.sample()
    print(f"{sampled_action=}")
    obs = env.step(sampled_action)
    print(f"{obs=}")


