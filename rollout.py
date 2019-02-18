def rollout(agents, env, horizon):
    while True:
        episode_length = 0
        timesteps = env.reset()
        for agent in agents:
            agent.reset()

        while True:
            episode_length += 1
            last_timesteps = timesteps
            actions = [agent.step(timestep) for agent, timestep in zip(agents, timesteps)]
            timesteps = env.step(actions)
            done = (episode_length >= horizon) or timesteps[0].last()

            yield [last_timesteps[0], actions[0], timesteps[0]], done

            if done:
                break
