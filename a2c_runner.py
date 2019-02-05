class A2CRunner():
    def __init__(self, agent, envs, summary_writer, train, num_steps, discount):
        self.agent = agent
        self.envs = envs
        self.summary_writer = summary_writer
        self.train = train
        self.num_steps = num_steps
        self.discount = discount

        self.num_episode = 0
        self.cumulative_score = 0.0
    
    def reset(self):
        obs_raw = self.envs.reset()
    
    def mean_score(self):
        return self.cumulative_score / self.num_episode
