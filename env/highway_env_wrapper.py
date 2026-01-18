import gymnasium as gym
import highway_env

class HighwayEnvWrapper:
    def __init__(self, env_name="highway-v0"):
        self.env = gym.make(env_name)

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def close(self):
        self.env.close()
