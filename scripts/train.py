from stable_baselines3 import PPO
from env.highway_env_wrapper import HighwayEnvWrapper

env = HighwayEnvWrapper()

model = PPO(
    "MlpPolicy",
    env.env,
    verbose=1,
    learning_rate=3e-4,
    gamma=0.99,
    clip_range=0.2
)

model.learn(total_timesteps=100_000)
model.save("ppo_highway")
