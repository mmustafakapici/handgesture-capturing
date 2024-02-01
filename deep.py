import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Create the game environment
env_name = "CartPole-v1"
env = gym.make(env_name)

# Wrap the environment with DummyVecEnv for compatibility with Stable Baselines3
env = DummyVecEnv([lambda: env])

# Initialize the agent using the Proximal Policy Optimization (PPO) algorithm
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=20000)

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Test the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()

env.close()
