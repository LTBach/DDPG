import gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
import joblib

enviroment_name = 'LunarLanderContinuous-v2'
env = gym.make(enviroment_name, render_mode='human')
agent = Agent(alpha=0.0001, beta=0.001,
              state_dims=env.observation_space.shape[0], tau=0.001,
              batch_size=64, fc1_dims=400, fc2_dims=300,
              action_dims = env.action_space.shape[0])

agent.load_model()
observation, _ = env.reset()
done = False
total_reward = 0
agent.noise.reset()
while not done:
    action = agent.choose_action(observation)
    next_observation, reward, done, _, _ = env.step(action)
    agent.remember(observation, action, reward, next_observation, done)
    agent.learn()
    total_reward += reward
    env.render()

print(total_reward)

scores = joblib.load('checkpoint\score_history')
x = [i+1 for i in range(1000)]
running_avg = np.zeros(len(scores))
for i in range(len(running_avg)):
    running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
plt.plot(x, running_avg)
plt.title('Running average of previous 100 scores')
plt.show()
