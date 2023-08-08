import os
import gym
import torch as T
import numpy as np
import joblib
from agent import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    enviroment_name = 'LunarLanderContinuous-v2'
    env = gym.make(enviroment_name)
    agent = Agent(alpha=0.0001, beta=0.001,
                  state_dims=env.observation_space.shape[0], tau=0.001,
                  batch_size=64, fc1_dims=400, fc2_dims=300,
                  action_dims = env.action_space.shape[0])

    n_games = 1000
    filename =  enviroment_name +'_alpha_' + str(agent.alpha) + '_beta_' \
                + str(agent.beta) + '_' + str(n_games) + '_games'
    figure_file = os.path.join('plots', filename + '.png')

    best_score = env.reward_range[0]
    score_history = [] 
    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, _, _ = env.step(action)
            agent.remember(observation, action, reward, next_observation,
                           done)
            agent.learn()
            score += reward
            observation = next_observation
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_model()

        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
    joblib.dump(score_history, os.path.join('checkpoint', 'score_history'))
