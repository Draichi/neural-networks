import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# learning rate
LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
inital_games = 10000

# random game
def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break
#some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(inital_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            # oly rearange 0's and 1's
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            # reward = 0, 1
            score += reward
            if done:
                break
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                training_data.append([data[0], output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data
initial_population()