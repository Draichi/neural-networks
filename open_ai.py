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

# trainning data
def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(inital_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            # only rearange 0's and 1's
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
                # convert to one-hot 
                # (this is the output layer for
                #  our neural network)
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
# descomment to get the trainning data:
#initial_population()

# neural network model
def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')
    # --- 5 fully connected layers ---
    # input, 128 nodes on dat layer, 
    # activation function = rectified linear
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    # 2
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    # 3
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    # 4
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    # 5
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    # --- output layer ---
    # 2 outputs = 0,1
    network = fully_connected(network, 2, activation='softmax')
    # --- regression ---
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')
    return model

# train the model using the trainnig data
def train_model(training_data, model=False):
    # trainning data contains [observations, output]
    # output = 1,0/0,1
    # the X featuresets are all observations
    # y = all outputs
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = [i[1] for i in training_data]
    # this is just if we haven't a model.
    # we have a model
    if not model:
        model = neural_network_model(input_size = len(X[0]))
    # fit = input, output, epochs, 
    model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='OpenAIStuff')
    return model

# RUN
training_data = initial_population()
model = train_model(training_data)