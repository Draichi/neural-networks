import tensorflow as tf
import gym, random, os, tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
# ------------------------------------------------------------>

# disable warnings from tensorflow, default '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# learning rate
LEARNING_RATE = 1e-3
GOAL_STEPS = 500
SCORE_REQUIREMENT = 50
INITIAL_GAMES = 10000
# ------------------------------------------------------------>

env = gym.make('CartPole-v0')
env.reset()
# ------------------------------------------------------------>
# trainning data
def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(INITIAL_GAMES):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(GOAL_STEPS):
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
        if score >= SCORE_REQUIREMENT:
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
    np.save('initial_population.npy', training_data_save)
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data
# ------------------------------------------------------------>

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
    network = fully_connected(
        network, 
        2, 
        activation='softmax'
    )
    # --- regression ---
    network = regression(
        network, 
        optimizer='adam', 
        learning_rate=LEARNING_RATE, 
        loss='categorical_crossentropy', 
        name='targets'
    )
    model = tflearn.DNN(network, tensorboard_dir='log')
    return model
# ------------------------------------------------------------>

# train the model using the trainnig data
def train_model(training_data, model=False):
    # trainning data contains [observations, output]
    # output = 1,0/0,1
    # the X featuresets are all observations
    # y = all outputs
    X = np.array(
        [i[0] for i in training_data]
    ).reshape(
        -1,
        len(training_data[0][0]),
        1
    )
    y = [i[1] for i in training_data]
    # this is just if we haven't a model.
    # we have a model
    if not model:
        model = neural_network_model(input_size = len(X[0]))
    # fit = input, output, epochs, 
    model.fit(
        {'input': X}, 
        {'targets': y}, 
        n_epoch=3, 
        snapshot_step=500, 
        show_metric=True, 
        run_id='model_fitting_log'
    )
    return model
# ------------------------------------------------------------>

# traning and running
# training_data = initial_population()
training_data = np.load('initial_population.npy')
model = train_model(training_data)
# PICKLE HERE
# ------------------------------------------------------------>

# run the game
scores = []
choices = []
for each_game in range(20):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(GOAL_STEPS):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            # argmax because is 'one hot' array
            # and action needs to be 0 or 1
            # we're gonna to predict our previous observations
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            # ---just to see wat happens---
            # print(model.predict(prev_obs.reshape(-1,len(prev_obs),1)))
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    scores.append(score)
# ------------------------------------------------------------>

print('Average Score:', sum(scores)/len(scores))
print('Choice1: {}, Choice 0: {}'.format(choices.count(1)/len(choices), choices.count(0)/len(choices)))