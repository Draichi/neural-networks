"""
A neural network which takes a state s as input and outputs an estimate
of the corresponding Q-value per action. The DQN is composed of three convolutional
layers, followed by two fully connected layers, including the output layer.
The training algorithm requires two DQNs with the same architecture (but different parameters),
one will be the actor and the other will be the critic. At regular intervals it will copy
the critic to the actor.

Run:
    $ python dqn.py
"""

import numpy as np
import tensorflow as tf
import gym

env = gym.make("Mspacman-v0")
input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides[4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activations = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10 # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n # 9 discrete actions
mspacman_color = np.array([210, 164, 74]).mean()
initializer = tf.variance_scaling_initializer()

def preprocess_observations(obs):
    """Get the obs from gym (game screenshot), crop and shrink it
    down to 88 x 80 px, convert it to grayscale, and improve the contrast.
    This reduce the amount of computations required by the DQN,
    and speed up training.
    
    obs -- the observation returned from env.step()
    """
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to grayscale
    img[img==mspacman_color] = 0 # improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(88, 80, 1)

def q_network(X_state, name):
    """