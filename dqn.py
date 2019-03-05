"""Actor-Critic DQN to play Ms.Pacman

A neural network which takes a state s as input and outputs an estimate
of the corresponding Q-value per action. The DQN is composed of three convolutional
layers, followed by two fully connected layers, including the output layer.
The training algorithm requires two DQNs with the same architecture (but different parameters),
one will be the actor and the other will be the critic. 
The critic will try to make its Q-value predictions match the Q-values estimated by the actor through 
its experience of the game. The actor will play for a while, storing all its experiences ina areplay memory,
each memory will be a 5-truple (s, a, s', r, done). Next, at a regular intervals it will sample a batch
of memories from the replay memory and it will estimate the Q-values from these memories, finally, it will
train the critic DQN to predict these Q-values. At regular intervals it will copy the critic to the actor.

Run:
    $ python dqn.py
"""

import numpy as np
import numpy.random as rnd
import tensorflow as tf
import gym
import os
from collections import deque

env = gym.make("Mspacman-v0")
input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activations = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10 # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
replay_memory_size = 10000
replay_memory = deque([], maxlen=replay_memory_size)
eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000
n_steps = 100000 # training steps
training_start = 1000 # start training after x game itarations
training_interval = 3 # run a training step after x game iteractions
save_steps = 50
copy_steps = 25
discount_rate = 0.95
skip_start = 90 # skip the waiting time
batch_size = 50
iteration = 0
checkpoint_path = "./dqn_checkpoint.ckpt"
done = True # env needs to be reset
n_outputs = env.action_space.n # 9 discrete actions
mspacman_color = np.array([210, 164, 74]).mean()
initializer = tf.variance_scaling_initializer()

# start construction phase
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
    """Creates the DQN

    The traineble_vars_by_name dict gathers all the trainable variables of this DQN,
    it will be useful when copy the critic DQN to the actor DQN.

    X_state -- enviroment's state.
    name -- name of the variable scope.
    """
    prev_layer = X_state
    conv_layers = []
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, stride, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activations):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size, stride=stride,
                padding=padding, activation=activation, kernel_initializer=initializer)
            conv_layers.append(prev_layer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden, activation=hidden_activation, kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    return outputs, trainable_vars_by_name

def sample_memories(batch_size):
    """Randomly sample a batch of experiences from the replay memory
    """
    indices = rnd.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # s, a , r, s', done
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))

def epsilon_greedy(q_values, step):
    """epsilon greedy policy to explore the game
    """
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if rnd.rand() < epsilon:
        return rnd.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

# input placeholders
X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
actor_q_values, actor_vars = q_network(X_state, name="q_network/actor")
critic_q_values, critic_vars = q_network(X_state, name="q_network/critic")
copy_ops = [ actor_var.assign(critic_vars[var_name]) for var_name, actor in actor_vars.items() ]
copy_critic_to_actor = tf.group(*copy_ops) # tf.group to group all the assignment operations into a single convenient operation

# critic DQN operations:
# Since the DQN outputs one Q-value for every possible action, we need to keep only
# the Q-value that corresponds to the action that was actually chosen in this memory.
# For this we will convert the action to a one-hot vector and multiply it by the Q-values.
# Then just sum over the first axis to obtain only the desired Q-value prediction for each memory.
X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, n_outputs), axis=1, keep_dims=True)

# training operations:
# the optimizer minimize() will take care to increment the global_step nontrainable variable
y = tf.placeholder(tf.float32, shape=[None, 1])
cost = tf.reduce_mean(tf.square(y - q_value))
global_step = tf.Variable(0, trainable=False, name="global_step")
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
# end construction phase

# start execution phase
with tf.Session() as sess:
    if os.path.isfile(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        init.run()
    while True:
        step = global_step.eval()
        if step >= n_steps:
            break