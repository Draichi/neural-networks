'''
Policy gradient

Each training iteration starts by running the policy for n_games_per_update (with maximum n_max_steps
per episode, to avoide running forever). At each step, we also compute the gradients. After n_games_per_update
have been run, we compute the action socres using the discount_and_normalize_rewards() function; we go through each
trainable variable, across all episodes and all steps, to multiply each gradient vector by its corresponding
action score; and we compute the mean of the resulting gradients. Finally, we run the training operation,
feeding it these mean gradients (one per trainable variable. We also save the model every save_iterations times.
'''

import tensorflow as tf
import numpy as np
import gym

# neural net params
n_inputs = 4
n_hidden = 4
n_outputs = 1
learning_rate = 0.01
# policy params
n_iterations = 250
n_max_steps = 1000
n_games_per_update = 10 # train the policy every 'x' episodes
save_iterations = 10 # save the model every 'x' iterations
discount_rate = 0.99

def discount_rewards(rewards, discount_rate):
    '''
    Return the discounted reward given the raw rewards
    '''
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    Return the normalize rewards with zero mean and standard deviation
    '''
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]

# start construction phase
initializer = tf.variance_scaling_initializer()
env = gym.make("CartPole-v0")

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

y = 1. - tf.to_float(action)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy) # compute_grandients() instead of minimize() bcause we want to tweak the gradient before apply them

gradients = [grad for grad, variable in grads_and_vars]

gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# end construction phase

# start execution phase
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradient_val = sess.run(
                    [action, gradients],
                    feed_dict={X: obs.reshape(1, n_inputs)}) # one obs
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradient_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
        # At this point we have run the policy for n_games_per_update, 
        # and we are ready for a policy update
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            # multiply the gradient by the action scores, and compute the mean
            mean_gradients = np.mean(
                [reward * all_gradients[game_index][step][var_index]
                    for game_index, rewards in enumerate(all_rewards)
                    for step, reward in enumerate(rewards)],
                axis=0)
            feed_dict[grad_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./policy_gradient.ckpt")

# end execution phase
