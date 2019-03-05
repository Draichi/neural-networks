'''
Convolutional neural network implemented with plain tensorflow on mnist examples.

This implementation use early stopping and works like this:
- every 100 training iterations, it evaluates the model on the validation set,
- if the model performs better than the best model found so far, then it saves the model to RAM,
- if there is no progress for 100 evaluations in a row, then training is interrupted,
- after training, the code restores the best model found.


Tip: use gpu stat to monitor the gpu usage https://github.com/wookayin/gpustat
'''

import tensorflow as tf
import numpy as np
import os

# Hyperparameters
height = 28
width = 28
channels = 1
conv1_fmaps, conv1_ksize, conv1_stride, conv1_pad = 16, 3, 1, "SAME"
conv2_fmaps, conv2_ksize, conv2_stride, conv2_pad = 32, 3, 1, "SAME"
conv2_dropout_rate = 0.25
n_fc1 = 32
fc1_dropout_rate = 0.5
n_outputs = 10
n_epochs, batch_size = 1000, 20
check_interval = 500
max_checks_without_progress = 20 
n_valid_set = 2000
n_test_set = 300

# Fixed parameters
n_inputs = height*width
pool3_fmaps = conv2_fmaps
iteration = 0
best_loss_val = np.infty
checks_since_last_progress = 0
best_model_params = None
logdir = "logs/conv/"

# Splitting data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:n_valid_set], X_train[n_valid_set:]
y_valid, y_train = y_train[:n_valid_set], y_train[n_valid_set:]
X_test = X_test[:n_test_set]
y_test = y_test[:n_test_set]

def reset_graph(seed=42):
    """
    make this notebook's output stable across runs
        :param seed=42: commom seed
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name="training")

# with tf.name_scope("conv"):
conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize, strides=conv1_stride, \
                        padding=conv1_pad, activation=tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride, \
                        padding=conv2_pad, activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):                         
    pool3 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps*14*14])
    pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def shuffle_batch(X, y, batch_size):
    """
    Return X_batch and y_batch accordly to the batch size
        :param X: np.float32
        :param y: np.float32
        :param batch_size: int
    """
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def get_model_params():
    """
    Gets the model's state (i.e., the value of all the variables)
    """
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):    
    """
    Restores a previous state.
    This is used to speed up early stopping: instead of storing the best
    model found so far to disk, we just save it to memory.
    At the end of training, we roll back to the best model found.
        :param model_params: 
    """
    gvar_names = list(model_params.keys())
    assign_ops = { gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign") for gvar_name in gvar_names}
    init_values = { gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items() }
    feed_dict = { init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names }
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

loss_summary = tf.summary.scalar('Loss', loss)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    # saver.restore(sess, "/tmp/conv_mnist_model.ckpt")
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            iteration += 1
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
                summary_str = loss_summary.eval(feed_dict={X: X_valid, y: y_valid})
                file_writer.add_summary(summary_str, iteration)
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(epoch, acc_batch*100, acc_val*100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("--- Early stopping ---")
            break
    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final accuracy on test set:", acc_test)
    save_path = saver.save(sess, "./conv_mnist_model")

file_writer.flush()
file_writer.close()