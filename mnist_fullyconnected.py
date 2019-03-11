from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = None

n_epochs = 30
batch_size = 50
n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
initial_learning_rate = 0.01
decay_steps = 10000
decay_rate = 1/10
momentum = 0.9
use_momentum_optimizer = True
use_nesterov = True
dropout_rate = 0.5 # == 1 - keep_prob

def train():
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    sess = tf.InteractiveSession()

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
        y = tf.placeholder(tf.int64, shape=(None), name="y")
        image_shaped = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped, 10)

        
    training = tf.placeholder_with_default(False, shape=(), name="training")
    x_drop = tf.layers.dropout(x, dropout_rate, training=training)
    tf.summary.scalar('dropout_rate', dropout_rate)

    with tf.name_scope("dnn"):
        hidden1 = tf.layers.dense(x_drop, n_hidden1, name="hidden1", activation=tf.nn.relu)
        hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
        hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2", activation=tf.nn.relu)
        hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
        logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

    with tf.name_scope("loss"):
        xentropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        # xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        # loss = tf.reduce_mean(xentropy, name="loss")
        tf.summary.scalar('xentropy', xentropy)
    # tf.summary.scalar('loss', loss)

    with tf.name_scope("train"):
        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
        if use_momentum_optimizer:
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=use_nesterov)
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(xentropy)

    with tf.name_scope("accuracy"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train_mnist_fullyconnected', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test_mnist_fullyconnected')
    tf.global_variables_initializer().run()

    # Train the model, and also write summaries.
    # Every 10th step, measure test-set accuracy, and write test summaries
    # All other steps, run train_step on training data, & add training summaries
    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = mnist.train.next_batch(100)
        else:
            xs, ys = mnist.test.images, mnist.test.labels
        return {x: xs, y: ys}

    saver = tf.train.Saver()

    for i in range(FLAGS.max_steps):
        if i % 10 == 0: # record summaries and test-set accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at %s: %s' % (i, acc))
        else: # record train set summaries, and train
            if i % 100 == 99: # record execution stats
                run_ops = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, training_op],
                                      feed_dict=feed_dict(True),
                                      options=run_ops,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, training_op], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
    save_path = saver.save(sess, FLAGS.save_dir + '/mnist/fullyconnected')

def main(_):
    with tf.Graph().as_default():
        train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='Number of training steps')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/data/',
        help='Directory for storing input data')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='./models',
        help='Directory for storing models')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='./logs',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)