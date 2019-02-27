import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")

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


x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
y = tf.placeholder(tf.int64, shape=(None), name="y")
training = tf.placeholder_with_default(False, shape=(), name="training")
x_drop = tf.layers.dropout(x, dropout_rate, training=training)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(x_drop, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
    hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, name="hidden2", activation=tf.nn.relu)
    hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
    logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)
    if use_momentum_optimizer:
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=use_nesterov)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={x: x_batch, y:y_batch})
        acc_train = accuracy.eval(feed_dict={x: x_batch, y:y_batch})
        acc_test = accuracy.eval(feed_dict={x: mnist.test.images, y:mnist.test.labels})
        print(epoch, "Train acc:", acc_train, "Test acc:", acc_test)
    save_path = saver.save(sess, "./model_final.ckpt")