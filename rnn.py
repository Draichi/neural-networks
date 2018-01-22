import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# one hot situation:
# 10 classes, 0-9
# 0 = [1,0,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0,0]
# 2 = [0,0,1,0,0,0,0,0,0,0]
# 3 = [0,0,0,1,0,0,0,0,0,0]...
# one hot, rest cold
mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

hm_epochs = 3
n_classes = 10
# this is gonna go through batchs of 128 features 
# and feed 'em through our network at a time
batch_sizes = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    # the weights are a tensorflow value and the value is a
    # tensor flow random normal with the shape specified by us.
    # (input data * weights) + biases
    # if all inputs are 0, the biases is there to help get 1
    layer = {
        'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }
    
    # formating data
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn.BasicLSTMCell(rnn_size)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

    # one hot array. ex.: [0,0,0,1,0,0,1,1,1,0,1,0,0,0,]
    return output

def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    #cross entropy with logics will calculate the diference of prediction and the known label
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # we wanna minimize that cost because we 
    # wanna minimize the diference of predicition and y.
    # AdamOptimizer is like a Stochastic Gradient Descent
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # cycles of feef forward + backprop

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # this for loop just train the network
        # once we have optimize those weights we 
        # run 'em thro our model
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_sizes)):
                # chunks through the dataset.
                # in a real work we'll need to create our own function
                # this is just a helper
                epoch_x, epoch_y = mnist.train.next_batch(batch_sizes)
                epoch_x = epoch_x.reshape((batch_sizes, n_chunks, chunk_size))
                # c of cost
                _, c = sess.run([optimizer, cost], feed_dict={x:epoch_x, y:epoch_y})
                epoch_loss += c
            print('Epoch:', epoch, '   completed out of:', hm_epochs, '   Loss:', epoch_loss)
        # tf.argmax is gonna return the index of maximum value in 
        # these arrays. And we're hoping that those index values are 
        # the same, both one hots.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # tf.cast change the variable to a type (float)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # we're gona evaluate all of those accuracies
        print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))
    
train_neural_network(x)