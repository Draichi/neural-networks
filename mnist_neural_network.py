import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# one hot situation:
# 10 classes, 0-9
# 0 = [1,0,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0,0]
# 2 = [0,0,1,0,0,0,0,0,0,0]
# 3 = [0,0,0,1,0,0,0,0,0,0]...
# one hot, rest cold
mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
# this is gonna go through batchs of 100 features 
# and feed 'em through our network at a time
batch_sizes = 100

# the matrix is
# height x width => none, 784 (28*28)
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    # the weights are a tensorflow value and the value is a
    # tensor flow random normal with the shape specified by us.
    # (input data * weights) + biases
    # if all inputs are 0, the biases is there to help get 1
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }
    hidden_2_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }
    hidden_3_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))
    }
    output_layer = {
        'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        # number of biases on our output_layer = number of classes
        'biases': tf.Variable(tf.random_normal([n_classes]))
    }

    # (input data * weights) + biases

    # matmul = matrix multiply
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # relu = rectified linear = activation function, threshold function
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    # one hot array. ex.: [0,0,0,1,0,0,1,1,1,0,1,0,0,0,]
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    #cross entropy with logics will calculate the diference of prediction and the known label
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    # we wanna minimize that cost because we 
    # wanna minimize the diference of predicition and y.
    # AdamOptimizer is like a Stochastic Gradient Descent
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # cycles of feef forward + backprop
    hm_epochs = 10

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
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    
train_neural_network(x)