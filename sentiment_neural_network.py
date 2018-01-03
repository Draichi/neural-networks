# u'll need to run `create_sentiment_featurements.py` before this code
import pickle
import numpy as np
import tensorflow as tf

train_x, train_y, test_x, test_y = pickle.load(open("sentiment_set.pickle", "rb"))

# one hot situation:
# 10 classes, 0-9
# 0 = [1,0,0,0,0,0,0,0,0,0]
# 1 = [0,1,0,0,0,0,0,0,0,0]
# 2 = [0,0,1,0,0,0,0,0,0,0]
# 3 = [0,0,0,1,0,0,0,0,0,0]...
# one hot, rest cold

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
# this is gonna go through batchs of 100 features 
# and feed 'em through our network at a time
# batch is like a 'slice'
batch_sizes = 100

# the matrix is
# height x width => none, the lenght of on specific train_x
x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
    # the weights are a tensorflow value and the value is a
    # tensor flow random normal with the shape specified by us.
    # (input data * weights) + biases
    # if all inputs are 0, the biases is there to help get 1
    hidden_1_layer = {
        'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
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
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_sizes
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])
                # c of cost
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_sizes
            print('Epoch:', epoch+1, '   completed out of:', hm_epochs, '   Loss:', epoch_loss)
        # tf.argmax is gonna return the index of maximum value in 
        # these arrays. And we're hoping that those index values are 
        # the same, both one hots.
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # tf.cast change the variable to a type (float)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # we're gona evaluate all of those accuracies
        print('Accuracy:', accuracy.eval({x: test_x, y: test_y}))
    
train_neural_network(x)