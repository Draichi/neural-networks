# u'll need to run `create_sentiment_featurements.py` before this code
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#train_x, train_y, test_x, test_y = pickle.load(open("sentiment_set.pickle", "rb"))

n_nodes_hl1 = 500
n_nodes_hl2 = 500

n_classes = 2
# 200 data? 53% accuracy (10 epochs)
# 2000 data? 62% accuracy (10 epochs) 9-10seconds on GPU tf. 14 seconds on CPU
# 2000 data? 63% (15 epochs)
# 200.000 data? 74% (15 epochs)
hm_data = 200000

batch_sizes = 32
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

current_epoch = tf.Variable(1)

# 1 hidden layer = linear data
# 2 hidden layers = nonlinear data
# 3 hidden layers = super nonlinear data
# more layer will overfit your problem
hidden_1_layer = {
    'f_fum': n_nodes_hl1,
    'weight': tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
    'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }
hidden_2_layer = {
    'f_fum': n_nodes_hl2,
    'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }
output_layer = {
    'f_fum': None,
    'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
    'bias': tf.Variable(tf.random_normal([n_classes]))
}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    output = tf.add(tf.matmul(l2, output_layer['weight']), output_layer['bias'])
    return output

# save and store your tensor flow model
# before you call this, u need to have some tf variables defined 
saver = tf.train.Saver()
# purely to log the epochs
tf_log = 'tf.log'

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # this try to read the tf_log file just to see what was the last epoch
        # and if we cant read that for whatever reazon we say epoch = 1
        try:
            epoch = int(open(tf_log, 'r').read().split('\n')[-2])+1
            print('STARTING:', epoch)
        except:
            epoch = 1
        # here we begin interating throu all our epochs
        # sometimes people iterate thro infinite epochs, not this time
        while epoch <= hm_epochs:
            if epoch != 1:
                saver.restore(sess, "./model.ckpt")
            epoch_loss = 1
            with open('lexicon.pickle', 'rb') as f:
                lexicon = pickle.load(f)
                print('lexicon lenght:', len(lexicon))
            with open('train_set_shuffled.csv', buffering=20000, encoding='latin-1') as f:
                counter = 0
                for line in f:
                    counter =+ 1
                    label = line.split(':::')[0]
                    tweet = line.split(':::')[1]
                    current_words = word_tokenize(tweet.lower())
                    current_words = [lemmatizer.lemmatize(i) for i in current_words]
                    features = np.zeros(len(lexicon))
                    for word in current_words:
                        if word.lower() in lexicon:
                            index_value = lexicon.index(word.lower())
                            features[index_value] += 1
                    batch_x = np.array([list(features)])
                    batch_y = np.array([eval(label)])
                    # saver.save
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c
                    if counter > hm_data:
                        print('reached', hm_data, 'data, breaking!')
                        break
            # everytime we work with it we will save this model dot checkpoint
            saver.save(sess, "./model.ckpt")
            # saves per epoch and print
            print('Epoch', epoch+1, 'completed out of:', hm_epochs, 'loss:', epoch_loss)
            # now we write this in the log file
            with open(tf_log, 'a') as f:
                f.write(str(epoch)+'\n')
            epoch += 1
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        feature_sets=[]
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested', counter, 'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:', accuracy.eval({x: test_x, y:test_y}))
#train_neural_network(x)

def test_neural_network():
    prediction = neural_network_model(x)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            try:
                saver.restore(sess,"./model.ckpt")
            except Exception as e:
                print(str(e))
            epoch_loss = 0
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        feature_sets = []
        labels = []
        counter = 0
        with open('processed-test-set.csv', buffering=20000) as f:
            for line in f:
                try:
                    features = list(eval(line.split('::')[0]))
                    label = list(eval(line.split('::')[1]))
                    feature_sets.append(features)
                    labels.append(label)
                    counter += 1
                except:
                    pass
        print('Tested',counter,'samples.')
        test_x = np.array(feature_sets)
        test_y = np.array(labels)
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

test_neural_network()