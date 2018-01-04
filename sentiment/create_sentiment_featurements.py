# pip3 intall nltk
# python3
# >>> import nltk
# >>> ntlk.download()
# >>> d
# >>> all

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000

def create_lexicon(pos, neg):
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f:
            contents = f.readlines()
            # : = up to
            for l in contents[:hm_lines]:
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words)
    # this is our way to stemming them into legitimate words
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    # ex.: w_counts = {'the':53154,'and':45632}
    w_counts = Counter(lexicon)
    l2 =[]
    for w in w_counts:
        # we dont want words with too much apperances 
        # because normally they are useless (the, and)
        if 1000 > w_counts[w] > 50:
            l2.append(w)
    print('Lenght of lexicon:',len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    # [
    #    [[features], [labels]]
    #    [[0 1 0 1 1 0 0 1],[1 0]],
    #    [[0 1 0 1 1 0 0 1],[0 1]]
    #    [[each index of the like at the bag of words model],[first element is the posistivity, and the second is the negativity]]
    # ]
    featureset = []
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    # we're gonna search our lexicon for that word.lower 
                    # and this will return our index value there
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
            features = list(features)
            featureset.append([features, classification])
    return featureset

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])
    # we need to shuffle to test, for statistics rasons, and because of neural networks 
    # the question is: does tf.argmax([output]) == tf.argmax([expextation]) ? 
    random.shuffle(features)
    features = np.array(features)
    # 10% of our features
    testing_size = int(test_size*len(features))
    # [:,0] will give us all the zeroth elemnts (all features)
    # [:-testing_size] up to the last 10%
    train_x = list(features[:,0][:-testing_size])
    # [:,1] will give us all the index 1 elemnts (all labels)
    train_y = list(features[:,1][:-testing_size])
    # the tests will be made with the last 10%, testing_sizes to the very end
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    return train_x, train_y, test_x, test_y
    
if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        # this create a pickle with all this info for us 
        # because we just need to run this one time and pass to our neural network
        pickle.dump([train_x, train_y, test_x, test_y], f)