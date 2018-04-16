from keras.models import Sequential
from keras.layers import Dense
import numpy
import pickle


##### to do ####
# - model.pickle
# - iterate trought some prediction data


# random seed for reproducibility
numpy.random.seed(7)

# loading load prima indians diabetes dataset, past 5 
# years of medical history
dataset = numpy.loadtxt("diabetes-dataset.csv", delimiter=",")
# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:8]
Y = dataset[:,8]

# create model, add dense layers one by one
# specifying activation funcion
model = Sequential()
# input layer requires input_dim param
# 8 => number of features(X)
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
# sigmoid instead of relu for final probability between 0 and 1
model.add(Dense(1, activation='sigmoid'))

#compíle the model, adam gradient descent (optimizer)
model.compile(loss="binary_crossentropy", optimizer="adam", 
metrics=['accuracy'])

# call the function to fit to the data (trainning the network)
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predict = model.predict(numpy.array([[1,89,66,23,94,28.1,0.167,21]]), verbose=1)
if predict[0] < 0.5:
  print('Descision: {}, U dont have diabetes :)'.format(predict[0]))
else:
  print('Descision: {}, U do have diabetes!'.format(predict[0]))

