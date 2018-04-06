from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

#Step 1: load data
dataset = numpy.loadtxt("letter_a_first_trial.txt", delimiter =",")
#split into input(X) and output(Y) variables
X = dataset[:,0:2]
Y = dataset[:,2]

#Step 2: create model 
model = Sequential()
model.add(Dense(12, input_dim=2, activation ='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#Step 3: compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['accuracy'])

#Step 4: fit the model
model.fit(X, Y, epochs=150, batch_size=10)

#Step 5: evaluate the model 
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#Step 6: calculate and round predictions
Z = numpy.loadtxt("letter_a_output.txt", delimiter=",")
predictions = model.predict(Z)
rounded = [x[0] for x in predictions]	
print(rounded)