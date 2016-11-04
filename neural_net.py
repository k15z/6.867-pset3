import nnet
import nnet.layers
import numpy as np
import matplotlib.pyplot as plt
from plotBoundary import plotDecisionBoundary

data = np.loadtxt("data/data_3class.csv")
x = data[:,:2]
y = np.zeros((x.shape[0], 3))
for i, y_i in enumerate(data[:,2:]):
	y[i, int(y_i)] = 1.0


if True:
	model = nnet.Model([
		nnet.layers.ReLU(2, 5),
		nnet.layers.Softmax(5, 3)
	], loss='xentropy')
	print(model.fit(x, y, epochs=32))
else:
	import matplotlib.pyplot as plt
	from plotBoundary import plotDecisionBoundary
	from keras.models import Sequential
	from keras.layers import Dense, Activation

	model = Sequential()
	model.add(Dense(5, input_dim=2, activation='relu'))
	model.add(Dense(3, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	model.fit(x, y)

print(y)
print(model.predict(x))

def predictOne(x_i):
	output = model.predict(np.array([x_i]))
	return output[0,0]
plotDecisionBoundary(x, data[:,2], predictOne, [0.5])
def predictOne(x_i):
	output = model.predict(np.array([x_i]))
	return output[0,1]
plotDecisionBoundary(x, data[:,2], predictOne, [0.5])
def predictOne(x_i):
	output = model.predict(np.array([x_i]))
	return output[0,2]
plotDecisionBoundary(x, data[:,2], predictOne, [0.5])
plt.show()
