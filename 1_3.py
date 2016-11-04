import nnet
import nnet.layers
import numpy as np

# Load dataset
x_train, y_train = [], []
x_test, y_test = [], []
x_val, y_val = [], []
for i in range(10):
    data = np.loadtxt('data/mnist_digit_' + str(i) + '.csv')

    x_train.append(data[0:200] * 2.0 / 255.0 - 1.0)
    arr = np.zeros((200, 10))
    arr[:,i] = 1.0
    y_train.append(arr)

    x_test.append(data[200:350] * 2.0 / 255.0 - 1.0)
    arr = np.zeros((150, 10))
    arr[:,i] = 1.0
    y_test.append(arr)

    x_val.append(data[350:500] * 2.0 / 255.0 - 1.0)
    arr = np.zeros((150, 10))
    arr[:,i] = 1.0
    y_val.append(arr)

x_train, y_train = np.vstack(x_train), np.vstack(y_train)
x_test, y_test = np.vstack(x_test), np.vstack(y_test)
x_val, y_val = np.vstack(x_val), np.vstack(y_val)

# Build model
model = nnet.Model([
    nnet.layers.ReLU(784, 50),
    nnet.layers.Softmax(50, 10)
], loss='xentropy')
prev_score = 0.0
new_score = model.score(x_train, y_train)
while new_score >= prev_score:
    model.fit(x_train, y_train, epochs=64)
    prev_score = new_score
    new_score = model.score(x_train, y_train)
    print("train", new_score)
    print("test", model.score(x_test, y_test))
    print("val", model.score(x_val, y_val))
    print(model.predict(x_train)[0])
    print(model.predict(x_train)[500])
    print("")
