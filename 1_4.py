import tqdm
import nnet
import nnet.layers
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2, activity_l2

# Load dataset
x_train, y_train = [], []
x_test, y_test = [], []
x_val, y_val = [], []
for i in range(10):
    data = np.loadtxt('data/mnist_digit_' + str(i) + '.csv')

    x_train.append(data[0:500] * 2.0 / 255.0 - 1.0)
    arr = np.zeros((500, 10))
    arr[:,i] = 1.0
    y_train.append(arr)

    x_test.append(data[500:650] * 2.0 / 255.0 - 1.0)
    arr = np.zeros((150, 10))
    arr[:,i] = 1.0
    y_test.append(arr)

    x_val.append(data[650:800] * 2.0 / 255.0 - 1.0)
    arr = np.zeros((150, 10))
    arr[:,i] = 1.0
    y_val.append(arr)

x_train, y_train = np.vstack(x_train), np.vstack(y_train)
x_test, y_test = np.vstack(x_test), np.vstack(y_test)
x_val, y_val = np.vstack(x_val), np.vstack(y_val)

from keras.layers import Dense, Dropout
from keras.models import Sequential
def test_model(num_hidden):
    train_acc = []
    test_acc = []
    val_acc = []
    
    for _ in range(3):
        model = Sequential()
        model.add(Dense(num_hidden, input_dim=784, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(x_train, y_train, verbose=False)
        train_acc.append(model.evaluate(x_train, y_train, verbose=False)[1])
        test_acc.append(model.evaluate(x_test, y_test, verbose=False)[1])
        val_acc.append(model.evaluate(x_val, y_val, verbose=False)[1])
    train_acc = sum(train_acc) / 3.0
    test_acc = sum(test_acc) / 3.0
    val_acc = sum(val_acc) / 3.0

    return (train_acc, test_acc, val_acc)

x = []
y_train_acc = []
y_test_acc = []
y_val_acc = []
for num_hidden in tqdm.tqdm([1, 5, 10] + range(50, 500+1, 25)):
    x.append(num_hidden)
    train_acc, test_acc, val_acc = test_model(num_hidden)
    y_train_acc.append(train_acc)
    y_test_acc.append(test_acc)
    y_val_acc.append(val_acc)

plt.plot(x, y_train_acc, label='train')
plt.plot(x, y_test_acc, label='test')
plt.plot(x, y_val_acc, label='val')
plt.xlabel("# hidden nodes")
plt.ylabel("accuracy")
plt.tight_layout()
plt.legend(loc='lower right')
plt.show()
