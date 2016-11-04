import numpy as np
import matplotlib.pyplot as plt

# Load dataset
dataset_id = "1"

train = np.loadtxt('data/data' + dataset_id + '_train.csv')
x_train = train[:,0:2]
y_train = np.zeros((train.shape[0], 2))
for i, y_i in enumerate(train[:,2:3]):
    y_train[i, int((y_i+1)/2)] = 1.0

test = np.loadtxt('data/data' + dataset_id + '_test.csv')
x_test = test[:,0:2]
y_test = np.zeros((test.shape[0], 2))
for i, y_i in enumerate(test[:,2:3]):
    y_test[i, int((y_i+1)/2)] = 1.0

val = np.loadtxt('data/data' + dataset_id + '_validate.csv')
x_val = val[:,0:2]
y_val = np.zeros((val.shape[0], 2))
for i, y_i in enumerate(val[:,2:3]):
    y_val[i, int((y_i+1)/2)] = 1.0

# Build model
if True:
    import nnet
    import nnet.layers
    model = nnet.Model([
        nnet.layers.ReLU(2, 50),
        nnet.layers.ReLU(50, 50),
        nnet.layers.Softmax(50, 2)
    ], loss='xentropy')

    prev_score = 0.0
    new_score = model.score(x_train, y_train)
    while new_score > prev_score:
        model.fit(x_train, y_train, epochs=64)
        prev_score = new_score
        new_score = model.score(x_train, y_train)
        print("train", new_score)
        print("test", model.score(x_test, y_test))
        print("val", model.score(x_val, y_val))
        print("")
else:
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential()
    model.add(Dense(5, input_dim=2, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x_train, y_train)
    print("train", model.score(x_train, y_train))
    print("test", model.score(x_test, y_test))
    print("val", model.score(x_val, y_val))

# Plot results
def plotDecisionBoundary(X, Y, scoreFn, values, title = ""):
    # Plot the decision boundary. For that, we will asign a score to
    # each point in the mesh [x_min, m_max]x[y_min, y_max].
    plt.figure()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = max((x_max-x_min)/200., (y_max-y_min)/200.)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x)[0] for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    CS = plt.contour(xx, yy, zz, values, colors = 'red', linestyles = 'solid', linewidths = 2)
    plt.clabel(CS, fontsize=9, inline=1)

    plt.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = plt.cm.cool)
    plt.title(title)
    plt.axis('tight')

def predictOne(x_i):
	output = model.predict(np.array([x_i]))
	return output.flatten()
plotDecisionBoundary(x_train, train[:,2:3], predictOne, [0.5])
plt.tight_layout()
plt.show()
