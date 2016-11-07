import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.loadtxt("data/data_3class.csv")
x = data[:,:2]
y = np.zeros((x.shape[0], 3))
for i, y_i in enumerate(data[:,2:]):
	y[i, int(y_i)] = 1.0

# Build model
if False:
    import nnet
    import nnet.layers
    model = nnet.Model([
        nnet.layers.ReLU(2, 5),
        nnet.layers.ReLU(5, 10),
        nnet.layers.Softmax(10, 3)
    ], loss='xentropy')

    prev_score = 0.0
    new_score = model.score(x, y)
    while new_score > prev_score:
        model.fit(x, y, epochs=64)
        prev_score = new_score
        new_score = model.score(x, y)
        print(new_score)
    print("Done.")
else:
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential()
    model.add(Dense(1, input_dim=2, activation='relu'))
#    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(x, y)
    print(model.evaluate(x, y))

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

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x)[1] for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    CS = plt.contour(xx, yy, zz, values, colors = 'green', linestyles = 'solid', linewidths = 2)
    plt.clabel(CS, fontsize=9, inline=1)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    zz = np.array([scoreFn(x)[2] for x in np.c_[xx.ravel(), yy.ravel()]])
    zz = zz.reshape(xx.shape)
    CS = plt.contour(xx, yy, zz, values, colors = 'blue', linestyles = 'solid', linewidths = 2)
    plt.clabel(CS, fontsize=9, inline=1)

    plt.scatter(X[:, 0], X[:, 1], c=(1.-Y), s=50, cmap = plt.cm.cool)
    plt.title(title)
    plt.axis('tight')

def predictOne(x_i):
	output = model.predict(np.array([x_i]))
	return output.flatten()
plotDecisionBoundary(x, data[:,2], predictOne, [0.5])
plt.tight_layout()
plt.show()
