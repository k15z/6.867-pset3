import numpy as np

def mse(expected, output):
	assert expected.shape == output.shape
	gradient = expected - output
	loss = np.sum(np.power(expected - output, 2))
	return (loss / expected.shape[0], gradient / expected.shape[0])

def xentropy(expected, output):
	assert expected.shape == output.shape
	gradient = expected / output
	loss = -np.sum(expected * np.log(output))
	return (loss / expected.shape[0], gradient / expected.shape[0])

class Model:
	def __init__(self, layers, loss='mse'):
		assert len(layers) > 0
		for i in range(1, len(layers)):
			assert layers[i-1].output_dims == layers[i].input_dims

		self.layers = layers
		if loss == 'mse':
			self.loss_fn = mse
		elif loss == 'xentropy':
			self.loss_fn = xentropy
		else:
			raise ValueError("Unsupported loss function.")

	def fit(self, x, y, epochs=10):
		assert x.shape[0] == y.shape[0]
		assert x.shape[1] == self.layers[0].input_dims
		assert y.shape[1] == self.layers[-1].output_dims

		losses = []
		for epoch in range(epochs):
			loss, gradient = self.loss_fn(y, self.predict(x))
			for layer in reversed(self.layers):
				gradient = layer.backpropagate(gradient)
			losses += [loss]
		return losses

	def score(self, x, y):
		wrong = 0.0
		total = 0.0
		output = self.predict(x)
		for i, y_i in enumerate(y):
			total += 1.0
			if np.argmax(y_i) != np.argmax(output[i]):
				wrong += 1.0
		return 1.0 - wrong / total

	def predict(self, x):
		assert x.shape[1] == self.layers[0].input_dims

		output = x
		for layer in self.layers:
			output = layer.feedforward(output)
		return output
