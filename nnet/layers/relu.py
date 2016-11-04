import numpy as np

class ReLU:
	def __init__(self, input_dims, output_dims):
		self.input_dims = input_dims
		self.output_dims = output_dims
		self.weight = np.random.normal(0.0, 1.0 / np.sqrt(output_dims), (output_dims, input_dims))

	def _relu(self, y):
		return y * (y > 0)

	def _grad_relu(self, y):
		return (y > 0).astype('float')

	def feedforward(self, data):
		self.data = data
		num_samples, num_dims = data.shape
		y = []
		for i, x_i in enumerate(self.data):
			y.append(np.dot(self.weight, x_i))
		y = np.array(y)
		return self._relu(y)

	def backpropagate(self, grad):
		ngrad = []
		for i, x_i in enumerate(self.data):
			partial = grad[i] * self._grad_relu(np.dot(self.weight, x_i))
			self.weight += 1e-3 * np.outer(partial, x_i)
			ngrad.append(np.dot(partial, self.weight))
		return np.array(ngrad)
