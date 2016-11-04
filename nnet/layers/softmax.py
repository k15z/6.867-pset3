import numpy as np

class Softmax:
	def __init__(self, input_dims, output_dims):
		self.input_dims = input_dims
		self.output_dims = output_dims
		self.weight = np.random.normal(0.0, 1.0 / np.sqrt(output_dims), (output_dims, input_dims))

	def _softmax(self, y):
		result = []
		for i, y_i in enumerate(y):
			e_y_i = np.exp(y_i - np.max(y_i))
			result.append(e_y_i / np.sum(e_y_i))
		return np.array(result)

	def _grad_softmax(self, y):
		# TODO: Actually compute the gradient...
		def approx(y_i):
			return 0.1
		approx = np.vectorize(approx)
		return approx(y)

	def feedforward(self, data):
		self.data = data
		num_samples, num_dims = data.shape
		y = []
		for i, x_i in enumerate(self.data):
			y.append(np.dot(self.weight, x_i))
		y = np.array(y)
		return self._softmax(y)

	def backpropagate(self, grad):
		ngrad = []
		for i, x_i in enumerate(self.data):
			partial = grad[i] * self._grad_softmax(np.dot(self.weight, x_i))
			self.weight += np.outer(partial, x_i)
			ngrad.append(np.dot(partial, self.weight))
		return np.array(ngrad)
