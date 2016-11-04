import numpy as np

class Linear:
	def __init__(self, input_dims, output_dims):
		self.input_dims = input_dims
		self.output_dims = output_dims
		self.weight = np.random.normal(0.0, 1.0 / np.sqrt(input_dims * output_dims), (output_dims, input_dims))

	def feedforward(self, data):
		self.data = data
		num_samples, num_dims = data.shape
		y = []
		for i, x_i in enumerate(self.data):
			y.append(np.dot(self.weight, x_i))
		return np.array(y)

	def backpropagate(self, grad):
		for i, x_i in enumerate(self.data):
			self.weight += 1e-3 * np.outer(grad[i], x_i)
		return np.dot(grad, self.weight)
