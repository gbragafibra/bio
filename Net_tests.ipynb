import numpy as np
import matplotlib.pyplot as plt

class Net_layer:
	def __init__(self, N_in, N_out, σ = False):
		# σ = True if using act function
		self.σ = σ
		self.W = np.random.randn(N_in, N_out)
		self.ε = np.random.randn(N_in, N_out)
	def forward(self, X):
		self.X = X

		self.X_ = np.dot(self.X.T, self.W)

		if self.σ:
			self.X_ = sigmoid(self.X_)

		return self.X_.T

	def update(self, η, T, λ):
		ΔW = η * np.outer(self.X, self.X_)
		self.W -= Hebbian_Metro(self.ε, T, ΔW)

		self.ε -= change_ε(self.X, self.X_) * λ

class Net:
	def __init__(self, layer_sizes, T, γ, β):
		"""
		T is temperature
		γ is fixed T decay
		β is fixed T increase according to Loss
		"""
		self.layers = [Net_layer(layer_sizes[i], layer_sizes[i + 1], \
		σ = (i == len(layer_sizes) - 2))\
		for i in range(len(layer_sizes) - 1)]

		self.T = T
		self.γ = γ
		self.β = β

	def forward(self, X):
		X_ = X
		for layer in self.layers:
			X_ = layer.forward(X_)
		return X_

	def update(self, η, λ):
		"""
		Update layers
		"""
		for layer in self.layers:
			layer.update(η, self.T, λ)

	def update_T(self, L):
		"""
		L is final loss
		"""
		self.T *= self.γ
		self.T += self.β * L
		#Don't want T < 0
		self.T = np.maximum(0, self.T)

	def train(self, X, y, η, λ, epochs, plot = False):
		losses = []
		y_ = []
		for epoch in range(epochs):
			Loss = 0
			for i in range(X.shape[0]):
				x_i = X[i].reshape(-1, 1)
				y_i = y[i].reshape(-1, 1)

				y_pred = self.forward(x_i)
				if epoch == epochs - 1:
					y_.append(y_pred.item())
				L = se(y_i, y_pred)
				Loss += L

				self.update_T(L)
				self.update(η, λ)

			μ_loss = Loss / X.shape[0]
			losses.append(μ_loss.item())
			print(f"Epoch {epoch + 1}/{epochs}, Loss: {μ_loss.item():.4f}")

		y_ = np.array(y_).reshape(-1, 1)
		Acc = np.sum(np.where(np.where(y_ > 0.5, 1, 0) == y, 1, 0), axis = 0)/X.shape[0]
		print(f"Training Accuracy: {Acc.item()}")
		if plot:
			plt.plot(range(1, epochs + 1), losses, "k")
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.show()

def change_ε(x_pre, x_pro):
	"""
	Assuming x_pre and x_pro with shape
	(N, N) or (N, 1)
	None with (N, )
	"""
	x_ = np.outer(x_pre, x_pro)
	Δε = (x_ - np.mean(x_))/np.std(x_)
	return Δε

def se(y, y_pred):
	"""
	Merely squared error loss
	Not considering batches
	"""
	return (y - y_pred)**2

def Hebbian_Metro(ε, T, ΔW):
	ξ = np.random.rand()
	return np.where(ξ < np.exp(-ε/T), ΔW, 0)

def sigmoid(x):
	return 1/(1 + np.exp(-x))


if __name__ == "__main__":
	np.random.seed(37)
	N = 10
	X = np.array([np.random.randint(0, 2, size=N).reshape(1, -1) for _ in range(20)])
	y = np.array([np.random.randint(0, 2) for _ in range(20)]).reshape(-1, 1)
	layer_sizes = [N, 20, 1]
	T = 50
	γ = 0.95
	β = 0.1
	net = Net(layer_sizes, T, γ, β)

	η = 0.1
	λ = 0.1
	epochs = 200
	net.train(X, y, η, λ, epochs, plot = True)