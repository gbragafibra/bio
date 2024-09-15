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
		self.W -= Hebbian_Metro(self.ε, T, ΔW, rev=False)

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

	def run(self, X, y, η, λ, epochs,
	train_ω, test_ω, plot = False):
		#Splitting sets
		idxs = np.random.permutation(X.shape[0])
		train_size = int(X.shape[0] * train_ω)
		#test_size = int(X.shape[0] * test_ω)

		train_idxs = idxs[train_size:]
		test_idxs = idxs[:train_size]
		X_train, y_train = X[train_idxs], y[train_idxs]
		X_test, y_test = X[test_idxs], y[test_idxs]


		train_losses = []
		train_y = []
		test_losses = []
		test_y = []
		for epoch in range(epochs):
			Loss_train = 0
			Loss_test = 0
			#train
			for i in range(X_train.shape[0]):
				x_i = X_train[i].reshape(-1, 1)
				y_i = y_train[i].reshape(-1, 1)

				y_pred = self.forward(x_i)
				if epoch == epochs - 1:
					train_y.append(y_pred.item())
				L_train = se(y_i, y_pred)
				Loss_train += L_train

				self.update_T(L_train)
				self.update(η, λ)
			#test
			for i in range(X_test.shape[0]):
				x_i = X_test[i].reshape(-1, 1)
				y_i = y_test[i].reshape(-1, 1)

				y_pred = self.forward(x_i)
				if epoch == epochs - 1:
					test_y.append(y_pred.item())
				L_test = se(y_i, y_pred)
				Loss_test += L_test

			μ_loss_train = Loss_train / X_train.shape[0]
			μ_loss_test = Loss_test / X_test.shape[0]
			train_losses.append(μ_loss_train.item())
			test_losses.append(μ_loss_test.item())
			print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {μ_loss_train.item():.4f},\
				Testing Loss: {μ_loss_test.item():.4f}")

		train_y = np.array(train_y).reshape(-1, 1)
		test_y = np.array(test_y).reshape(-1, 1)
		Acc_train = np.sum(np.where(np.where(train_y > 0.5, 1, 0) == y_train, 1, 0), axis = 0)/X_train.shape[0]
		Acc_test = np.sum(np.where(np.where(test_y > 0.5, 1, 0) == y_test, 1, 0), axis = 0)/X_test.shape[0]
		print(f"Training Accuracy: {Acc_train.item()}, Testing Accuracy: {Acc_test.item()}")
		if plot:
			plt.plot(range(1, epochs + 1), train_losses, "k", label = "Train Loss")
			plt.plot(range(1, epochs + 1), test_losses, "r", label = "Test Loss")
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.legend()
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

def Hebbian_Metro(ε, T, ΔW, rev = False):
	if rev:
		ξ = np.random.rand()
		χ = np.random.uniform(0, 1-ξ)
		return np.where(ξ < np.exp(-ε/T), ΔW, \
			np.where(χ < np.exp(-ε/T), -ΔW, 0))
	else:
		ξ = np.random.rand()
		return np.where(ξ < np.exp(-ε/T), ΔW, 0)

def sigmoid(x):
	return 1/(1 + np.exp(-x))


if __name__ == "__main__":
	np.random.seed(37)
	N = 10
	X = np.array([np.random.randint(0, 2, size=N).reshape(1, -1) for _ in range(100)])
	y = np.array([np.random.randint(0, 2) for _ in range(100)]).reshape(-1, 1)
	layer_sizes = [N, 20, 1]
	T = 50
	γ = 0.95
	β = 0.1
	net = Net(layer_sizes, T, γ, β)

	η = 0.1
	λ = 0.1
	epochs = 200
	net.run(X, y, η, λ, epochs, 0.7, 0.3, plot = True)