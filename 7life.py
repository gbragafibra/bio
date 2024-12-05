"""
Continuous version of variant of GoL as seen and dissussed in
https://homepages.math.uic.edu/~kauffman/ReflexPublished.pdf
(Section 11)
with neighbors coming alive with 3 or 7 live neighbors.
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d


N = 200
A = np.zeros((N, N))
A[80:120, 80:120] = np.random.rand()
iterations = 200

#Concentric Kernel
α = 4
r = 20
kernel_size = 2*r + 1

x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
d = np.sqrt((x - r)**2 + (y - r)**2)

r_norma = d/r
#verify which fall out of the radius
mask = d <= r 
K = np.zeros_like(r_norma)
K[mask] = np.exp(α - (α/(4 * r_norma[mask] * (1 - r_norma[mask]) + 1e-6)))
K /= K.sum()


### Params for growth function
μ1 = 0.3 # ~3/9
σ1 = 0.07
μ2 = 0.77 # ~7/9
σ2 = 0.01

G = lambda u, μ1, σ1, μ2, σ2: 2 * np.exp(-(u - μ1)**2 / (2*σ1**2)) - 1 + 2 * np.exp(-(u - μ2)**2 / (2*σ2**2))
dt = 0.1 #arbitrary

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
mat = ax.imshow(A, cmap="hot", vmin=0, vmax=1)
ax.axis("off")

def update(frame, *args):
	global A 
	if frame % 11 == 0:
		print(f"Iteration {frame + 1}/{iterations}")

	A += dt * G(convolve2d(A, K, mode = "same", boundary = "wrap"), μ1, σ1, μ2, σ2)
	A = np.clip(A, a_min = 0, a_max = 1)

	mat.set_array(A)
	return mat


ani = FuncAnimation(fig, update, frames=iterations, interval=1000)
ani.save("life.gif", writer="pillow", fps=30)