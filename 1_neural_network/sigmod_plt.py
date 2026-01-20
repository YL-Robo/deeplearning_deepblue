import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-100, 100, 1000)
plt.plot(z, sigmoid(z))
plt.title("Sigmoid Function: Mapping (-inf, inf) to (0, 1)")
plt.grid(True)
plt.show()