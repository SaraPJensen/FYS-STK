import numpy as np

x = np.linspace(0, 10, 11)
t = np.linspace(0, -10, 11)

x, t = np.meshgrid(x, t)

print(x)
