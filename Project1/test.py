import numpy as np
import matplotlib.pyplot as plt

m = 200
x = np.linspace(0.8,1.8,10000)
y = 0.5 - 1/(1 + m*np.abs(x-1.2))

plt.plot(x,y)
plt.axhline(0)
plt.show()
