import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10,10))

x = np.linspace(-10, 10, 1000)

w = 2.1
c = -1.4
y1 = w * x + c
y2 = 0.45 * np.exp(1)**x

plt.plot(x, y2)
plt.grid('on')
plt.title(f'y = {w} * x + {c}')
plt.show()