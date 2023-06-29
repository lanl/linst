
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,1.,101)
y = np.linspace(0,5.,101)

xx, yy = np.meshgrid(x, y)

zz = xx**2. + yy**2

plt.contourf(xx, yy, zz, cmap = 'jet')
plt.colorbar()
plt.show()

