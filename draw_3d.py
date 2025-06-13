import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') 
theta = np.linspace(-4 * np.pi , 4 * np.pi , 200)
z = np.linspace(-4 , 4 , 200) * 0.4

r = z ** 3 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x , y , z , label = 'parametric curve')
ax.legend()

plt.show()