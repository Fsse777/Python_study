import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def f_t(x,t):
    return 5 * np.cos(x) - np.cos(5 * x + t)

def max_f_given_t(t):
    x_vals = np.linspace(0 , 2 * np.pi , 2000)
    y_vals = f_t(x_vals , t)
    return np.max(y_vals)

result = minimize_scalar(max_f_given_t , bounds=(0,2*np.pi),method = 'bounded')
optimal_t = result.x
minimal_max = result.fun

print(f"t = {optimal_t:.6f} rad")
print(f"b = {minimal_max:.6f}")

x_vals = np.linspace(0, 2*np.pi,2000)
y_vals = f_t(x_vals,optimal_t)

plt.figure(figsize = (10,6))
plt.plot(x_vals,y_vals,label = f'Optimal_t:.4f')
plt.axhline(minimal_max,color = 'red',linestyle = '--',label=f'Max f_t(x) = {minimal_max:.4f}')
plt.title(r'$f_t(x) = 5\cos x -\cos(5x + t)$ at optimal $t$')
plt.xlabel('x')
plt.ylabel('f_t(x)')
plt.legend()
plt.grid(True)
plt.show()
