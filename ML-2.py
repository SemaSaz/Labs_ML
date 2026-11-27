import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x1, x2):
    return 2 * x1 ** 2 + 3 * x2 ** 2


x1 = np.linspace(-3, 3, 31)
x2 = np.linspace(0, 2, 31)

rows = []
for a in x1:
    for b in x2:
        rows.append([a, b, f(a, b)])

data = pd.DataFrame(rows, columns=['x1', 'x2', 'y'])
data.to_csv('lab02_data.csv', index=False)

print('=== Пункт 1 ===')
print(data.shape)
print(data.head())

d = pd.read_csv('lab02_data.csv')

print()
print('=== Пункт 2 ===')
c2 = 2.0
c1 = 0.0
p1 = d[np.isclose(d['x2'], c2)]
p2 = d[np.isclose(d['x1'], c1)]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(p1['x1'], p1['y'], 'o-', color='C0')
ax[0].set_xlabel('x1')
ax[0].set_ylabel('y')
ax[0].set_title('y(x1), x2 = 2')
ax[0].grid(True)

ax[1].plot(p2['x2'], p2['y'], 'o-', color='C1')
ax[1].set_xlabel('x2')
ax[1].set_ylabel('y')
ax[1].set_title('y(x2), x1 = 0')
ax[1].grid(True)
plt.show()

print()
print('=== Пункт 3 ===')
for col in ['x1', 'x2', 'y']:
    print(col, 'mean =', round(d[col].mean(), 4),
          'min =', round(d[col].min(), 4),
          'max =', round(d[col].max(), 4))

print()
print('=== Пункт 4 ===')
m1 = d['x1'].mean()
m2 = d['x2'].mean()
new = d[(d['x1'] < m1) | (d['x2'] < m2)]
new.to_csv('lab02_filtered.csv', index=False)
print(new.shape)
print(new.head())

print()
print('=== Пункт 5 ===')
X, Y = np.meshgrid(x1, x2)
Z = f(X, Y)
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='viridis')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('y')
ax3.set_title('y(x1, x2)')
plt.show()