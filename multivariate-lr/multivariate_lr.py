import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)
print(x)
print(y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

x_new = np.arange(10).reshape((-1, 2))
print(x_new)
y_new = model.predict(x_new)
print(y_new)
df_s = pd.DataFrame(x, columns=['x1','x2'])
df_s['y'] = pd.Series(y)
print(df_s)
#visualisation
x_surface, y_surface = np.meshgrid(np.linspace(df_s.x1.min(), df_s.x1.max(), 100)
,np.linspace(df_s.x2.min(), df_s.x2.max(), 100))
only_x = pd.DataFrame({'x1':x_surface.ravel(), 'x2':y_surface.ravel()})
y_pred = model.predict(only_x)
y_pred=np.array(y_pred)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_s['x1'],df_s['x2'],df_s['y'],c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surface,y_surface,y_pred.reshape(x_surface.shape), color='b', alpha=0.3)
ax.set_xlabel('x1')

ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.show()