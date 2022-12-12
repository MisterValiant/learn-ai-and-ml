# Step 1: Do imports
import numpy as np
from sklearn.linear_model import LinearRegression

# Step 2: Input
x=np.array([5,15,25,35,45,55]).reshape((-1,1))
y=np.array([5,20,14,32,22,38])

# Content of arrays
print(x.shape)
print(y.shape)

# Step 3: Train the model
model=LinearRegression().fit(x,y)

# Q0
print('Interception Q0:', model.intercept_)
# Q1
print('slope Q1:', model.coef_)
# The model is not very accurate

# Prediction
y_pred=model.predict(x)
print('prediction for verification: ', y_pred, sep='\n')

x_new=np.array([4,10,80]).reshape((-1,1))
y_pred_new=model.predict(x_new)
print('new predictions: ', y_pred_new)