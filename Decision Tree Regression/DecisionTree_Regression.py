import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X,y)

prediction=regressor.predict([[6.5]])
print(prediction)


x_grid = np.arange(1, 10, 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(X,y,color='blue')
plt.plot(x_grid,regressor.predict(x_grid),color='red')
plt.title('Position Vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()