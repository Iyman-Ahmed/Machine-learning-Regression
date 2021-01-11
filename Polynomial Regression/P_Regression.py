import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
Lin_reg = LinearRegression()
Lin_reg.fit(X, Y)

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)
Lin_reg_2 = LinearRegression()
Lin_reg_2.fit(X_poly, Y)

plt.scatter(X, Y, color='Red')
plt.plot(X, Lin_reg.predict(X), color="blue")
plt.title('Position Vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
#plt.show()

plt.scatter(X, Y, color='Red')
plt.plot(X, Lin_reg_2.predict(X_poly), color="blue")
plt.title('Position Vs Salary')
plt.xlabel('Position')
plt.ylabel('Salary')
#plt.show()

# Linear Regression Prediction
print(Lin_reg.predict([[6.5]]))
print(Lin_reg_2.predict(poly_regressor.fit_transform([[6.5]])))