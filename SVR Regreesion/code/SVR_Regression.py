# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y),1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting
#print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]]))))
# test_input = float(input('Enter Position to Find Estimated Salary = '))
sc_in = sc_X.transform([[6.5]])
sc_prediction = regressor.predict(sc_in)
prediction = sc_y.inverse_transform(sc_prediction)
print(prediction)

X_in = sc_X.inverse_transform(X)
y_in = sc_y.inverse_transform(y)
# Visualisation
plt.scatter(X_in, y_in, color='blue')
plt.plot(X_in, sc_y.inverse_transform(regressor.predict(X)) , color='red')
plt.title('Salary Vs Postion')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
