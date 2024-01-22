import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('CCPP_data.txt', delimiter='\t')

# Define the target variable and the feature variables
X = df[['AT', 'V', 'AP', 'RH']]
Y = df['EP']

# Define the maximum number of features to consider
max_features = 4

# Define the list of feature combinations to consider
feature_combinations = []
for i in range(1, max_features + 1):
    feature_combinations += list(combinations(X.columns, i))

# Define the list of adjusted R-squared values for each feature combination
adj_r2_values = []
for features in feature_combinations:
    reg = LinearRegression().fit(X[list(features)], Y)
    adj_r2_values.append(1 - (1 - reg.score(X[list(features)], Y)) * ((len(Y) - 1) / (len(Y) - len(features) - 1)))

# Plot the adjusted R-squared values versus the number of features
plt.plot(range(1, len(adj_r2_values) + 1), adj_r2_values)
plt.xlabel('Number of features')
plt.ylabel('Adjusted R-squared')
plt.show()

# Select the best feature combination
best_features = feature_combinations[np.argmax(adj_r2_values)]
print(f"The best feature combination is: {best_features}\n")

# Fit the linear regression model using the best feature combination
reg = LinearRegression().fit(X[list(best_features)], Y)

# Print the coefficient estimates and the coefficient of determination R^2
beta_0 = reg.intercept_
beta_i = reg.coef_
R2 = reg.score(X[list(best_features)], Y)
print(f"The coefficient estimates are: beta_0 = {beta_0:.2f}, beta_i = {beta_i}")
print(f"The coefficient of determination R^2 is: {R2:.4f}")
