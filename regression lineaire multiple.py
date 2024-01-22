import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('CCPP_data.txt', sep='\t')

# Define the target variable and predictors
y = data['EP']
X = data.drop('EP', axis=1)

# Store the results
results = []

# Iterate over all possible combinations of predictors (from 1 to 4)
for k in range(1, 5):
    for variables in itertools.combinations(X.columns, k):
        # Fit the model
        X_subset = sm.add_constant(X[list(variables)])
        model = sm.OLS(y, X_subset)
        results.append({
            'variables': variables,
            'model': model.fit()
        })

# Find the best model (the one with the highest adjusted R-squared)
best_model = max(results, key=lambda item: item['model'].rsquared_adj)

# Print the best model's details
print('Best model variables:', best_model['variables'])
print('Best model adjusted R-squared:', best_model['model'].rsquared_adj)
print('Best model coefficients:', best_model['model'].params)

# Plot the adjusted R-squared versus the number of features
plt.plot([len(item['variables']) for item in results], [item['model'].rsquared_adj for item in results], 'ro-')
plt.xlabel('Number of features')
plt.ylabel('Adjusted R-squared')
plt.show()

# Perform the zero slope hypothesis test for all the coefficients except β0
print('Zero slope hypothesis test results:')
for variable in best_model['variables']:
    if variable in best_model['model'].params.index:
        print(f'{variable}:', best_model['model'].pvalues[variable])

