import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the csv file into a DataFrame

data = pd.read_csv('dataset.txt', names=['AT', 'V', 'AP', 'RH', 'EP'], sep='\t')


# Check for missing values
missing_values = data.isnull().sum()

print(missing_values)

description_stats = data.describe()
print(description_stats)