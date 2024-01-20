import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset into a DataFrame
df = pd.read_csv('dataset.txt', sep='\t')

# Determine the number of columns in the DataFrame
num_columns = df.shape[1]

# Create a figure with a subplot for each column
fig, axes = plt.subplots(nrows=num_columns, ncols=1, figsize=(10, num_columns * 5))

# Create individual box plots for each column
for i, column in enumerate(df.columns):
    df.boxplot(column=column, ax=axes[i])
    axes[i].set_title(f'Box Plot of {column}')
    axes[i].set_ylabel('Values')

plt.tight_layout()
plt.show()
