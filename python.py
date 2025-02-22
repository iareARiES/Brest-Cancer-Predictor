import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

dataset = pd.read_csv("breast_cancer.csv")
dataset.columns = dataset.columns.str.lower()

X = dataset.drop(columns =["class"])
y = dataset["class"]

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust width for better readability
pd.set_option('display.max_colwidth', None)  # Show full content of each column

print(dataset.head())
print(" ")
print(dataset.drop(columns =["sample code number","class"]).describe())
#print(dataset.to_string())  # or simply use for one time view Converts DataFrame to a string with full data

#visualisation
sns.set_style("whitegrid")
plt.figure(figsize=(15,15))
for i, column in enumerate(['clump thickness', 'uniformity of cell size','uniformity of cell shape', 'marginal adhesion',
                            'single epithelial cell size', 'bare nuclei', 'bland chromatin','normal nucleoli', 'mitoses'],1):
    plt.plot(i)
    sns.histplot(dataset[column],kde = True)
    plt.title(f"Distribution of {column}")
    plt.show()
