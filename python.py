import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("breast_cancer.csv")

print(dataset.head())

X = dataset.drop(columns =["Class"])
y = dataset["Class"]


