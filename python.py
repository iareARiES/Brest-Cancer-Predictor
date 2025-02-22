import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score

dataset = pd.read_csv("breast_cancer.csv")
dataset.columns = dataset.columns.str.lower()


X = dataset.drop(columns =["class"])
y = dataset["class"]


pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust width for better readability
pd.set_option('display.max_colwidth', None)  # Show full content of each column

#seeing the data and reading it
print(dataset.head())
print(" ")
print(dataset.drop(columns =["sample code number","class"]).describe())
#print(dataset.to_string())  # or simply use for one time view Converts DataFrame to a string with full data
print(f"\nData set information\n: {dataset.info()}")


#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


#dropping this as we don't want this to scale
X_train = X_train.drop(columns=(["sample code number"]))
X_test = X_test.drop(columns=(["sample code number"]))


# Standard scaling the dataset
sc = StandardScaler()
X_train =sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X.columns,X.shape)
print(X_train.shape)


#converting back to pandas dataframe
X_train= pd.DataFrame(X_train, columns=['clump thickness', 'uniformity of cell size',
       'uniformity of cell shape', 'marginal adhesion',
       'single epithelial cell size', 'bare nuclei', 'bland chromatin',
       'normal nucleoli', 'mitoses'])
X_test= pd.DataFrame(X_test, columns =X_train.columns)

#checaking the standard scaling
print(X_train.head())
print(X_test.head())


#Training the model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#making the confusion matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)

#checking the scores
print("The r2 score is : ",r2_score(y_test, y_pred))
