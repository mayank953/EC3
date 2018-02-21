import numpy as np
import pandas as pd
import os
import math
from copy import copy

def cattoval(tarcol):
    for i in range(len(tarcol)):
        for j in range(l):
            if(tarcol[i] == o[j]):
                tarcol[i] = j
    return tarcol

# Load the data 

df = pd.read_csv("iris.csv",header=0)

# Duplicate the Data
from sklearn.cross_validation import train_test_split
tar = int(input("Enter your target variable column number"))

X = copy(df)
y = X.pop('tar')

l = len(np.unique(y))   #(X.iloc[:,e]))  # set of classes
y = cattoval(y)


O = len(y)   # set of N objects. 


size = int(input("enter the size of sample in percentage ( out of 100 )"))
size = size/100
ans = int(input("do you want stratify sampling ( y or n )"))

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=size,stratify=ans)





