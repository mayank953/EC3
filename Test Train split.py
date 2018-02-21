##########################################################################

#Import Library
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import pandas as pd

#load the libraries 
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift

#################################################################################



import pandas as pd
from copy import copy
import time 
import numpy as np

def cattoval(tarcol):
    for i in range(len(tarcol)):
        for j in range(l):
            if(tarcol[i] == o[j]):
                tarcol[i] = j
    return tarcol

data = pd.read_csv("iris.csv", header = 0)
#copy the data 
df = copy(data)
df = df.drop('Id',axis =1)
# resetting the name of the columns  
df.columns = np.arange(0,len(df.columns))

# enter the col number of target number
tar = int(input("Enter your target variable column number: "))


# print percentage of each category 
print (df.groupby(tar).size()/df.groupby(tar).size().sum())*100

size = float(input("enter the size of the sample in fraction: "))

import numpy as np
from sklearn.cross_validation import train_test_split


X = copy(df)
y = X.pop(X.columns[tar])

#splitting 20% into Test & 80% onto Train
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20,stratify=y)
O = len(X_test)
l = len(np.unique(y))



del size,tar , df 
