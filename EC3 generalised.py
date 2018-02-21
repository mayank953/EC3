# ------------------------ EC3 code ----------------------------------#
# based on paper by Dr. tanmoy chakraborty 

#by Mayank & pratham 

#==============================================================================
# #                           load the  basic libraries                          #
#==============================================================================
import numpy as np
#import scipy 
import pandas as pd
from copy import copy
import time
import os
from sklearn.cross_validation import train_test_split
from sklearn import datasets

#==============================================================================
#                            code starts                             #
#==============================================================================
# user defined function to change categories to number 

def cattoval(coln):
    for i in range(len(coln)):
        for j in range(L):
            if(coln[i] == cat[j]):
                coln[i] = j
    return coln

def cattoval2(coln):
    coln = coln.reset_index(drop=True)
    for i in range(len(coln)):
        for j in range(L):
            if(coln[i] == cat[j]):
                coln[i] = j
    return coln

#change the working dir if required
#os.chdir ("~")
#get working directory
os.getcwd()

# load the  data ( assumed preprocessed ) # remove column with id index etc. 
# the data should be in format pandas dataframe
data = pd.read_csv("iris.csv",header = 0)
#data = datasets.load_iris()
#copy the data into df to preserve the original data
df = copy(data)

""" if any column like id index ( generally first ) is to be removed
# as it causes problem during unsupervised modelling"""

#df.drop(df.columns[0], axis=1,inplace = True) 


#resetting the index with index number 
df.columns = np.arange(0,len(df.columns))
# enter the col number of target number
tar = int(input("Enter your target variable column number: "))

""" to print out the percentage distribution of target variable """
print (df.groupby(tar).size()/df.groupby(tar).size().sum())*100

# to select the size of sample 
size = float(input("enter the size of the sample in fraction: "))

X = copy(df)
y = X.pop(X.columns[tar])


# stratfied taken as default

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=size,stratify=y)

O = len(X_test)
L = len(np.unique(y))

cat = []
cat = list(np.unique(y))

#==============================================================================
# #--------           TRAIN TEST SPLIT DONE --------------#
#==============================================================================



#==============================================================================
# #                       Load ML libraries                                       #
#==============================================================================
#Import Library
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
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

#==============================================================================
# #                       SUPERVISED MACHINE LEARNING ALGORITHM START         #
#==============================================================================

Salgo = np.zeros(shape=(O, 0), dtype = np.int64)
grp = []
accuracy = []
names = []


models = []
models.append(('DT - gini ', tree.DecisionTreeClassifier(criterion='gini')))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('RF', RandomForestClassifier()))
models.append(('SGD', SGDClassifier(loss="hinge", penalty="l2")))
models.append(('NN ', MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)))
models.append(('GBM',GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)))
models.append(('SVM',svm.SVC()))
models.append(('ADB',AdaBoostClassifier(n_estimators=100)))

print("What classification algorithms you would like to choose ?")

print("Decision Tree :- 1")
print("Naive Bayes :- 2")
print("KNN :- 3")
print("Random Forest Tree :- 4")
print("SGD :- 5")
print("Neural Network :- 6")
print("Gradient Boost :- 7")
print("Support vector machine :- 8")
print("Adaboost :- 9")

n = int(input("Please the enter the corresponding number of classification algorithms (out of 9)"))

print("Decision Tree :- 1")
print("Naive Bayes :- 2")
print("KNN :- 3")
print("Random Forest Tree :- 4")
print("SGD :- 5")
print("Neural Network :- 6")
print("Gradient Boost :- 7")
print("Support vector machine :- 8")
print("Adaboost :- 9")

a = []
while len(a) < n:
    item = int(input("Enter your algorithm to the list: "))
    a.append(item)

print("Here are your chosen algorithms")
print(a)

def classifier(a):
    global Salgo 
    global accuracy
    global names
    global grp

    for i in a:
        if ( i ==5 or i == 6):
            i = i-1
            model = models[i][1]
            model.fit(X_train.as_matrix(),y_train.as_matrix())
            predicted = model.predict(X_test)
            acc = accuracy_score(y_test,predicted)
            predicted = cattoval(predicted)
            Salgo = np.c_[Salgo,predicted]
            names.append(models[i][0])
            accuracy.append(acc)
            grp.append(L)
#            print str(models[i][0])
#            print classification_report(y_test,predicted)            
        else:
            i = i-1
            model = models[i][1]
            model.fit(X_train, y_train)
            model.score(X_train, y_train)
            predicted= model.predict(X_test)
            acc = accuracy_score(y_test,predicted)
            predicted = cattoval(predicted)
            Salgo = np.c_[Salgo,predicted]
            names.append(models[i][0])
            accuracy.append(acc)
            grp.append(L)
#            print str(models[i][0])
#            print classification_report(y_test,predicted)   
            
classifier(a)
Salgo = np.array(Salgo,dtype = np.int64)



#==============================================================================
#--------------------       UNSUPERVISED MACHINE LEARNING ALGO      -------------------#
#==============================================================================
def silhouette(X,y):
    for n_cluster in range(2, 11):
        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        label = kmeans.labels_
        sil_coeff = silhouette_score(X, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    


Ualgo = np.zeros(shape=(O, 0),dtype = np.int64) 

Umodels = []
Umodels.append(('KNearest', KMeans(n_clusters=L, random_state=0)))
Umodels.append(('Affinity', AffinityPropagation(preference=-50)))
Umodels.append(('DBSCAN', DBSCAN(eps=0.9, min_samples=10)))
Umodels.append(('MeanShift', MeanShift(bin_seeding=True)))

print("What clustering algorithms you would like to choose ?")

print("Kmeans :- 1")
print("Affinity propagation :- 2")
print("DBSCAN :- 3")
print("Mean Shift :- 4")


n = int(input("Please the enter the corresponding numbers to apply the algorithms"))

print("Kmeans :- 1")
print("Affinity propagation :- 2")
print("DBSCAN :- 3")
print("Mean Shift :- 4")

b = []
while len(b) < n:
    item = int(input("Enter your algorithm to the list: "))
    b.append(item)

print("Here are your chosen algorithms number")
print(b)

"""for selecting no. of clusters using silhoutte method for Kmeans but as it gives 2 for the iris dataset
(which have 3 classes) hence it was discarded.  
silhouette(X,y)
"""

def clustering(b):
    global Ualgo
    global accuracy
    global names
    global grp
    
    for i in b:
        i = i-1
        model = Umodels[i][1]
        model.fit(X)
        labels = model.labels_
        index_test = X_test.index.values
        predicted = [labels[j] for j in index_test]
        n_clusters_ = len(np.unique(predicted))
        predicted = [((n_clusters_) - 1)  if x == -1 else x for x in predicted]
        Ualgo = np.c_[Ualgo,predicted]
        names.append(Umodels[i][0])
        grp.append(n_clusters_)
        accuracy.append(0)


clustering(b)

summary = pd.DataFrame(np.column_stack([names,accuracy,grp]), columns=["algorithm","accuracy","Grp"])
 
#==============================================================================
# # --------------------    Parameter initialisation  ---------------------------------#
#==============================================================================

N = len(X_test)                        #  defined
# L                            number of classes
C1= len(a)                    #no. of classifier
C2 = len(b)                  # no. of clusters
C = C1 + C2
G1 = sum(grp[:len(a)])
G2 = sum(grp[len(a):len(grp)])
G = G1+G2

Algo = np.column_stack((Salgo, Ualgo))


#==============================================================================
# # --------------------------- Matrix Formation ------------------------------------------------#
#==============================================================================

#------------------------------------------------------------------------------------------#
s = time.time()
print ("starting building matrices")

def normalise(A,n=1):  
    if n == 1 :
        A = A/A.sum(axis=1)[:,None]
    elif n == 0:
        A = A/A.sum(axis=0)[None,:]
    return A

#-------------------------------------------------------------------------------------#
#-------------K from A -------------# 
from numpy import linalg as LA
from math import sqrt
from copy import copy, deepcopy

def StochasticK(Ac,ep = 0.1):
    K = np.zeros((N, N))
    KM =copy(Ac)
    
#    KM = np.zero((int(Ac.shape[0]), int(Ac.shape[1])))

    while LA.norm(np.array(K)-np.array(KM)) / (N*N) > ep:
        K=deepcopy(KM)
        d=KM.sum(axis=1)
        for i in range(N):
            for j in  range(N):
                K[i][j] = KM[i][j] / d[i] 
#        K = normalise(KM,1)
        for i in range(N): 
            for j in  range(N): 
                K[i][j] = K[j][i]= sqrt(K[i][j]*K[j][i])
        print LA.norm(np.array(K)-np.array(KM)) / (N*N)
    return K 
#-----------------------------------------------------------------------------------#

#==============================================================================
# #-----------------------------------     MEMBERSHIP MATRIX ------------------------------------#
#==============================================================================
def getMemMat():
    MemMat = np.zeros((N, G))
    index = 0
    for k in range(C):
        for i in range(N):
            j = index +  Algo[i,k]
            MemMat[i][j] = 1
        index = index + grp[k]
    return MemMat


MemMat = getMemMat()                      #  NxG
MemDF = pd.DataFrame(MemMat)  
MemMat = normalise(MemMat,1)

#performing the column wise normalisation of matrix if required for iEC3 
#MemMat = normalise(MemMat,1)





#==============================================================================
# #-----------              Co-occurence Matrix                  -----------#
#==============================================================================

def Count(m,n):
    score = 0
    for k in range(C):    
        if ( Algo[m,k] == Algo[n,k]):
            score = score + 1
    return score
    
CoMat = np.zeros((N, N))        # N x N

for i in range(N):
    for j in range(i,N):
        value = Count(i,j)
        CoMat[i][j]=CoMat[j][i] = value



""" learning the approximate stochastic matrix from Ac to Kc """
""" running an infinite loop where the value of norm does not changes """
#CoMat = StochasticK(CoMat)

#--------------------  average object class matrix -------------------------------#
#--------------------only supervised algo is considered-------------------------#


def fun(m,n):
    score = 0
    for k in range(C1):
        if Algo[m,k] == n:
            score = score + 1
    return score

def getObjclass():
    Objclass = np.zeros((N, L))
    for i in range(N):
        for j in  range(L):
    #        temp = int (Salgo[i,j])
            value = fun(i,j)
            value2 = value/float(C1)
    #        Objclass[i][temp]= Objclass[i][temp] + 1/C1
            Objclass[i,j] = value2
    return Objclass
        
Objclass = getObjclass()

#----------------- average group class matrix --------------------------------#
def getGrpclass():
#    global MemDF
    Grpclass = np.zeros(shape=(G,L))
    for gno in range(G):
        idx = MemDF[MemDF[gno] == 1 ].index.tolist()
        tot = float(len(idx) * C1)
        for i in idx:
            for j in range(C1):
                score = Salgo[i,j] 
                Grpclass[gno,score]= (Grpclass[gno,score] + 1/tot )   # for average
    return Grpclass

Grpclass = getGrpclass()


#------------------- object - class matrix ------------------------------------#
"""
# condition satisfied   Fo >= 0 , |Fo i. | = 1  for every i in 1:n """

Fo = np.random.rand(N,L)

Fo = Fo/Fo.sum(axis=1)[:,None]




    
#------------------- Group - class matrix --------------------------------------#
# condition satisfied   Fg >= 0 , |Fg .j | = 1  for every j in 1:l

Fg = np.random.rand(G,L)

Fg = Fg/Fg.sum(axis=0)[None,:]




e = time.time()
print ("all matrices have been made ")
print (e-s)

#-------------------------------------------------------------------------------------------------------#

#==============================================================================
# # ---------------- All Matrices Built --------------------------------------------#
#==============================================================================

import time 
from numpy import linalg as LA
from math import sqrt
from copy import copy, deepcopy
# input Km Kc Yo Yg alpha beta gamma delta epsilon
# initialised Fo & Fg with condition preserved 
# output Fo ( N x l ) probability of each element N belonging to class l 
def getdiagonal (A, x):
    A = A.sum(axis = x)
    return np.diag(A)


Dm = getdiagonal(MemMat , 0)
one = np.ones((G,G))
Dmdash = getdiagonal(MemMat,1)
Dc     = getdiagonal(CoMat,0)
oneN = np.ones((N,N))
ideN = np.identity(N)
t = 1
    
def EC3(Fo , Fg , Km , Kc , Yo , Yg , alpha = 0.25 , beta =0.35, gamma = 0.35, delta = 0.35 , eps = 0.0001):
    
    global t
    
    Fot = copy(Objclass)
    
    while LA.norm(np.array(Fot)-np.array(Fo)) / (N*L) > eps :             #Fo = Fo(t-1)
        
        t = t + 1 
        print ("loop run")
        lhs = np.linalg.inv(2*delta*one + alpha * Dm)                     #GxG
        rhs = alpha * np.matmul(Km.transpose() , Fo) + 2*delta*Yg         #GxL
        Fg =  np.matmul(lhs,rhs)                                          # GxG x GxL = G x L
        
        a = alpha *Dmdash
        b = 2*beta*Dc
        c = beta* np.matmul(ideN, Kc)
        d= beta * np.matmul(oneN,Kc)
        e = 2*gamma*oneN
       
        
        s = a + b
         
        s = s - c
        s = s - d
        s = s + e
        lhs = np.linalg.inv(s)
        f = alpha * np.matmul(Km,Fg)
        g  = 2*gamma*Yo
        rhs = f + g
        
        Fo = copy(Fot)
        Fot = np.matmul(lhs,rhs)
        
        print (Fot)
    return Fot
  
startAlgo = time.time() 

#def EC3(Fo , Fg , Km , Kc , Yo , Yg , alpha = 0.25 , beta =0.35, gamma = 0.35, delta = 0.35 , eps = 0.0001):
MainMat = EC3(Fo,Fg,MemMat,CoMat,Objclass,Grpclass,0.25,0.35,0.35,0.05,0.0001)
    
Endalgo = time.time()
print  (Endalgo - startAlgo)
output = pd.DataFrame(MainMat)


#output.to_csv("outputmatrix.csv")

result = np.argmax(MainMat, axis=1)
y_tes = cattoval2(y_test)
y_tes = np.array(y_tes , dtype = np.int64)
print accuracy_score(y_tes,result)
#with open('accuracyEC3.txt','w') as f:
#    f.write ('%0.19f' %accuracy)
    
print (classification_report(y_tes,result))    

print (" completed ")


