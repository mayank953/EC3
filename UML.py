from sklearn.cluster import DBSCAN

Ualgo = np.zeros(shape=(O, 0),dtype = np.int64) 
groups = []
#KMeans
def KMEANS():
    global Ualgo 
    k_means = KMeans(n_clusters=l, random_state=0)
    model = k_means
    model.fit(X_train)
    predictedKMeans = model.predict(X_test)
    Ualgo = np.c_[Ualgo,predictedKMeans]
    groups.append(l)
    print(k_means.predict(X_test))


#Affinity Propagation
def AFP():
    global Ualgo
    af = AffinityPropagation(preference=-50)
    af.fit(X)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    predictedAFP = af.predict(X_test)
    Ualgo = np.c_[Ualgo,predictedAFP]
    groups.append(n_clusters_)
    print('Estimated number of clusters: %d' % n_clusters_)

#DBSCAN
def DBS():
    global Ualgo
    db = DBSCAN(eps=0.5, min_samples=5)
    db.fit(X)
    labels = db.labels_
    index_test = X_test.index.values
    predictedDBS = [labels[i] for i in index_test]
    Ualgo = np.c_[Ualgo,predictedDBS]
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    groups.append(n_clusters_)
    print('Estimated number of clusters: %d' % n_clusters_)

#Mean Shift
def MNS(): 
    global Ualgo
    ms = MeanShift( bin_seeding=True)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    predictedMNS = ms.predict(X_test)
    Ualgo = np.c_[Ualgo,predictedMNS]
    groups.append(N_clusters_)
    print("number of estimated clusters : %d" % n_clusters_)
    
print("What clustering algorithms you would like to choose ?")

n = int(input("Please the enter the corresponding numbers to apply the algorithms"))

print("Kmeans :- 1")
print("Mean Shift :- 2")
print("DBSCAN :- 3")
print("Mean Shift :- 4")

b = []
while len(b) < n:
    item = int(input("Enter your algorithm to the list: "))
    b.append(item)

print("Here are your chosen algorithms")
print(b)
    
def USML():
    for i in range(len(b)):
        if(b[i] == 1):
            KMEANS()
        elif(b[i]== 2):
            MNS()
        elif(b[i]== 3):
            DBS()
#        elif(b[i] == 4):
#            MNS()
        else:
            print("invalid number")
    return 0
        
USML()
    
