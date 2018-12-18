import pickle
import sys
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
sys.path.insert(0, './../preprocessing')

import session as ss

def topK(K,centroid,data,X):
    data['distance'] = np.linalg.norm(centroid - X, axis=1)#euclidian(centroid,data['session'].values)
    data = data.sort_values(by=['distance'])
    return data.head(K)

df = pickle.load(open("./../data_set.p", "rb"))

# create session UUID
df = ss.define_session(df)

action_set = set(df['action'])

sessions = []
labels = []
for uuid, row in df.groupby('UUID'):
    session = np.array([1 if s in row['action'].values else 0 for s in action_set])
    sessions.append(session)

    l = row['action']
    labels.append(l)


X = np.array(sessions)
kmeans = KMeans(n_clusters=10).fit(X)

centroids = kmeans.cluster_centers_



ss = pd.DataFrame({'session':sessions})
ss['label'] = labels


for i, c in enumerate(centroids):
    top10 = topK(K=10, centroid=c, data=ss, X=X)
    for j,l in enumerate(top10['label']):
        print('cluster %d, neighbour %d' % (i,j))
        print(l.values,'\n')
    print()



