import os
import torch
import glob
from parse import parse
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
import pickle


files = glob.glob('queries/*.pt')
d = defaultdict(list)
cluster_centers = defaultdict(list)
for file in  files:
    #name = os.path.splitext(os.path.basename(file))[0]
    [c, idx, iou] = parse("queries/class{}_idx{}_iou{}.pt", file)
    d[c].append(file)

K = 3
for key, v_list in d.items():
    queries = []
    for v in v_list:
        q = torch.load(v)
        queries.append(q.cpu().numpy())
    #kmeans over queries
    X = np.array(queries)
    km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
    #km.cluster_centers_
    for i in range(K):        
        ind = np.argsort(km.transform(X)[:,i])[:1]
        filename = d[key][ind[0]]
        cluster_centers[key].append(filename)

print(cluster_centers)

filename = 'cluster_centers.p'
with open(filename, 'wb') as fp:
    pickle.dump(cluster_centers, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('data.p', 'rb') as fp:
#     data = pickle.load(fp)



