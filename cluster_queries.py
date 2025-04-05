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
#load all files to dict
for file in  files:
    #name = os.path.splitext(os.path.basename(file))[0]
    [c, idx, iou] = parse("queries/class{}_idx{}_iou{}.pt", file)
    d[c].append([file, float(iou)])

#perform kmeans ib latents
K = 3
for key, v_list in d.items():
    queries = []
    queries_files = []
    for v in v_list:
        q_file, q_iou = v
        #add queries with iou > thr
        if q_iou > 0.75:
            q = torch.load(q_file)
            queries.append(q.cpu().numpy())
            queries_files.append(q_file)
    #kmeans over queries
    X = np.array(queries)
    km = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(X)
    #km.cluster_centers_
    for i in range(K):        
        ind = np.argsort(km.transform(X)[:,i])[:1]
        filename = queries_files[ind[0]]#d[key][ind[0]]
        cluster_centers[key].append(filename)

print(cluster_centers)

filename = 'cluster_centers.p'
with open(filename, 'wb') as fp:
    pickle.dump(cluster_centers, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open('data.p', 'rb') as fp:
#     data = pickle.load(fp)



