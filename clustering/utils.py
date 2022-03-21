
from sklearn.cluster import MiniBatchKMeans

def get_cluster_labels(feat_vecs, n_clusters=7):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(feat_vecs)
    return kmeans.labels_