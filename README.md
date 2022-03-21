# clustering_packages
API package for evaluating clusters so that it can be used by pother developers
This package allows you to call functions that will help in the clustering xray images particularly from
the chexpert dataset.
It allows for visualisation, dataset and model loading as well as evaluation of images seen in the 
chexpert dataset.

Evaluation is generic since it the get_purity_score and get_silhouette_score can take in any
feature vectors and the images associated with them.

---------------------------------------------------------------------------------------------
for example_usage.py to run successfully, 
1) place CheXper-v1.0-small in the same directory as this file
2) CheXper-v1.0-small should also contain chexpert_weights.hdf5
