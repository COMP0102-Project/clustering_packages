# for this file to run successfully,
# 1) place CheXper-v1.0-small in the same directory as this file
# 2) CheXper-v1.0-small should also contain chexpert_weights.hdf5
from clustering import ClusteringPipeline, Evaluator
import matplotlib.pyplot as plt


# root, path_to_pretrained_chexpert_model, CSVPATH, chexnet_targets, n_images,
chexnet_targets = ['No Finding',
                   'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                   'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
                   'Support Devices']
# Visualising clusters
pipeline = ClusteringPipeline(root='./',
                              path_to_pretrained_chexpert_model='CheXpert-v1.0-small/chexpert_weights.hdf5',
                              CSVPATH='CheXpert-v1.0-small/valid.csv',
                              chexnet_targets=chexnet_targets,
                              n_images=20,)

pipeline.build_3d_clusters(n_clusters=3)

# evaluation
evaluator = Evaluator()
dataset = pipeline.get_dataset()
feature_vectors = pipeline.get_feature_vectors()

# purity
x, y = evaluator.get_purity_score(dataset, feature_vectors)

# show silhouette graph
plt.plot(x, y)
plt.ylabel("Purity")
plt.xlabel("Number of clusters")
plt.show()

# silhouette score
x, y = evaluator.get_silhouette_score(feature_vectors)

# show silhouette graph
plt.plot(x,y)
plt.ylabel("silhouette score")
plt.xlabel("Number of clusters")
plt.show()