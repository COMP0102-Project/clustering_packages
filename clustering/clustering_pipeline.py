import numpy as np
import cv2
import os
from sklearn.manifold import TSNE
from .dataset_loader import ChexpertDatasetLoader
from .model_loader import ChexpertModelLoader
from .clustering_visualiser import ClusteringVisualiser
from .utils import get_cluster_labels


class ClusteringPipeline:
    def __init__(self, root, path_to_pretrained_chexpert_model, CSVPATH, chexnet_targets, n_images, height=224, width=224, n_channels=3):
        '''
            :param root: path to the directory that contains CheXpert-v1.0-small
            :param path_to_pretrained_chexpert_model:
            :param CSVPATH: path to chexpert csv
            :param chexnet_targets: targets of chexnet dataset
            :param n_images: number images to process
            :param height: height of images to be clustered
            :param width: width of images to be clustered
            :param n_channels: number of channels in images
        '''
        self.root = root
        dataset_loader = ChexpertDatasetLoader(chexnet_targets)
        self.height = height
        self.width = width
        model_loader = ChexpertModelLoader(path_to_pretrained_chexpert_model, height, width, n_channels)

        self.model = model_loader.get_model()
        self.dataset = dataset_loader.get_dataset(CSVPATH, n_images)
        self.visualiser = ClusteringVisualiser(self.dataset)
        self.feat_vecs  = None

    def build_2d_clusters(self, n_clusters):
        self.feat_vecs = self.get_feature_vectors()
        reduced_feat_vecs_2d = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(self.feat_vecs)
        cluster_labels = get_cluster_labels(self.feat_vecs, n_clusters=n_clusters)
        figure_2d = self.visualiser.visualise_2d(reduced_feat_vecs_2d, cluster_labels.astype(str))
        figure_2d.show()

    def build_3d_clusters(self, n_clusters):
        self.feat_vecs = self.get_feature_vectors()
        reduced_feat_vecs_3d = TSNE(n_components=3, learning_rate='auto', init='random').fit_transform(self.feat_vecs)
        cluster_labels = get_cluster_labels(self.feat_vecs, n_clusters=n_clusters)
        figure_3d = self.visualiser.visualise_3d(reduced_feat_vecs_3d, cluster_labels.astype(str))
        figure_3d.show()

    def get_feature_vectors(self):
        return self.feat_vecs if self.feat_vecs is not None else self._get_clustering_attributes(self.dataset, self.model)

    def get_dataset(self):
        return self.dataset

    def _get_clustering_attributes(self, df, model):
        feat_vecs = []
        for i, row in df.iterrows():
            print("Number of Images Processed : ", i)
            path = self.root + row['Path']
            if os.path.exists(path):
                image = cv2.imread(path)
                resized_img = cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_AREA)
                normalised_img = np.array([resized_img]) / 255
                prediction = model(normalised_img).numpy().flatten()
                feat_vecs.append(prediction)
        return np.array(feat_vecs)
