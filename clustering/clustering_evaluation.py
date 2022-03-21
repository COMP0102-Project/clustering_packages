from sklearn.metrics import silhouette_score
from .utils import get_cluster_labels
from sklearn.manifold import TSNE


class Evaluator:

    def get_silhouette_score(self, feat_vecs, max_number_clusters=14):
        '''
        :param feat_vecs: (n x m) matrix where n is the number of images and m is the size of vector representing each image
        :param max_number_clusters: number of clusters to examine
        :return: x,y set of coordinates where a y represents the silhoutte score and x the number of clusters examined for a particular y
        '''
        y = []
        x = range(2, max_number_clusters)
        for n_clusters in x:
            print("Clusters Evaluated", n_clusters - 1)
            cluster_labels = get_cluster_labels(feat_vecs, n_clusters=n_clusters)
            reduced_feat_vecs_3d = TSNE(n_components=3, learning_rate='auto', init='random').fit_transform(feat_vecs)
            score = silhouette_score(reduced_feat_vecs_3d, cluster_labels, metric='euclidean')
            y.append(score)
        return x, y

    def get_purity_score(self, dataset, feat_vecs, max_number_clusters=14):
        '''
        :param dataset:  dataframe containing the chexpert csv
        :param feat_vecs: (n x m) matrix where n is the number of images and m is the size of vector representing each image
        :param max_number_clusters: number of clusters to examine
        :return: x,y set of coordinates where a y represents the purity score and x the number of clusters examined for a particular y
        '''
        y = []
        x = range(2, max_number_clusters)
        for n_clusters in x:
            print("Clusters Evaluated", n_clusters - 1)
            cluster_labels = get_cluster_labels(feat_vecs, n_clusters=n_clusters)
            score = self._get_purity_helper(dataset, cluster_labels)
            y.append(score)
        return x, y

    def _get_purity_helper(self, dataset, cluster_labels):
        frequency_dict, number_of_labels = self._build_frequency_dict(dataset, cluster_labels)
        total_max_freq_in_clusters = 0
        for cluster_label in frequency_dict:
            total_max_freq_in_clusters += max(frequency_dict[cluster_label].values())

        return total_max_freq_in_clusters / number_of_labels

    def _build_frequency_dict(self, dataset, cluster_labels):
        frequency_dict = {}
        number_of_labels = 0
        for i, labels in enumerate(dataset['positive']):
            number_of_labels += 1 if labels else 0
            for label in labels:
                if cluster_labels[i] in frequency_dict:
                    subdict = frequency_dict[cluster_labels[i]]
                    if label in subdict:
                        subdict[label] += 1
                    else:
                        subdict.update({label: 1})
                else:
                    frequency_dict[cluster_labels[i]] = {label: 1}
        return frequency_dict, number_of_labels
