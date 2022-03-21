import pandas as pd
import plotly.express as px


class ClusteringVisualiser:

    def __init__(self, chexpert_dataframe):
        self.df = chexpert_dataframe

    def visualise_2d(self, reduced_feat_vecs, cluster_labels):
        df = pd.DataFrame(
            {"X": reduced_feat_vecs[:, 0], "Y": reduced_feat_vecs[:, 1], "Labels": self.df['positive'],
             'Clusters': cluster_labels
                , 'Age': self.df['Age'], 'Sex': self.df['Sex'], 'Frontal/Lateral': self.df['Frontal/Lateral']})
        fig = px.scatter(df, x="X", y="Y",
                         hover_name="Labels",
                         color='Clusters', hover_data=['Age', 'Sex', 'Frontal/Lateral'], title='2D - Image Clustering')

        return fig

    def visualise_3d(self, reduced_feat_vecs, cluster_labels):
        df = pd.DataFrame(
            {"X": reduced_feat_vecs[:, 0], "Y": reduced_feat_vecs[:, 1], "Z": reduced_feat_vecs[:, 2],
             "Labels": self.df['positive'], 'Clusters': cluster_labels
                , 'Age': self.df['Age'], 'Sex': self.df['Sex'], 'Frontal/Lateral': self.df['Frontal/Lateral']})
        fig = px.scatter_3d(df, x='X', y='Y', z='Z',
                            hover_name="Labels",
                            hover_data=['Age', 'Sex', 'Frontal/Lateral'],
                            color='Clusters',
                            title='3D - Image Clustering'
                            )
        return fig
