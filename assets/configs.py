from dataclasses import dataclass


@dataclass
class ClusteringModelsHyperparameters:
    """
    Class to hold hyperparameters for various clustering models.
    """

    RANDOM_STATE: int = 100
    PCA_COMPONENTS: int = 14
    TSNE_COMPONENTS: int = 2
    KMEANS_N_CLUSTERS: int = 10
    DATA_SAMPLE_SIZE: int = 10000
    KMEANS_RAGE: range = range(2, 25)
