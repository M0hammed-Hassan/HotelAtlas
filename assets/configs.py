import logging
from dataclasses import dataclass

RANDOM_STATE = 100


@dataclass
class ClusteringModelsConfig:
    """
    Class to hold configurations for various clustering models.
    """

    PCA_COMPONENTS: int = 14
    TSNE_COMPONENTS: int = 2
    KMEANS_N_CLUSTERS: int = 10
    DATA_SAMPLE_SIZE: int = 10000
    RANDOM_STATE: int = RANDOM_STATE
    KMEANS_RAGE: range = range(2, 25)


class ClassificationModelsConfig:
    """
    Class to hold configurations for various classification models.
    """

    TEST_SIZE = 0.3
    LOG_LEVEL = logging.INFO
    RANDOM_STATE = RANDOM_STATE
    RESULTS_DIR = "model_results"
