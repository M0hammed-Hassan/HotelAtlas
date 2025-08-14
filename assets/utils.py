import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


def subplotNumericalDistributions(df: pd.DataFrame, columns: list) -> None:
    """
    Plot the distribution of multiple numerical columns in a DataFrame.
    """
    num_cols = len(columns)
    fig, axes = plt.subplots(
        nrows=(num_cols + 1) // 2, ncols=2, figsize=(9, 3 * ((num_cols + 1) // 2))
    )
    axes = axes.flatten()

    for i, column in enumerate(columns):
        sns.histplot(df[column], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f"Distribution of {column}")
        axes[i].set_xlabel(column.title())
        axes[i].set_ylabel("Frequency")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plotCategoricalFeatures(df: pd.DataFrame, column: str, top_n: int = 10) -> None:
    """
    This function to plot categorical columns data.
    """
    value_counts = df[column].value_counts()
    value_counts = value_counts.head(top_n)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=value_counts.values, y=value_counts.index, palette="viridis")
    plt.xlabel("Count")
    plt.ylabel(column.title())
    plt.title(f"Distribution of {column.title()}")
    plt.tight_layout()
    plt.show()


def plotNumericalCounts(df: pd.DataFrame, column: str) -> None:
    """
    Plot the count of unique values in a numerical column.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(x=df[column], data=df, palette="crest")
    plt.xlabel(column.title())
    plt.ylabel("Count")
    plt.title(f"Countplot for {column.title()}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def applyStandardScaler(numeric_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale numerical columns in a DataFrame using StandardScaler.
    """
    scaler = StandardScaler()
    scaled_numeric_array = scaler.fit_transform(numeric_df)
    scaled_df = pd.DataFrame(
        scaled_numeric_array, index=numeric_df.index, columns=numeric_df.columns
    )
    return scaled_df


def applyHiralricalClustering(scaled_df: pd.DataFrame, metric: str = "cosine") -> None:
    """
    Perform hierarchical clustering and plot the dendrogram.

    Parameters:
    scaled_df (DataFrame): The scaled sample data for clustering.
    metric (str): The distance metric to use for clustering.
    """
    methods = ["single", "complete", "average", "ward"]
    if metric == "cosine":
        methods.pop()  # 'ward' method is not applicable for cosine distance

    plt.figure(figsize=(15, 5))
    for i, method in enumerate(methods):
        plt.subplot(1, 4, i + 1)
        plt.title(method)
        dendrogram(
            linkage(scaled_df.values, method=method, metric=metric),
            labels=scaled_df.index,
            leaf_rotation=90,
            leaf_font_size=2,
        )
    plt.show()


def calculateKmeansMtrics(
    data: pd.DataFrame, k_range: range, random_state: int
) -> tuple:
    """
    Calculates inertia and silhouette scores for a range of k values.

    Returns:
    --------
    inertia : list
        Inertia values for each k.
    silhouette_scores : list
        Silhouette scores for each k.
    """
    inertia = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        labels = kmeans.labels_
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, labels))

    return inertia, silhouette_scores


def plotKmeansMetrics(
    k_range: range, inertia: list, silhouette_scores: list, figsize=(14, 5)
) -> None:
    """
    Plots inertia and silhouette scores side by side.
    """
    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.title("Inertia (Elbow Method)")
    sns.lineplot(x=list(k_range), y=inertia, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")

    plt.subplot(1, 2, 2)
    plt.title("Silhouette Score")
    sns.lineplot(x=list(k_range), y=silhouette_scores, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.show()


def kmeansPredictClusters(
    data: pd.DataFrame, n_clusters: int, random_state: int
) -> pd.Series:
    """
    Fit KMeans and return cluster labels.

    Returns:
    labels (array): Cluster labels for each data point.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(data)
    return kmeans.labels_
