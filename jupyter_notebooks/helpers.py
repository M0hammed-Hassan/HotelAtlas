import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def subplot_numerical_distributions(df:pd.DataFrame, columns:list):
    """
    Plot the distribution of multiple numerical columns in a DataFrame.
    """
    num_cols = len(columns)
    fig, axes = plt.subplots(nrows=(num_cols + 1) // 2, ncols=2, figsize=(9, 3 * ((num_cols + 1) // 2)))
    axes = axes.flatten()
    
    for i, column in enumerate(columns):
        sns.histplot(df[column], kde=True, bins=30, ax=axes[i])
        axes[i].set_title(f'Distribution of {column}')
        axes[i].set_xlabel(column.title())
        axes[i].set_ylabel('Frequency')
    
    for j in range(i + 1, len(axes)): 
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

def plot_categorical_feature(df:pd.DataFrame, column:str, top_n:int = 10):
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

def plot_categorical_counts(df:pd.DataFrame, column:str):
    plt.figure(figsize=(8, 5))
    sns.countplot(
        x=df[column],  
        data = df,
        palette="crest"
    )
    plt.xlabel(column.title())
    plt.ylabel("Count")
    plt.title(f"Countplot for {column.title()}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()