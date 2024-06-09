import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def perform_pca(df):
    X_standardized = (df - df.mean()) / df.std()
    pca = PCA()
    X_pca = pca.fit_transform(X_standardized)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    loadings = pca.components_
    loadings_df = pd.DataFrame(loadings.T, index=df.columns, columns=[f'PC{i+1}' for i in range(loadings.shape[0])])
    
    # Print the explained variance ratio
    print("Explained Variance Ratio:")
    for i, variance in enumerate(explained_variance_ratio, start=1):
        print(f"PC{i}: {variance:.4%}")
    
    # Print cumulative explained variance
    print("\nCumulative Explained Variance:")
    for i, cumulative_variance in enumerate(cumulative_explained_variance, start=1):
        print(f"PC{i}: {cumulative_variance:.4%}")

    return loadings_df, explained_variance_ratio, cumulative_explained_variance

def plot_variance_ratios(explained, cumulative):
    # Plot the explained variance ratio
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, len(explained) + 1), 
                   explained, alpha=0.6, align='center', label='Individual explained variance')
    plt.step(range(1, len(cumulative) + 1), 
             cumulative, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Components')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Add percentage labels on the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}', ha='center', va='bottom')
    
    plt.show()


def plot_circle_of_correlations(df):
    X_standardized = (df - df.mean()) / df.std()
    pca = PCA()
    X_pca = pca.fit_transform(X_standardized)
    pcs = pca.components_
    num_vars = pcs.shape[1]
    
    plt.figure(figsize=(10, 10))
    # Draw the unit circle
    circle = plt.Circle((0, 0), 1, color='g', fill=False)
    plt.gca().add_artist(circle)
    # Draw horizontal and vertical lines
    plt.axhline(0, color='k', linestyle='--', linewidth=1)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    
    # Plot the arrows
    for i in range(num_vars):
        plt.arrow(0, 0, pcs[0, i], pcs[1, i], alpha=0.5, head_width=0.03, head_length=0.05, color='r')
        plt.text(pcs[0, i] * 1.1, pcs[1, i] * 1.1, df.columns[i], color='b', ha='center', va='center', fontsize=10)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Circle of Correlations')
    plt.grid(True)
    plt.axis('equal')

    # Adjusting limits for visibility
    max_limit = np.abs(pcs).max()
    plt.xlim(-max_limit, max_limit)
    plt.ylim(-max_limit, max_limit)
    plt.show()


def plot_projection(df):
    # Initialize PCA with 2 components
    X_standardized = (df - df.mean()) / df.std()
    pca = PCA(n_components=2)
    pca.fit(X_standardized)
    projected_data = pca.transform(X_standardized)

    plt.figure(figsize=(8, 6))
    plt.scatter(projected_data[:, 0], projected_data[:, 1], alpha=0.8)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Projection of Data onto First Two Principal Components')
    plt.grid(True)
    plt.show()