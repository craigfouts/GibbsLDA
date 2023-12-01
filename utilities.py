import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import os
import random
import torch
from sklearn.neighbors import NearestNeighbors


def set_seed(seed=9):  
    """Initializes random number generators.

    Parameters
    ----------
        seed : int, default=9
            Optional.
            Seed value to be used for all random number generation.

    Returns
    -------
        None
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def remove_lonely(data, labels, threshold=225., n_neighbors=12):
    """Remove small clusters and points that should not be kept in the dataset because they are to far away from other data points 
    
    Parameters
    ----------
        data : numpy.ndarray, shape=(x locations, y locations)
            The data from the anndata object 
        labels : numpy.ndarray, shape=(points, n_topics)
            The labels for the data 
        threshold : float, default=225.
            Optional. Distance that will be used to remove small clusters and points that should not be kept in the dataset
        n_neighbors : int, default=12
            Optional. Number of neighbors to be checked so that small clusters that should not be kept are removed 
    
    Returns
    -------
        data : numpy.ndarray, shape=(x locations, y locations)
            The datapoints that are kept after far away points are removed 
        labels : numpy.ndarray, shape=(points, n_topics)
            The labels for the kept datapoints
    """

    locs = data[:, :2]
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(locs)
    max_dist = knn.kneighbors()[0].max(-1)
    remove_idx, = np.where(max_dist > threshold)
    data = np.delete(data, remove_idx, axis=0)
    labels = np.delete(labels, remove_idx, axis=0)

    return data, labels

def read_spine_data(filename, threshold=225., n_neighbors=12, feature_key='protein', id_key='protein:celltype'):
    """Reads in anndata object according to how it is annotated in the NYGC Tech Innovation Lab
    
    Parameters
    ----------
        filename : str
            Location of the data to be read into the anndata object 
        id_key : str, default='protein:celltype'
            Optional. Key in order to access the labels 
        feature_key : str, default='protein'
            Optional. Key in order to access the feature locations 
        threshold : float, default=None
            Optional. Distance that will be used to remove small clusters and points that should not be kept in the dataset
        n_neighbors : int, default=12
            Optional. Number of neighbors to be checked so that small clusters that should not be kept are removed 

    Returns
    -------
        data : numpy.ndarray, shape=(x locations, y locations)
            The points from the anndata object 
        labels : numpy.ndarray, shape=(points, n_topics)
            The labels for the data 
    """

    mdata = mu.read(filename)
    x, y = mdata['physical'].obsm['spatial'].T
    features = mdata[feature_key].X
    data = np.concatenate([x[None].T, y[None].T, features], -1)
    ids = mdata.obs[id_key]
    labels = np.unique(ids, return_inverse=True)[1]

    if threshold is not None:
        data, labels = remove_lonely(data, labels, threshold, n_neighbors)

    return data, labels

def read_anndata(filename, id_key='leiden', spatial_key='spatial', threshold=225., n_neighbors=12):
    """Reads in anndata object according to how it is commonly annotated by other labs 

    Parameters
    ----------
        filename : str
            Location of the data to be read into the anndata object 
        id_key : str, default='leiden'
            Optional. Key in order to access the labels 
        spatial_key : str, default='spatial'
            Optional. Key in order to access the data locations 
        threshold : float, default=None
            Optional. Distance that will be used to remove small clusters and points that should not be kept in the dataset
        n_neighbors : int, default=12
            Optional. Number of neighbors to be checked so that small clusters that should not be kept are removed 
    
            
    Returns
    -------
        data : numpy.ndarray, shape=(x locations, y locations)
            The data from the anndata object 
        labels : numpy.ndarray, shape=(points, n_topics)
            The labels for the data 
    """

    mdata = mu.read(filename)
    x, y = mdata.obsm[spatial_key].T
    features = mdata.X
    data = np.concatenate([x[None].T, y[None].T, features], -1)
    ids = mdata.obs[id_key]
    labels = np.unique(ids, return_inverse=True)[1]

    if threshold is not None:
        data, labels = remove_lonely(data, labels, threshold, n_neighbors)

    return data, labels

def visualize_dataset(X, X_labels, size=32, show_ax=True, filename=None, colormap='tab20'):
    """
    Used to visualize the specified dataset 

    Parameters
    ----------
        X : array-like
            Matrix of features for each cell 
        X_labels : array-like
            Labels for each cell 
        show_ax : bool, default=True
            Optional. If true the axis will be shown, false otherwise
        filename : str, optional
            Optional. File location where the visualization will be saved

    Returns
    -------
        None
    """
    
    fig, ax = plt.subplots()
    if not show_ax:
        ax.axis('off')

    ax.scatter(X[:, 0], X[:, 1], s=size, c=X_labels, cmap=colormap)

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')

def visualize_datasets(X, X_labels, size=32, show_ax=True, filename=None, colormap='tab20'):
    # n_cols = max(X.shape[0] % max_cols, max_cols)
    # n_rows = X.shape[0] // n_cols + 1
    # print(n_rows, n_cols)

    s = X.shape[0] if len(X.shape) == 3 else 1
    fig, ax = plt.subplots(1, s)
    # if not show_ax:
    #     ax.axis('off')

    if (len(X.shape) == 3 and X.shape[0] == 1):
        labels = X_labels[0] if len(X_labels.shape) > 1 else X_labels
        ax.scatter(X[0, :, 1], X[0, :, 2], s=size, c=labels, cmap=colormap)
        if not show_ax:
            ax.axis('off')
    elif len(X.shape) == 2:
        labels = X_labels[0] if len(X_labels.shape) > 1 else X_labels
        ax.scatter(X[:, 1], X[:, 2], s=size, c=labels, cmap=colormap)
        if not show_ax:
            ax.axis('off')
    else:
        for i in range(X.shape[0]):
            ax[i].scatter(X[i, :, 1], X[i, :, 2], s=size, c=X_labels[i], cmap=colormap)
            ax[i].set_aspect('equal', 'box')
            if not show_ax:
                ax[i].axis('off')

    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
