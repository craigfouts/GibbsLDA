import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Sequence
from scipy.stats import skewnorm

## Predefined simulated data options that define boundries for different block dataset 

SIZE = 30
## Produces five topics modeled after https://www.nature.com/articles/s41592-022-01687-w
GGBLOCKS = [[[[15, 20], [0, 5]], [[25, 30], [0, 5]], [[10, 20], [5, 10]],
            [[25, 30], [5, 10]], [[5, 15], [10, 15]], [[0, 5], [15, 20]],
            [[10, 15], [15, 20]], [[20, 25], [20, 25]], [[0, 5], [25, 30]],
            [[10, 15], [25, 30]]],
            [[[0, 15], [0, 5]], [[0, 10], [5, 10]], [[0, 5], [10, 15]]],
            [[[20, 25], [0, 10]], [[15, 30], [10, 15]]],
            [[[15, 30], [15, 20]], [[15, 20], [20, 30]], [[25, 30], [20, 30]],
            [[20, 25], [25, 30]]],
            [[[5, 10], [15, 30]], [[0, 5], [20, 25]], [[10, 15], [20, 25]]]]

## Produces two topics that are patterened in checkered blocks
CHBLOCKS = [[[[0, 10], [0, 10]], [[20, 30], [0, 10]], [[10, 20], [10, 20]],
            [[0, 10], [20, 30]], [[20, 30], [20, 30]]], 
            [[[10, 20], [0, 10]], [[0, 10], [10, 20]], [[20, 30], [10, 20]],
            [[10, 20], [20, 30]]]]

def generate_means(n_genes, n_informative, n_topics, min_=10, max_=30):
    """Builds an array of means for each class. The first n_informative genes 
    have different means between classes. 

    Parameters
    ----------
        n_genes : int
            The number of total genes in the dataset
        n_informative : int
            The number of informative (different) genes between classes
        n_topics : int
            The number of classes/topics/clusters in the data
        min_ : int, default=50
            The minimum mean
        max_ : int, default=50
            The maximum mean

    Returns
    -------
        mean_arr : numpy.ndarray, shape=(n_topics, n_genes)
            The array of means
        variances : numpy.ndarray, shape=(n_genes,)
            The array of variances 
    """

    assert isinstance(n_genes, int), f'Expected type of n_genes to be integer but got {type(n_genes)}.'
    assert n_genes > 0, f'Expected n_genes > 0 but got n_genes = {n_genes}.'
    assert isinstance(n_informative, int), f'Expected type of n_informative to be integer but got {type(n_informative)}.'
    assert n_informative > 0, f'Expected n_informative > 0 but got n_informative = {n_informative}.'
    assert n_informative <= n_genes, f'Expected n_informative <= n_genes but got n_informative = {n_informative} and n_genes = {n_genes}.'
    assert isinstance(n_topics, int), f'Expected type of n_topics to be integer but got n_topics = {n_topics}.'
    assert n_topics > 0, f'Expected n_topics > 0 but got n_topics = {n_topics}.'
    assert isinstance(min_, int), f'Expected type of min_ to be integer but got {type(min_)}.'
    assert min_ > 0, f'Expected min_ > 0 but got min_ = {min_}.'
    assert isinstance(max_, int), f'Expected type of max_ to be integer but got {type(max_)}.'
    assert max_ >= min_, f'Expected max_ > min_ but got max_ = {max_} and min_ = {min_}.'

    init_means = np.random.randint(min_, max_, n_genes)
    variances = np.random.randint(1, min_/5, n_genes)

    # Initialize the array of means
    mean_arr = np.empty((n_topics, n_genes))
    for i in range(n_topics):
        mean_arr[i, :] = init_means

    for i in range(1, n_topics):
        for j in range(n_informative):
            gene_var = variances[j]

            while True:  # TODO: move break into condition
                if np.random.rand() > 0.5:
                    new_mean = init_means[j] + np.random.randint(2, 10) * gene_var
                else:
                    new_mean = init_means[j] - np.random.randint(2, 10) * gene_var
                
                if new_mean - 5 * variances[j] > 0:
                    break

            mean_arr[i, j] = new_mean

    return mean_arr, variances

def generate_from_array(n_cells, mean_arr, variances):  # TODO: Consolidate (if n_cells is count propagate to array)
    """
     Given cells in each topic and mean and variance returns the features and their labels 

    Parameters
    ----------
        n_cells : numpy.ndarray, shape=(n_topics,)
            Number of cells in each topic
        means : numpy.ndarray, shape=(n_topics, n_genes), optional
            the `n_topics x n_genes` array of means
        variances : array-like, shape=(n_topics,), optional
            The variances of each gene

    Returns
    -------
        data : numpy.ndarray, shape=(n_cells, n_genes)
            The list of cells by genes 
        labels : numpy.ndarray, shape=(n_cells,)
            The labels for each cell 
    """

    assert isinstance(n_cells, np.ndarray), f'Expected type of n_cells to be numpy.ndarray but got {type(n_cells)}.'
    assert len(n_cells.shape) == 1, f'Expected dimension of n_cells to be 1 but got {len(n_cells.shape)}.'
    assert isinstance(mean_arr, np.ndarray), f'Expected type of mean_arr to be numpy.ndarray but got {type(mean_arr)}.'
    assert len(mean_arr.shape) == 2, f'Expected dimension of mean_arr to be 2 but got {len(mean_arr.shape)}.'
    assert isinstance(variances, np.ndarray), f'Expected type of variances to be numpy.ndarray but got {type(variances)}.'
    assert len(variances.shape) == 1, f'Expected dimension of variances to be 1 but got {len(variances.shape)}.'
    assert n_cells.shape[0] == mean_arr.shape[0], f'Incompatible shapes of n_cells {n_cells.shape} and mean_arr {mean_arr.shape}.'
    assert mean_arr.shape[1] == variances.shape[0], f'Incompatible shapes of mean_arr {mean_arr.shape} and variances {variances.shape}.'
    
    n_genes = mean_arr.shape[1]
    data = np.empty((sum(n_cells), n_genes))
    labels = np.empty(sum(n_cells), dtype=int)
    cell_count = 0

    for i in range(0, len(n_cells)):
        cells_in_class = n_cells[i]

        # Add class data to the larger dataset and save the class assignment
        class_data = np.random.normal(mean_arr[i, :], variances, (cells_in_class, n_genes))
        data[cell_count:cell_count+cells_in_class, :] = class_data
        labels[cell_count:cell_count+cells_in_class] = i
        
        cell_count += cells_in_class
    
    return data, labels

def generate_data(n_genes, n_informative, n_cells=900, n_topics=2, means=None, variances=None):
    """
    Generates a data matrix and the labeling given the specififed means and variances. 

    Parameters
    ----------
        n_genes : int
            The number of total genes in the dataset
        n_informative : int
            The number of informative (different) genes between classes
        n_cells : int, default=900
            The total number of cells to include in the data
        n_topics : int, default=2
            The number of classes/types/clusters in the data
        means : numpy.ndarray, shape=(n_topics, n_genes), optional
            the `n_topics x n_genes` array of means
        variances : array-like, shape=(n_topics,), optional
            The variances of each gene
    
    Returns
    -------
        data : numpy.ndarray, shape=(n_cells, n_genes)
            The `cells x genes` data array
        labels : numpy.ndarray, shape=(n_cells,)
            The labeling assignments
    """

    if means is not None:
        assert variances is not None, 'Expected `variances` argument.'
    elif variances is not None:
        assert means is not None, "Expected `means` argument."
    else:
        means, variances = generate_means(n_genes, n_informative, n_topics)

    if isinstance(n_cells, np.ndarray):
        data, labels = generate_from_array(n_cells, means, variances)
    else:
        data, labels = generate_from_array(np.repeat(n_cells, n_topics), means, variances)

    return data, labels

def scale(X, min_, max_, mirror=False):
    """TODO
    
    Parameters
    ----------
        X : numpy.ndarray, TODO
            TODO
        min_ : int
            TODO
        max_ : int
            TODO
        mirror : bool, default=False
            TODO

    Returns
    -------
        X : numpy.ndarray, TODO
            TODO
    """

    assert isinstance(X, np.ndarray), f'Expected type of X to be numpy.ndarray but got {type(X)}.'
    # TODO: assert shape of X
    assert isinstance(min_, int), f'Expected type of min_ to be int but got {type(min_)}.'
    assert isinstance(max_, int), f'Expected type of max_ to be int but got {type(max_)}.'
    assert max_ >= min_, f'Expected max_ >= min_ but got max_ = {max_}.'
    assert isinstance(mirror, bool), f'Expected type of mirror to be bool but got {type(mirror)}.'
    
    X = X + (X.min() if X.min() > 0 else -X.min())
    X = X / X.max()

    if mirror:
        X = 1. - X

    X = X * (max_ - min_) + min_
    
    return X

def rand_locs(n_locs, x_min, x_max, y_min, y_max, dist='uniform', skew=100, mirror=False):
    """
    Gives a requested number of random locations

    Parameters
    ----------
        n_locs : int
            The number of random points to return 
        x_min : int
            The minimum x-coordinate
        x_max : int
            The maximum x-coordinate
        y_min : int
            The minimum y-coordinate
        y_max : int
            The maximum y-coordinate

    Returns
    -------
        locs : numpy.ndarray, shape=(n_locs, 2)
            List of randome points 
    """

    # assert isinstance(n_locs)
    
    if dist == 'uniform':
        x_locs = np.random.randint(x_min, x_max, size=(n_locs, 1))
    elif dist == 'skewnorm':
        x_locs = scale(skewnorm(skew).rvs((n_locs, 1)), x_min, x_max, mirror)
    else:
        raise NotImplementedError(f'Distribution "{dist}" not supported.')

    y_locs = np.random.randint(y_min, y_max, size=(n_locs, 1))
    locs = np.concatenate([x_locs, y_locs], axis=1)

    return locs

def generate_dist(n_genes, n_informative, n_cells=900, means=None, variances=None, mode='split', mixed=False, x_max=30, y_max=30):
    """
    Will generate a split,two topic, dataset

    Parameters
    ----------
        n_genes : int
            The number of total genes in the dataset
        n_informative : int
            The number of informative (different) genes between classes
        n_cells : int, default=900
            Optional. The total number of cells to include in the data
        means : numpy.ndarray, shape=(n_topics, n_genes), optional
            Optional. The `n_topics x n_genes` array of means
        variances : array-like, shape=(n_topics,), optional
            Optional. The variances of each gene
        mode : str, optional
            Optional. What kind of dataset will be simulated
            'split' - Two topics 
            'gradient' - Two topics, with points that mix in the middle
            'blocks' - Topics occuring in blocks 
        mixed : bool, default=False
            Optional. If true adds noise to the simulated data, false otherwise
        x_max : int, default=30
            Optional. Maximum point value in x-direction 
        y_max : int, default=30
            Optional. Maximum point value in y-direction , default=30
    
    Returns
    -------
        X : numpy.ndarray or pandas.DataFrame, shape=(n_cells, n_genes)
            Matrix of features for each cell 
        X_labels : numpy.ndarray, shape=(n_cells)
            Labels for each cell 
    """

    if means is not None and variances is not None:
        assert means.shape == (2, n_genes), "Expected argument 'means' of shape (2, n_genes)."
        assert variances.shape == (n_genes,), "Expected argument 'variances' of shape (n_genes,)"

    X, X_labels = generate_data(n_genes, n_informative, n_cells, 2, means, variances)

    class0_idx = np.where(X_labels==0)[0]
    class1_idx = np.where(X_labels==1)[0]
    n_class0 = class0_idx.shape[0]
    n_class1 = class1_idx.shape[0]

    if mixed:  # TODO: need to fix this so mixed isn't treated like a mode
        n_mixed0 = int(n_class0 / 5)
        n_mixed1 = int(n_class1 / 5)

        M_labels = X_labels.copy()
        X_labels[class0_idx[n_mixed0:]] = 0
        X_labels[class0_idx[:n_mixed0]] = 1
        X_labels[class1_idx[n_mixed1:]] = 1
        X_labels[class1_idx[:n_mixed1]] = 0

        X[class0_idx[:n_mixed0], :2] = rand_locs(n_mixed0, x_max/2, x_max, 0, y_max, 'uniform')
        X[class0_idx[n_mixed0:], :2] = rand_locs(n_class0-n_mixed0, 0, x_max/2, 0, y_max, 'uniform')
        X[class1_idx[:n_mixed1], :2] = rand_locs(n_mixed1, 0, x_max/2, 0, y_max, 'uniform')
        X[class1_idx[n_mixed1:], :2] = rand_locs(n_class1-n_mixed1, x_max/2, x_max, 0, y_max, 'uniform' )

        return X, X_labels, M_labels
    elif mode == 'split':
        X[class0_idx, :2] = rand_locs(n_class0, 0, x_max/2, 0, y_max, 'uniform')
        X[class1_idx, :2] = rand_locs(n_class1, x_max/2, x_max, 0, y_max, 'uniform')
    elif mode == 'gradient':
        X[class0_idx, :2] = rand_locs(n_class0, 0, x_max, 0, y_max, 'skewnorm')
        X[class1_idx, :2] = rand_locs(n_class1, 0, x_max, 0, y_max, 'skewnorm', mirror=True)
    else:
        raise NotImplementedError(f'Mode "{mode}" not supported.')
    
    return X, X_labels

def cells_per_block(blocks):
    """
    Counts the number of cells in each block 

    Parameters
    ----------
        blocks : numpy.ndarray, shape=(n_topics, number of boundries, 2, 2)
            Provides the block boundries 
    
    Returns
    -------
        cell_counts : list, shape=(number of blocks, counts)
            Cells in each block in a numpy array 
    """
    
    cell_counts = []

    for block in blocks:
        cell_count = 0
        for b in block:
            cell_count = cell_count + (b[0][1] - b[0][0]) * (b[1][1] - b[1][0])
        cell_counts.append(cell_count)

    return np.array(cell_counts)

def in_block(p, block):
    """
    Checks whether a point is within the specified block boundry 

    Parameters
    ----------
        p : numpy.ndarray, shape=(2,)
            The point to be checked 
        block : numpy.ndarray, shape = (number of boundries, 2, 2)
            Provides the block boundries 

    Returns
    -------
        in_block : bool
            True if the point is in the block, false otherwise  
    """
    
    return any(p[0] in range(*b[0]) and p[1] in range(*b[1]) for b in block)

def rand_blocks(n_topics, n_cells):
    """
    Generates a list of block boundries randomly 

    Parameters
    ----------
        n_topics : int
            The number of classes/topics/clusters in the data

    Returns
    -------
        blocks : list, shape = (n_topics, number of boundries, 2, 2)
            Provides the block boundries 
    """
    
    blocks = [np.array([]) for _ in range(n_topics)]
    total_size = SIZE * n_cells
    for x in range(0, total_size, 5):
        for y in range(0, total_size, 5):
            c = np.random.randint(n_topics)
            blocks[c].append(np.array([[x, x + 5], [y, y + 5]]))

    return blocks

def generate_blocks(n_genes, n_informative, n_topics=2, n_cells=1, mixed=False, blocks=None, means=None, variances=None):    
    """
    Gives all necessary information to understand a block dataset based on the specified boundaries 
    
    Parameters
    ----------
        n_genes : int
            The number of total genes in the dataset
        n_informative : int
            The number of informative (different) genes between classes
        n_topics : int, default=2
            The number of classes/topics/clusters in the data
        n_cells : int, default=1
            Hhow many times bigger should the grid be than the base size
        mixed : bool, default=False
            Optional. If true adds noise to the simulated data, false otherwise
        blocks : array-like, optional
            Optional. Constants that give the boundries for specified blocks 
        means : numpy.ndarray, shape=(n_topics, n_genes), optional
            Optional. The `n_topics x n_genes` array of means
        variances : array-like, shape=(n_topics,), optional
            Optional. The variances of each gene
    
    Returns
    ------- 
        X : numpy.ndarray or pandas.DataFrame, shape=(n_cells, n_genes)
            Matrix of features for each cell 
        X_labels : numpy.ndarray, shape=(n_cells)
            Labels for each cell 
        M_labels : numpy.ndarry 
            Labels for each cell with noise added 
    """
    total_size = SIZE * n_cells

    if blocks is None:
        blocks = rand_blocks(n_topics, n_cells)
    blocks = blocks * n_cells

    if means is not None and variances is not None:
        assert means.shape == (len(blocks), n_genes), "Expected argument 'means' of shape (n_topics, n_genes)."
        assert variances.shape == (n_genes,), "Expected argument 'variances' of shape (n_genes,)"

    cells_per_class = cells_per_block(blocks)
    X, X_labels = generate_data(n_genes, n_informative, cells_per_class, len(blocks), means, variances)

    idx = np.zeros(len(cells_per_class), dtype=np.int32)
    for i in range(1, idx.shape[0]):
        idx[i] = idx[i - 1] + cells_per_class[i - 1]

    if mixed:
        M = rand_locs(total_size * 3, 0, total_size, 0, total_size)
        M_labels = X_labels.copy()

    for p in np.mgrid[:total_size, :total_size].reshape(2, total_size * total_size).T:
        for i, block in enumerate(blocks):
            if in_block(p, block):
                X[idx[i], 0] = p[0]
                X[idx[i], 1] = p[1]

                if mixed and any(np.array_equal(m, p) for m in M):
                    M_labels[idx[i]] = (i + np.random.randint(1, len(blocks))) % len(blocks)

                idx[i] += 1

    if mixed:  
        return X, X_labels, M_labels

    return X, X_labels

def generate_dataset(n_genes, n_informative, n_cells=900, n_topics=2, blocks=None, mode=None, mixed=False, means=None, variances=None, x_max=30, y_max=30, as_df=False):
    """
     Used for testing any segmentation algorthim by returning a lebeled dataset.

    Parameters
    ----------
        n_genes : int
            The number of total genes in the dataset
        n_informative : int
            The number of informative (different) genes between classes
        n_cells : int, default=900
            The total number of cells to include in the data
            or if using the block dataset how many times bigger should the grid be than the base size
        n_topics : int
            The number of classes/topics/clusters in the data
        blocks : numpy.ndarray, shape=(n_topics, number of boundries, 2, 2), optional
            Optional. Constants that give the boundries for specified blocks 
        means : numpy.ndarray, shape=(n_topics, n_genes), optional
            Optional. The `n_topics x n_genes` array of means
        variances : array-like, shape=(n_topics,), optional
            Optional. The variances of each gene
        mode : str, optional
            Optional. What kind of dataset will be simulated
            'split' - Two topics 
            'gradient' - Two topics, with points that mix in the middle
            'blocks' - Topics occuring in blocks 
        mixed : bool, default=False
            Optional. If true adds noise to the simulated data, false otherwise
        x_max : int, default=30
            Optional. Maximum point value in x-direction 
        y_max : int, default=30
            Optional. Maximum point value in y-direction 
        as_df : bool, default=False
            Optional. If true returns a dataframe instead of numpy array 

    Returns
    -------
        X : numpy.ndarray or pandas.DataFrame, shape=(n_cells, n_genes)
            Matrix of features for each cell 
        X_labels : numpy.ndarray, shape=(n_cells)
            Labels for each cell 
        M_labels : numpy.ndarry 
            Labels for each cell with noise added 
    """

    if mode == 'blocks' or blocks is not None:
        if mixed:  # TODO: clean this up
            X, X_labels, M_labels = generate_blocks(n_genes, n_informative, n_topics, n_cells, mixed, blocks, means, variances)
        else:
            X, X_labels = generate_blocks(n_genes, n_informative,  n_topics, n_cells, mixed, blocks, means, variances)
    elif mode in ('split', 'gradient'):
        if mixed:  # TODO: clean this up
            X, X_labels, M_labels = generate_dist(n_genes, n_informative, n_cells, means, variances, mode, mixed, x_max, y_max)
        else:
            X, X_labels = generate_dist(n_genes, n_informative, n_cells, means, variances, mode, mixed, x_max, y_max)
    else:
        X, X_labels = generate_data(n_genes, n_informative, n_cells, n_topics, means, variances)

    if as_df:
        X = pd.DataFrame(X)
        columns = ['x', 'y'] + [f'Gene_{i}' for i in range(1, n_genes - 1)]
        X = X.set_axis(columns, axis=1)
    
    if mixed:
        return X, X_labels, M_labels

    return X, X_labels

def generate_datasets(n_sections, n_genes, n_informative, n_cells=900, n_topics=2, blocks=None, mode=None, mixed=False, means=None, variances=None, x_max=30, y_max=30):
    """
    Used for testing any segmentation algorthim by returning multiple lebeled datasets. 
    
    Parameters
    ----------
        ids : int or array-like
            The number of datasets to be produced 
        n_genes : int
            The number of total genes in the dataset
        n_informative : int
            The number of informative (different) genes between classes
        n_cells : int, default=900
            The total number of cells to include in the data 
            or if using the block dataset how many times bigger should the grid be than the base size 
        n_topics : int
            The number of classes/topics/clusters in the data
        blocks : numpy.ndarray, shape=(n_topics, number of boundries, 2, 2), optional
            Optional. Constants that give the boundries for specified blocks 
        means : numpy.ndarray, shape=(n_topics, n_genes), optional
            Optional. The `n_topics x n_genes` array of means
        variances : array-like, shape=(n_topics,), optional
            Optional. The variances of each gene
        mode : str, optional
            Optional. What kind of dataset will be simulated
            'split' - Two topics 
            'gradient' - Two topics, with points that mix in the middle
            'blocks' - Topics occuring in blocks 
        mixed : bool, default=False
            Optional. If true adds noise to the simulated data, false otherwise
        x_max : int, default=30
            Optional. Maximum point value in x-direction 
        y_max : int, default=30
            Optional. Maximum point value in y-direction 

    Returns
    -------
        X : numpy.ndarray or pandas.DataFrame, shape=(ids, n_cells, n_genes)
            Matrix of features for each cell 
        X_labels : numpy.ndarray, shape=(ids, n_cells)
            Labels for each cell 
    """

    args = tuple(locals().values())[1:]

    if means is not None:
        means, variances = generate_means(n_genes, n_informative, n_topics)
    sections, labels = [], []

    for i in range(n_sections):
        dataset, label = generate_dataset(n_genes, n_informative, n_cells, n_topics, blocks, mode, mixed, means, variances, x_max, y_max)
        dataset = np.concatenate([np.ones((dataset.shape[0], 1))*i, dataset], -1)
        sections.append(dataset)
        labels.append(label)

    datasets = np.stack(sections)
    labels = np.stack(labels)

    return datasets, labels
    # if type(ids) == int:
    #     ids = [f'X{i}_df' for i in range(ids)]

    # datasets, labels = {}, {}
    # for id in ids:
    #     datasets[id], labels[id] = generate_dataset(*args, as_df=True)

    # return datasets, labels
    