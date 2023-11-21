import numpy as np
from sklearn.datasets import make_classification

def generate_means(n_features, n_topics):
    means, _ = make_classification(n_samples=n_topics, n_features=n_features, n_informative=n_features, n_redundant=0, n_repeated=0, n_classes=n_topics)
    return means

def partition(n, n_partitions):
    parts = [n // n_partitions]
    for i in range(1, n_partitions):
        parts.append((n - sum(parts))//(n_partitions - i))
    return parts

def shuffle_data(data, labels):
    idx = np.random.permutation(data.shape[0])
    return data[idx], labels[idx]

def generate_data(n_samples, n_features, n_topics, scale=1., means=None, shuffle=True):
    means = means if means is not None else generate_means(n_features, n_topics)
    data, labels = np.empty((n_samples, n_features)), np.empty(n_samples, dtype=np.int32)
    sample_counts = partition(n_samples, n_topics)
    data[:sample_counts[0]] = np.random.normal(means[0], scale, (sample_counts[0], n_features))
    labels[:sample_counts[0]] = 0
    for i in range(1, n_topics):
        min_idx, max_idx = i*sample_counts[i - 1], (i + 1)*sample_counts[i]
        data[min_idx:max_idx] = np.random.normal(means[i], scale, (sample_counts[i], n_features))
        labels[min_idx:max_idx] = i
    if shuffle:
        data, labels = shuffle_data(data, labels)
    return data, labels
