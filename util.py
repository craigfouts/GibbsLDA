import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def map_labels(X_labels, Y_labels):
    scores = confusion_matrix(Y_labels, X_labels)
    row, col = linear_sum_assignment(scores, maximize=True)
    labels = np.empty_like(X_labels)
    for i in row:
        labels[Y_labels == i] = col[i]
    return labels

def evaluate(X_labels, Y_labels):
    Y_labels = map_labels(X_labels, Y_labels)
    score = (X_labels == Y_labels).sum()/X_labels.shape[0]
    return Y_labels, score

def visualize_data(data, labels, title='Title', size=32, show_ax=True):
    _, ax = plt.subplots()
    if not show_ax:
        ax.axis('off')
    ax.scatter(data[:, 0], data[:, 1], s=size, c=labels)
    plt.title(title)
    plt.show()

def visualize_log(log, title='Title', x_label='X-Label', y_label='Y-Label'):
    x = np.arange(len(log))
    plt.plot(x, log)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()
