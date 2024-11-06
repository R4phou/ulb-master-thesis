import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def read_data(path):
    data = pd.read_csv(path)
    return data


def normalize_list(lst):
    """
        Normalize a list
    """
    return [(x) / (max(lst)) for x in lst]


def normalize_matrix_columns(matrix):
    """
        Normalize the columns of a matrix
    """
    for col in range(matrix.shape[1]):
        matrix[:, col] = normalize_list(matrix[:, col])
    return matrix


def plot_heatmap(data, title):
    """
        Plot a heatmap of the data
        data : matrix to plot
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_title(title)
    plt.show()