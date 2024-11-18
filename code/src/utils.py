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


def normalize_columns(data1, data2):
    """
    Normalize each columns of the dataframes data1 and data2 independently except the date column 
    such that the values are in [-1, 1]
    """
    data1_norm = data1.copy()
    data2_norm = data2.copy()
    for column in data1.columns[1:]:
        # Use the mean and standard deviation of the column of both dataframes to normalize
        mean = (data1[column].mean() + data2[column].mean()) / 2
        std = (data1[column].std() + data2[column].std()) / 2

        data1_norm[column] = (data1[column] - mean) / std
        data2_norm[column] = (data2[column] - mean) / std
    return data1_norm, data2_norm