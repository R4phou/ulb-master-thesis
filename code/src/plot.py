import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image

def plot_data(data, legend=False):
    """ 
    For each column, plot a graph with one color per row (country in this case)
    On the x-axis, the years (1990-2022)
    Using subplots with two columns
    """
    x = np.arange(1990, 2023)
    Nb_cols = len(data.columns) - 1
    fig, axs = plt.subplots((Nb_cols + 1) // 2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, col in enumerate(data.columns[1:]):
        for j, row in data.iterrows():
            axs[i].plot(x, row[col], label=row["iso3"]) 
        axs[i].set_title(col)
        # axs[i].set_xticks(x)
        # axs[i].set_xticklabels(x, rotation=45)
        if legend:
            axs[i].legend()  # Add legend here
    plt.tight_layout()
    plt.show()


def plot_phi_c_all(PHI_c_all, col_names, alt_names, labels=True):
    """"
    Returns subplots for each criteria
    - PHI_c_all: A list of k lists of N time series
    """
    K = len(PHI_c_all)
    N = len(PHI_c_all[0])
    plt.figure(figsize=(15, 5))
    x = np.arange(1990, 2023)
    for c in range(K):
        plt.subplot((K + 1) // 2, 2, c + 1)
        PHI_c = PHI_c_all[c]
        for i in range(N):
            plt.plot(x, PHI_c[i], label=alt_names[i])
        plt.title(f"PHI {c+1} - {col_names[c]}")
        if labels:
            plt.legend()
    plt.tight_layout()
    plt.show()

def plot_Phi_c_ai(PHI_c, title, labels=True):
    """
    PHI_c is a list of N time series, plot each time series in the same plot
    """
    x = np.arange(1990, 2023)
    N = PHI_c.shape[0]
    for i in range(N):
        plt.plot(x, PHI_c[i], label=f"a_{i}")
    plt.title(title)
    if labels:
        plt.legend()
    plt.show()

def plot_PHI(PHI, alt_names, labels=True):
    """
    Plot the aggregated preference function
    """
    x = np.arange(1990, 2023)
    N = PHI.shape[0]
    plt.figure(figsize=(15, 5))
    for i in range(N):
        plt.plot(x, PHI[i], label=alt_names[i])
    plt.title("PHI")
    plt.grid()
    if labels:
        plt.legend()
    plt.show()


def plot_gammas(gamma_matrix, alt_names):
    """
    Plot the gamma matrix for each pair of alternatives
    """
    N = gamma_matrix.shape[0]
    x = np.arange(1990, 2023)
    fig, axes = plt.subplots(N, 1, figsize=(15, 5*N))
    for i in range(N):
        for j in range(N):
            if i != j:
                axes[i].plot(x, gamma_matrix[i][j], label=f"{alt_names[j]}")
                axes[i].set_title(f"Gamma: {alt_names[i]} > others")
                axes[i].grid()
                axes[i].legend()
    # plt.tight_layout()
    plt.show()