import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from PIL import Image
from tqdm import tqdm
import pandas as pd

def plot_data(data, legend=False):
    """ 
    For each column, plot a graph with one color per row (country in this case)
    On the x-axis, the years (1990-2022)
    Using subplots with two columns
    """
    x = np.arange(1990, 2023)
    Nb_cols = len(data.columns)
    fig, axs = plt.subplots((Nb_cols + 1) // 2, 2, figsize=(15, 10))
    axs = axs.flatten()

    lines = []
    labels = []

    for i, col in enumerate(data.columns[:]):
        for j, row in data.iterrows():
            line, = axs[i].plot(x, row[col], label=row.name)
            if i == 0:  # Collect labels only once
                lines.append(line)
                labels.append(row.name)
        axs[i].set_title(col)
    
    if legend:
        fig.legend(lines, labels, loc='center left', bbox_to_anchor=(1, 0.5))  # Add a single legend to the right
    
    plt.tight_layout()
    plt.show()


def plot_cluster(groups, data, legend=False):
    """ 
    Receives a list of groups (a group is a list of indexes of the data)
    For each column (criterion), plot a graph with one color per group
    On the x-axis, the years (1990-2022)
    Using subplots with two columns
    """
    x = np.arange(1990, 2023)
    Nb_cols = len(data.columns)
    fig, axs = plt.subplots((Nb_cols + 1) // 2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, col in enumerate(data.columns[:]):
        for j, group in enumerate(groups):
            color = plt.cm.tab10(j)
            for country in group:
                axs[i].plot(x, data.loc[country][col], label=country, color=color) 
        axs[i].set_title(col)
        # axs[i].set_xticks(x)
        # axs[i].set_xticklabels(x, rotation=45)
        if legend:
            axs[i].legend()  # Add legend here
    plt.tight_layout()
    plt.show()

def plot_cluster_phi(PHI, groups, legend=False):
    """
    Plot the net flow series for all alternatives
    """
    fig, ax = plt.subplots()
    # size
    fig.set_size_inches(10, 5)
    for i in range(len(groups)):
        color = plt.cm.tab10(i)
        for j in groups[i]:
            ax.plot(PHI.loc[j], label=j, color=color)
    ax.set_xlabel("Year")
    ax.set_ylabel("Net flow")
    ax.set_title("PHI scores for all alternatives")
    if legend:
        ax.legend()
    # ax.legend()
    plt.show()

def plot_phi_c_all(PHI_c_all, col_names, alt_names, labels=True, pos=3):
    """" 
    Returns subplots for each criteria
    - PHI_c_all: A list of k lists of N time series
    """
    K = len(PHI_c_all)
    N = len(PHI_c_all[0])
    plt.figure(figsize=(15, 10))
    x = np.arange(1990, 2023)
    for c in range(K):
        plt.subplot((K + 1) // 2, 2, c + 1)
        PHI_c = PHI_c_all[c]
        for i in range(N):
            plt.plot(x, PHI_c[i], label=alt_names[i])
        # plt.axhline(y=-0.8, color='white', linestyle='-')
        # plt.axhline(y=0.8, color='white', linestyle='-')
        plt.title(f"Phi_c {c+1} - {col_names[c]}")
        plt.xlabel("Year")
        plt.ylabel("Phi_c(a_i)")
        if c == K - pos:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
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

def plot_PHI(PHI, labels=True):
    """
    Plot the net flow series for all alternatives
    :param PHI: dataframe with alternatives as index and years as columns
    """
    if labels:
        score = {}
        for i in range(PHI.shape[0]):
            score[PHI.index[i]] = sum(PHI.iloc[i])
        score = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))

        # Sort the alternatives by score
        PHI = PHI.loc[list(score.keys())]

    alt_names = PHI.index

    fig, ax = plt.subplots()
    # size
    fig.set_size_inches(10, 5)
    for i in range(PHI.shape[0]):
        ax.plot(PHI.iloc[i], label=alt_names[i])
    ax.set_xlabel("Year")
    ax.set_ylabel("Net flow")
    ax.set_title("PHI scores for all alternatives")
    if labels:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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


if __name__ == "__main__":
    print("plot.py loaded")