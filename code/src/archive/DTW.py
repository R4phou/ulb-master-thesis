"""
Dynamic Time Warping (DTW) algorithm
"""

import numpy as np
import matplotlib.pyplot as plt


# Define the Time Series we need the distance from
Y = [3,1,2,2,1, 8, 9, 3, 2, 5]
Z = [2,0,0,3,3,1,0, 6, 2, 3]


def dist(x, y):
    """Compute the euclidean distance between two scalars"""
    return (x - y) ** 2

def compute_distMatrix(X,Y):
    """Compute a matrix of size(X)*size(Y) containing the euclidean distance between each pair of points"""
    distMatrix = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            distMatrix[i, j] = dist(X[i], Y[j])
    return distMatrix

def DTW_matrix(distMatrix):
    """Compute the Dynamic Time Warping matrix using dynamic programming
        - gamma_ij = distMatrix_ij + min(gamma_i-1,j, gamma_i,j-1, gamma_i-1,j-1)
    """
    N, M = distMatrix.shape
    gamma = np.zeros((N, M))
    gamma[0, 0] = distMatrix[0, 0]

    # Initialize the first row and column
    for i in range(1, N): 
        gamma[i, 0] = distMatrix[i, 0] + gamma[i - 1, 0]
    for j in range(1, M):
        gamma[0, j] = distMatrix[0, j] + gamma[0, j - 1]

    # Compute the rest of the matrix using the 
    for i in range(1, N):
        for j in range(1, M):
            gamma[i, j] = distMatrix[i, j] + min(gamma[i - 1, j], gamma[i, j - 1], gamma[i - 1, j - 1])
    return gamma

def plot_heatmap_matrix(matrix, title):
    """Plot a heatmap of the matrix"""
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_title(title)
    plt.show()


def DTW(X, Y):
    """Compute the Dynamic Time Warping distance between two time series"""
    distMatrix = compute_distMatrix(X,Y)
    gamma = DTW_matrix(distMatrix)
    dtw = np.sqrt(gamma[-1, -1])
    return dtw

def dtw_path(dtw_matrix):
    """Receives the DTW matrix and returns the path of the warping"""
    N, M = dtw_matrix.shape
    path = []
    i, j = N - 1, M - 1
    while i > 0 and j > 0:
        path.append((i, j))
        if i == 0:
            j = j - 1
        elif j == 0:
            i = i - 1
        else:
            if dtw_matrix[i - 1, j] == min(dtw_matrix[i - 1, j - 1], dtw_matrix[i - 1, j], dtw_matrix[i, j - 1]):
                i = i - 1
            elif dtw_matrix[i, j - 1] == min(dtw_matrix[i - 1, j - 1], dtw_matrix[i - 1, j], dtw_matrix[i, j - 1]):
                j = j - 1
            else:
                i = i - 1
                j = j - 1
    path.append((0, 0))
    return path



def plot_warping_path(X, Y, path):
    """Plot the warping path on the two time series"""
    plt.plot(X, label='X', color='blue')
    plt.plot(Y, label='Y', color='green')
    for (map_x, map_y) in path:
        plt.plot([map_x, map_y], [X[map_x], Y[map_y]], color='red', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()

def DTW(X, Y, plotting=False):
    """Compute the Dynamic Time Warping distance between two time series"""
    distMatrix = compute_distMatrix(X,Y)
    gamma = DTW_matrix(distMatrix)
    dtw = np.sqrt(gamma[-1, -1])
    if plotting:
        path = dtw_path(gamma)
        plot_warping_path(X, Y, path)
    return dtw

if __name__ == "__main__":
    dtw = DTW(Y, Z, plotting=True)

    print("DTW distance between Y and Z: ", dtw)


