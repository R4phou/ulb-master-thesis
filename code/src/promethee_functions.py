from utils import *


def linear_P_c(a_i, a_j, c, P, Q):
    """ 
    Returns: The preference function within criteria c, linear version
    - a_i: a multi-criteria time series
    - a_j: a multi-criteria time series
    - c: the criteria identifier
    - P: the preference threshold
    - Q: the indifference threshold
    """
    d = a_i[c] - a_j[c]
    for i in range(len(d)): # For each time step
        if d[i] <= Q[c-1]:
            d[i] = 0
        elif d[i] > P[c-1]:
            d[i] = 1
        else:
            d[i] = (d[i] - Q[c-1]) / (P[c-1] - Q[c-1])
    return d

def P_c_matrix(data, c, P, Q):
    """
    Returns: The preference matrix within criteria c, size NxN where each cell is a time series
    - data: the multi-criteria time series data
    - c: the criteria identifier
    """
    L = data.iloc[0]["co2prod"].shape[0] #lenght of the time series
    N = data.shape[0]
    P_c = np.zeros((N, N, L))
    for i in range(N):
        for j in range(N):
            if i != j:
                P_c[i][j] = linear_P_c(data.iloc[i], data.iloc[j], c+1, P, Q)
    return P_c

def get_Phi_c_ai(i, P_c):
    """
    Returns: The preference function for a_i within criteria c
    Phi_c(a_i) = 1/N-1 sum_j (P_c(a_i, a_j) - P_c(a_j, a_i))
    - a_i: a multi-criteria time series
    - P_c: the preference matrix within criteria c
    - c: the criteria identifier
    """
    N = P_c.shape[0]
    sum = 0
    for j in range(N):
        sum += P_c[i][j] - P_c[j][i] 
    return 1/(N-1) * sum


def get_Phi_c(data, c, P, Q):
    """
    Returns: The preference function for all time series within criteria c
        - Phi_c is a list of size N where each cell is a time series
    - data: the multi-criteria time series data
    - c: the criteria identifier
    """
    L = data.iloc[0]["co2prod"].shape[0] #lenght of the time series
    P_c = P_c_matrix(data, c, P, Q)
    N = data.shape[0]
    Phi_c = np.zeros((N, L))
    for i in range(N):
        Phi_c[i] = get_Phi_c_ai(i, P_c)
    return Phi_c


def get_all_Phi_c(data, P, Q):
    """
    Returns: A list of all preference functions for all criteria, K is the number of criteria
    """
    K = data.columns.shape[0] -1 # Number of criteria
    return [get_Phi_c(data, c, P, Q) for c in range(K)]


def PHI_all(PHI_c_all, W, N, L, K):
    """
    Returns: The aggregated preference function
    PHI = sum(W_c * PHI_c)
    """
    PHI = np.zeros((N, L))
    for c in range(K):
        PHI += W[c] * PHI_c_all[c]
    return PHI

def get_PHI(data, W, P, Q):
    """
    Returns: The aggregated preference function for all time series
    """
    K = data.columns.shape[0] -1 # Number of criteria
    N = data.shape[0]
    L = data.iloc[0]["co2prod"].shape[0] #lenght of the time series
    PHI_c_all = get_all_Phi_c(data, P, Q)
    alternate_names = data["iso3"].values
    plot_phi_c_all(PHI_c_all, data.columns[1:], alternate_names, labels=False)
    return PHI_all(PHI_c_all, W, N, L, K)


def main():
    data = read_data()
    data = scale_data(data)
    plot_data(data)
    K = data.columns.shape[0] -1 # Number of criteria
    W = [1/K for _ in range(K)]
    P = [0.9 for _ in range(K)]
    Q = [0.1 for _ in range(K)]
    PHI = get_PHI(data, W, P, Q)
    alt_names = data["iso3"].values
    plot_PHI(PHI, alt_names, labels=False)

if __name__ == "__main__":
    main()
