if __name__ == "__main__":
    from utils import *
else:
    from utils.utils import *


def linear_P_c(a_i, a_j, c, P, Q):
    """ 
    Returns: The preference function within criteria c, linear version
    - a_i: a multi-criteria time series
    - a_j: a multi-criteria time series
    - c: the criteria identifier
    - P: the preference threshold
    - Q: the indifference threshold

    The values in the time series are between 0 and 1
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

    The values in P_c matrix are between 0 and 1
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

    The values in Phi_c are between -1 and 1
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

def gamma_ai_aj(ai, aj, i, j, PHI_c_all, W, L, criterias):
    """ 
    Compute the gamma value between two alternatives
    - ai: a multi-criteria time series
    - aj: a multi-criteria time series
    - i: the index of ai in the data
    - j: the index of aj in the data
    - PHI_c_all: A list of k lists of N time series
    - W: The weights of the criteria
    - L: The length of the time series
    - criterias: the names of the criteria

    Returns:
     - gamma(ai, aj): a time series of size L where each cell is the sum of the weighted differences between PHI_c(ai) and PHI_c(aj) for the times where ai is preferred to aj

    gamma(ai, aj) = sum_c (W_c * (PHI_c(ai) - PHI_c(aj))) but only for the times where ai is preferred to aj
    """
    # Initialize gamma list of L zeros
    gamma = [0 for _ in range(L)]
    for t in range(L): # For each time step
        # Check if the value of ai is preferred to aj
        c_nb=0
        for c in criterias:
            if ai[c][t] > aj[c][t]: # If ai has a higher value than aj (the value is the score)
                gamma[t] += W[c_nb]* (PHI_c_all[c_nb][i][t] - PHI_c_all[c_nb][j][t])
            c_nb += 1

    return gamma

def get_gamma_matrix(data, PHI_c_all, W):
    """
    Returns: The gamma matrix, size NxN where each cell is a time series
    - data: the multi-criteria time series data
    - PHI_c_all: A list of k lists of N time series
    - W: The weights of the criteria
    """
    criterias = data.columns[1:]

    L = data.iloc[0][criterias[1]].shape[0] # Length of the time series
    N = data.shape[0] # Number of time series/alternatives

    gamma_matrix = np.zeros((N, N, L))
    for i in range(N):
        for j in range(N):
            if i != j:
                gamma_matrix[i][j] = gamma_ai_aj(data.iloc[i], data.iloc[j], i, j, PHI_c_all, W, L, criterias)
    return gamma_matrix



def score_function(column, maximize=True):
    """ 
    Function that gets the datasets and modify the data to evaluate the scores of the different criterias
    - column: each value is a time series, if we want to minimize it, we have to invert it
    - mininimize: boolean, if we want to minimize the value
    """
    def invert_values(column):
        return [-x for x in column]

    if maximize:
        return column
    else:
        # Invert the values: the higher the value, the lower the score
        column = invert_values(column)
        # Get the minimum value of the column
        min_value = min([min(x) for x in column])
        # Add the minimum value to the column in order to have only positive values
        column = [x + abs(min_value) for x in column]
        return column


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
