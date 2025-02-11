if __name__ == "__main__":
    from utils import *
else:
    from utils.utils import *

def K_Medoid_Eta(alternatives, distance_matrix, k=3, prototype_method="random", print_results=True):
    """ 
    K-Medoid clustering algorithm using the Aggregated Eta matrix
        - alternatives: np.array of the alternatives names only
        - distance_matrix: pd.DataFrame of the distance matrix with index and columns as the alternatives names
        - k: the number of clusters

    Returns:
        - the medoids of the clusters
        - the clusters
    """
    # Initialize medoids
    if prototype_method == "random":
        medoids = np.random.choice(alternatives, k, replace=False) # Randomly select k alternatives

    elif prototype_method == "farthest":
        # Select the farthest alternatives from each other
        medoids = [alternatives[0]]
        for _ in range(k-1):
            distances = [np.min([distance_matrix.loc[alternative, medoid] for medoid in medoids]) for alternative in alternatives]
            new_medoid = alternatives[np.argmax(distances)]
            medoids.append(new_medoid)


    if print_results:
        print("Initial medoids:", medoids)

    # Initialize clusters
    clusters = {medoid: [] for medoid in medoids}

    iter = 0
    # Iterate until convergence
    converged = False
    while not converged and iter < 100:
        # print(f"Iteration {iter}")
        iter += 1

        # Assign each alternative to the closest medoid
        for alternative in alternatives:
            distances = [distance_matrix.loc[alternative, medoid] for medoid in medoids]
            closest_medoid = medoids[np.argmin(distances)]
            clusters[closest_medoid].append(alternative)

        # Update medoids
        converged = True
        for medoid in medoids:
            cluster = clusters[medoid]
            distances = [np.sum([distance_matrix.loc[alternative1, alternative2] for alternative1 in cluster]) for alternative2 in cluster] # Sum of distances to all other alternatives in the cluster
            new_medoid = cluster[np.argmin(distances)] # Alternative with the smallest sum of distances
            if new_medoid != medoid: # If the medoid has changed
                medoids[np.where(medoids == medoid)[0][0]] = new_medoid # Replace the medoid
                clusters = {medoid: [] for medoid in medoids} # Reset the clusters
                converged = False # The algorithm has not converged -> continue the iterations
                # print("New medoid:", new_medoid, " replaces Old medoid:", medoid)
                break # Stop the loop and start a new iteration, this stops the loop:
    
    if iter == 100:
        if print_results:
            print("The algorithm did not converge after 100 iterations, assigning the closest alternatives to the medoids")
        # Assign each alternative to the closest medoid
        for alternative in alternatives:
            distances = [distance_matrix.loc[alternative, medoid] for medoid in medoids]
            closest_medoid = medoids[np.argmin(distances)]
            clusters[closest_medoid].append(alternative)

    return medoids, clusters, iter


def euclid_distance(series1, series2):
    """ 
    Compute the euclidean distance between two time series
    """
    return np.linalg.norm(series1 - series2)

def manhattan_distance(series1, series2):
    """ 
    Compute the manhattan distance between two time series
    """
    return np.sum(np.abs(series1 - series2))


def dynamic_time_warping(series1, series2):
    from fastdtw import fastdtw
    """
    Compute the dynamic time warping distance between two time series
    """
    return fastdtw(series1, series2)[0]


def select_prototype(series, k, random=True):
    """ 
    Select the k prototypes from the series (dataframe)
        - random: if True, the prototypes are selected randomly
        - random: if False, the prototypes are selected using the K-medoids algorithm
    """
    if random:
        return series.sample(k)
    else:
        prototypes = []
        prototypes.append(series.iloc[0])
        for i in range(1, k):
            distances = []
            for j in range(series.shape[0]):
                if series.index[j] in [p.name for p in prototypes]:
                    distances.append(0)
                else:
                    distances.append(np.min([euclid_distance(series.iloc[j], p) for p in prototypes]))
            prototypes.append(series.iloc[np.argmax(distances)])
        return pd.DataFrame(prototypes)


def kMeans(series, k, max_it=1000, distance_function=euclid_distance):
    """ 
    kMeans clustering algorithm
    - series is a dataframe with the time series that we want to cluster
    - k is the number of clusters we want to create
    """ 

    # Select k random centroids
    centroids = select_prototype(series, k, random=False)
    # Create a dictionary to store the clusters
    clusters = {i: [] for i in range(k)}
    # Create a dictionary to store the previous clusters
    old_clusters = {i: [] for i in range(k)}
    # Create a dictionary to store the distances between the centroids and the time series
    distances = {i: [] for i in range(k)}
    # Initialize the assignment of the time series to the clusters
    assignment = np.zeros(series.shape[0])
    for it in tqdm(range(max_it)):
        # Update the assignment of the time series to the clusters
        for i in range(series.shape[0]):
            distances = [distance_function(series.iloc[i], centroids.iloc[j]) for j in range(k)]
            assignment[i] = np.argmin(distances)
        # Update the centroids
        for i in range(k):
            centroids.iloc[i] = series[assignment == i].mean()
        # Update the clusters
        for i in range(k):
            clusters[i] = series[assignment == i]
        # Update the old clusters
        for i in range(k):
            old_clusters[i] = series[assignment == i]
    return clusters


