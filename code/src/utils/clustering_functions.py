if __name__ == "__main__":
    from utils import *
else:
    from utils.utils import *

def K_Medoid_Eta(alternatives, distance_matrix, k=3, prototype_method="random", print_results=True, iter_max=100, seed=None):
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
        medoids = np.random.choice(alternatives, k, replace=False)
    
    elif prototype_method == "farthest":
        # Select the farthest alternatives from each other
        medoids = [alternatives[0]] # Start with the first alternative
        for _ in range(k-1):
            distances = [np.min([distance_matrix.loc[alternative, medoid] for medoid in medoids]) for alternative in alternatives]
            new_medoid = alternatives[np.argmax(distances)]
            medoids.append(new_medoid)
        medoids = np.array(medoids)
    
    elif prototype_method == "km++":
        # Select randomly the first medoid
        random_index = np.random.choice(len(alternatives))
        medoids = [alternatives[random_index]]
        for _ in range(k-1):
            distances = [np.min([distance_matrix.loc[alternative, medoid] for medoid in medoids]) for alternative in alternatives]
            new_medoid = alternatives[np.argmax(distances)]
            medoids.append(new_medoid)
        medoids = np.array(medoids)

    elif prototype_method == "seed":
        if seed:
            medoids = np.array(seed)
        else:
            raise ValueError("Seed medoids not provided")

    if print_results:
        print("Initial medoids:", medoids)

    # Initialize clusters
    clusters = {medoid: [] for medoid in medoids}

    # Initialize assignment check
    assigned = {alternative: False for alternative in alternatives}


    # When entering the loop, we just have the medoids and no assigned alternatives
    iter = 0
    converged = False
    while not converged and iter < iter_max:

        # Assign medoid to its cluster
        for medoid in medoids:
            clusters[medoid].append(medoid)
            assigned[medoid] = True


        # Assign each alternative to the closest medoid
        for alternative in alternatives:
            if not assigned[alternative]: # If not yet assigned, assign it to the closest medoid
                distances = [distance_matrix.loc[alternative, medoid] for medoid in medoids] # Take the distances to each medoid
                closest_medoid = medoids[np.argmin(distances)] # Take the medoid with the smallest distance
                clusters[closest_medoid].append(alternative) # Assign the alternative to the cluster of the closest medoid
                assigned[alternative] = True

        if print_results:
            print("Iteration", iter)
            print("Clusters:", clusters)
            print("Assigned:", all(assigned.values()))

        # Update medoids
        converged = True
        for medoid in medoids:
            cluster = clusters[medoid]

            # Compute the sum of the distance for each alternative in the cluster towards the other alternatives in the cluster
            distances = [np.sum([distance_matrix.loc[alternative, alternative2] for alternative2 in cluster]) for alternative in cluster]
            if len(distances) > 1:
                new_medoid = cluster[np.argmin(distances)] # Take the alternative with the smallest sum of distances
            else:
                new_medoid = medoid
            if new_medoid != medoid:
                # print("Medoid", medoid, "changed to", new_medoid, "in array", medoids)
                index = np.where(medoids == medoid)
                # print("Index:", index[0])
                medoids[index[0][0]] = new_medoid # Update the medoid in the list
                converged = False # If at least one medoid has changed, we have not converged
                clusters = {medoid: [] for medoid in medoids} # Reinitialize the cluster
                assigned = {alternative: False for alternative in alternatives} # Reinitialize the cluster assignment check
        iter += 1
    
    if iter_max == iter:
        if print_results:
            print("Max iterations reached, no convergence but assigning the alternatives to the closest last medoid computed:")
        for alternative in alternatives:
            if not assigned[alternative]:
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
        index = np.random.choice(series.shape[0])
        prototypes.append(series.iloc[index])
        for i in range(1, k):
            distances = []
            for j in range(series.shape[0]):
                if series.index[j] in [p.name for p in prototypes]:
                    distances.append(0)
                else:
                    distances.append(np.min([euclid_distance(series.iloc[j], p) for p in prototypes]))
            prototypes.append(series.iloc[np.argmax(distances)])
        return pd.DataFrame(prototypes)


def kMeans(series, k, max_it=100, distance_function=euclid_distance, random_selec=True):
    """ 
    kMeans clustering algorithm
    - series is a dataframe with the time series that we want to cluster
    - k is the number of clusters we want to create
    """ 

    # Select k random centroids
    centroids = select_prototype(series, k, random=random_selec)
    # Create a dictionary to store the clusters
    clusters = {i: [] for i in range(k)}
    # Create a dictionary to store the previous clusters
    old_clusters = {i: [] for i in range(k)}
    # Create a dictionary to store the distances between the centroids and the time series
    distances = {i: [] for i in range(k)}
    # Initialize the assignment of the time series to the clusters
    assignment = np.zeros(series.shape[0])
    for it in range(max_it):
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


