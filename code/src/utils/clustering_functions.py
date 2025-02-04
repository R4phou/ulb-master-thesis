if __name__ == "__main__":
    from utils import *
else:
    from utils.utils import *

def K_Medoid_Eta(alternatives, distance_matrix, k=3, prototype_method="random"):
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
        print("The algorithm did not converge after 100 iterations, assigning the closest alternatives to the medoids")
        # Assign each alternative to the closest medoid
        for alternative in alternatives:
            distances = [distance_matrix.loc[alternative, medoid] for medoid in medoids]
            closest_medoid = medoids[np.argmin(distances)]
            clusters[closest_medoid].append(alternative)

    return medoids, clusters, iter


