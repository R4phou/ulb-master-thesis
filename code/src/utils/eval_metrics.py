if __name__ == "__main__":
    from utils import *
else:
    from utils.utils import *

def dunn_index_multivariate(clusters, data):
    """ 
        Compute the Dunn index for a clustering of multivariate time series data
        - clusters: list of lists of indexes of the time series in each cluster
        - data: the data set (dataframe with index as id of the time series), each cell is a np.array (time series)
    """
    def dunn_index_univariate(clusters, data):
        """
            Compute the Dunn index for a clustering of univariate time series data
            - clusters: list of lists of indexes of the time series in each cluster
            - data: the data set (dataframe with index as id of the time series), only one column where each cell is a np.array (time series)
        """
        centroids = []

        # Define the centroids of the clusters
        for cluster in clusters:
            centroid = np.zeros_like(data.iloc[0])
            for country in cluster:
                centroid += data.loc[country]
            centroid /= len(cluster)
            centroids.append(centroid)

        centroids = np.array(centroids)
        
        # Compute the distances between clusters
        inter_cluster_distances = []
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                inter_cluster_distances.append(np.linalg.norm(centroids[i] - centroids[j]))

        # Compute the diameter of each cluster
        cluster_diameters = []

        # Diameter = avg distance between all pairs of countries in the cluster
        for cluster in clusters:
            diameter = 0
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    diameter += np.linalg.norm(data.loc[cluster[i]] - data.loc[cluster[j]])
            if len(cluster) > 1:
                diameter = diameter / (len(cluster)*(len(cluster)-1)/2)
            else:
                diameter = -1
            cluster_diameters.append(diameter)


        # # Diameter = max distance between two countries in the cluster
        # for cluster in clusters:
        #     diameter = 0
        #     for i in range(len(cluster)):
        #         for j in range(i+1, len(cluster)):
        #             diameter = max(diameter, np.linalg.norm(data.loc[cluster[i]] - data.loc[cluster[j]]))
        #     cluster_diameters.append(diameter)

        dunn_index = min(inter_cluster_distances) / max(cluster_diameters)
        return dunn_index

    index = 0
    criterias = data.columns

    for criteria in criterias:
        index += dunn_index_univariate(clusters, data[criteria])

    return index / len(criterias)

def compute_silhouette_score(cluster_groups, dataset):
    """ 
    Compute the silhouette score of a clustering
    - cluster_groups: the clustering to evaluate (a list of lists of indices)
    - dataset: the data used for clustering (a dataframe) (each cell is a np.array (time series)), each column is a criterion

    Silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.
    The silhouette score ranges from -1 to 1. A value of 1 indicates that the object is well clustered, while a value of -1 indicates that the object is poorly clustered.

    For each point in a cluster, we compute:
    - a(i): the average distance between the point and all other points in the same cluster
    - b(i): the average distance between the point and all points in the nearest cluster
    - s(i): the silhouette score for the point, computed as (b(i) - a(i)) / max(a(i), b(i))
    
    The silhouette score for the clustering is the average of the silhouette scores for all points in all clusters.

    """
    def euclidean_distance(point1, point2):
        return np.sum(np.linalg.norm(point1 - point2))

    silhouette_scores_per_cluster = []
    for cluster in cluster_groups:
        silhouette_scores_for_points = []

        for current_point in cluster:
            # Compute a(i) and b(i) for each cluster (silhouette)
            intra_cluster_distance = 0

            if len(cluster) > 1: # Avoid division by zero (if singleton cluster)
                # Compute a(i) for each cluster (a(i) is the average distance to all other points in the same cluster)
                for other_point in cluster:
                    if other_point != current_point:
                        intra_cluster_distance += euclidean_distance(dataset.loc[current_point], dataset.loc[other_point])
                intra_cluster_distance /= len(cluster) - 1

            # Compute b(i) for each cluster (b(i) is the average distance to the nearest cluster)
            # Initialize b(i) to a large value
            nearest_cluster_distance = float("inf")
            for other_cluster in cluster_groups:
                if other_cluster != cluster:
                    average_distance_to_other_cluster = 0
                    for other_cluster_point in other_cluster:
                        average_distance_to_other_cluster += euclidean_distance(dataset.loc[current_point], dataset.loc[other_cluster_point])
                    average_distance_to_other_cluster /= len(other_cluster)
                    nearest_cluster_distance = min(nearest_cluster_distance, average_distance_to_other_cluster)

            # Compute silhouette score
            silhouette_score = (nearest_cluster_distance - intra_cluster_distance) / max(intra_cluster_distance, nearest_cluster_distance)
            silhouette_scores_for_points.append(silhouette_score)
        silhouette_scores_per_cluster.append(silhouette_scores_for_points)

    # Compute the average silhouette score for the clustering
    all_silhouette_scores = []
    for cluster_scores in silhouette_scores_per_cluster:
        all_silhouette_scores += cluster_scores
    overall_silhouette_score = np.mean(all_silhouette_scores)
    return overall_silhouette_score

def evaluate_results_on_data(results, data):
    """ 
    Evaluate the Dunn index of the clustering results on the data
    """
    dunn_indices = []
    for clusters in results:
        dunn_index = dunn_index_multivariate(clusters, data)
        dunn_indices.append(dunn_index)
    
    return dunn_indices

def evaluate_results_on_net_flow_scores(results, PHI_df):
    """ 
    Evaluate the Dunn index of the clustering results on the net flow scores
    """
    dunn_indices = []
    for clusters in results:
        dunn_index = dunn_index_multivariate(clusters, PHI_df)
        dunn_indices.append(dunn_index)
    
    return dunn_indices

def evaluate_results_on_mono_criteria(results, phi_c_all_df):
    """ 
    Evaluate the Dunn index of the clustering results on the mono criteria scores
    """
    dunn_indices = []
    for clusters in results:
        dunn_index = dunn_index_multivariate(clusters, phi_c_all_df)
        dunn_indices.append(dunn_index)
    
    return dunn_indices

def evaluate_result_repartition_on_data(p2km_clusters, gkm_clusters, km_clusters, data, method=evaluate_results_on_data, title="Dunn Index of the clustering on the data", metrics="Dunn index"):
    """ 
    Evaluate the Dunn index of the clustering results on the data
    """
    p2km_evaluations = np.array(method(p2km_clusters, data))
    gkm_evaluations = np.array(method(gkm_clusters, data))
    km_evaluations = np.array(method(km_clusters, data))


    # Plot the Boxplots in order to compare the Dunn index of the three methods
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([p2km_evaluations, gkm_evaluations, km_evaluations], labels=["P2KMeans", "G-KMedoid", "KMeans"])
    ax.set_title(title)
    ax.set_ylabel(metrics)
    ax.set_xlabel("Clustering method")
    plt.show()

def evaluate_silhouette_on_data(results, data):
    """ 
    Evaluate the Dunn index of the clustering results on the data
    """
    scores = []
    for clusters in results:
        score = compute_silhouette_score(clusters, data)
        scores.append(score)   
    return scores