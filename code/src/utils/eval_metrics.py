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
            diameter /= len(cluster)
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

def evaluate_result_repartition_on_data(p2km_clusters, gkm_clusters, km_clusters, data, method=evaluate_results_on_data, title="Dunn Index of the clustering on the data"):
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
    ax.set_ylabel("Dunn index")
    ax.set_xlabel("Clustering method")
    plt.show()