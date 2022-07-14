import numpy as np
from Clustering import plot_clusters
from DataManagement import load_data
from Clustering import modified_kmeans
from DataManagement import write_clusters_to_json
from Clustering import modified_kmeans
from DataManagement import get_home


# Inputs
input_data_path = 'Data/Drivers_PickupsTemplate.json'
template_path = 'Data/ClustersOutputTemplate.json'

# Load Input Data
drivers, drivers_home, drivers_carload, pickups = load_data(input_data_path)
drivers_home[2, :] = drivers_home[1, :]+0.5  # TODO: remove from final version
drivers_carload = [10, 15, 60]  # TODO: remove from final version

# Clustering
clustering = modified_kmeans(drivers_home, drivers_carload)
clustering.fit(pickups)
cluster_labels = clustering.labels
n_points_per_cluster = clustering.n_points_per_cluster

# Plot Clusters
figure_name = 'Clusters sizes : ' + str(n_points_per_cluster)
plot_clusters(np.array(pickups), cluster_labels, drivers_home, x_label='', y_label='', figure_name=figure_name)


# Write clusters to file
write_clusters_to_json(cluster_labels, template_path)
