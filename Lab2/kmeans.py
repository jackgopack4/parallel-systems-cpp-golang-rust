from sklearn.cluster import KMeans
import numpy as np

def calculate_kmeans(points, k):
    # Convert the list of points to a NumPy array
    data = np.array(points)

    # Create a KMeans instance
    kmeans = KMeans(n_clusters=k,init=np.array([data[3],data[2],data[0],data[7]]))

    # Fit the model to the data
    kmeans.fit(data)

    # Get the centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels

# Example usage
points = [[0.37454012, 0.95071431, 0.73199394],
[0.59865848, 0.15601864, 0.15599452],
 [0.05808361, 0.86617615, 0.60111501],
 [0.70807258, 0.02058449, 0.96990985],
 [0.83244264, 0.21233911, 0.18182497],
 [0.18340451, 0.30424224, 0.52475643],
 [0.43194502, 0.29122914, 0.61185289],
 [0.13949386, 0.29214465, 0.36636184],
 [0.45606998, 0.78517596, 0.19967378],
 [0.51423444, 0.59241457, 0.04645041]]

k = 4  # Number of clusters

centroids, labels = calculate_kmeans(points, k)

# Print the results
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
