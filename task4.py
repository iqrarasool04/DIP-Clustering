import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

#calculation of euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#clustering function
def k_means_clustering(data, k, epochs):
    centroids = data.sample(n=k, random_state=42)

    for epoch in range(epochs):
        data['cluster'] = data.apply(lambda row: np.argmin([euclidean_distance(row['sepal length (cm)'], row['sepal width (cm)'], c['sepal length (cm)'], c['sepal width (cm)']) for _, c in centroids.iterrows()]), axis=1)
        centroids = data.groupby('cluster').mean()[['sepal length (cm)', 'sepal width (cm)']]
        # Plotting data
        plt.scatter(data['sepal length (cm)'], data['sepal width (cm)'], c=data['cluster'], cmap='viridis')
        plt.scatter(centroids['sepal length (cm)'], centroids['sepal width (cm)'], c='red', marker='X', s=200)
        plt.title(f'Iteration {epoch + 1}')
        plt.xlabel('sepal length (cm)')
        plt.ylabel('sepal width (cm)')
        plt.show()

    return data, centroids

#specifying values
K = 3
epochs = 3
result, final_centroids = k_means_clustering(iris_df[['sepal length (cm)', 'sepal width (cm)']], K, epochs)
result.to_csv('clustered_data5.csv', index=False)
final_centroids.to_csv('final_centroids5.csv', index=False)
