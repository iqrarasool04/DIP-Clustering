import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('income.csv')

#calculation of euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#clustering function
def k_means_clustering(data, k, epochs):
    centroids = data.sample(n=k, random_state=42)

    for epoch in range(epochs):
        data['cluster'] = data.apply(lambda row: np.argmin([euclidean_distance(row['Age'], row['Income($)'], c['Age'], c['Income($)']) for _, c in centroids.iterrows()]), axis=1)
        centroids = data.groupby('cluster').mean()[['Age', 'Income($)']]
        #plotting data
        plt.scatter(data['Age'], data['Income($)'], c=data['cluster'], cmap='viridis')
        plt.scatter(centroids['Age'], centroids['Income($)'], c='red', marker='X', s=200)
        plt.title(f'Iteration {epoch + 1}')
        plt.xlabel('Age')
        plt.ylabel('Income($)')
        plt.show()

    return data, centroids

#specifying values
K = 4
epochs = 3
result, final_centroids = k_means_clustering(df[['Age', 'Income($)']], K, epochs)
result.to_csv('clustered_data2.csv', index=False)
final_centroids.to_csv('final_centroids2.csv', index=False)