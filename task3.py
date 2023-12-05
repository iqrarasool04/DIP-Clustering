import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('income.csv')

#calculation of euclidean distance
def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

#clustering function
def k_means_clustering(data, k, epochs):
    costs = []
    for K in range(1, k + 1):
        centroids = data.sample(n=K, random_state=42)
        for epoch in range(epochs):
            data['cluster'] = data.apply(lambda row: np.argmin([euclidean_distance(row['Age'], row['Income($)'], c['Age'], c['Income($)']) for _, c in centroids.iterrows()]), axis=1)
            centroids = data.groupby('cluster').mean()[['Age', 'Income($)']]
            #plotting data
            plt.scatter(data['Age'], data['Income($)'], c=data['cluster'], cmap='viridis')
            plt.scatter(centroids['Age'], centroids['Income($)'], c='red', marker='X', s=200)
            plt.title(f'K={K}, Iteration {epoch + 1}')
            plt.xlabel('Age')
            plt.ylabel('Income($)')
            plt.show()

        #calculation of cost
        cost = (1 / len(data)) * np.sum([np.sum((data[data['cluster'] == i][['Age', 'Income($)']] - centroids.iloc[i])**2, axis=1) for i in range(K)])
        costs.append(cost)
        print(f'Cost for K={K}: {cost}')

    #plotting cost
    plt.plot(range(1, k + 1), costs, marker='o')
    plt.title('K vs. Cost')
    plt.xlabel('K')
    plt.ylabel('Cost')
    plt.show()

    return data, centroids

#specifying values
K_max = 10
epochs = 20
result, final_centroids = k_means_clustering(df[['Age', 'Income($)']], K_max, epochs)
result.to_csv('clustered_data4.csv', index=False)
final_centroids.to_csv('final_centroids4.csv', index=False)
