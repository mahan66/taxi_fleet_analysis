import seaborn as sns
from sklearn import cluster
import scipy.cluster as scipy_cluster
import numpy as np
import matplotlib.pyplot as plt

from analysis.visualization import Visualizer


class Models:
    @staticmethod
    def find_best_k_for_KMeans(df, show_plot=False, lat='start_lat', lon='start_lon', max_k=30):

        # select start latitude and start longitiude from the rides
        X = df[[lat, lon]]

        ## sum of squared error of the models
        ssd = []
        for i in range(1, max_k + 1):
            if len(X) >= i:
                model = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                model.fit(X)
                ssd.append(model.inertia_)

        ## best k: the lowest derivative
        k = [i * 100 for i in np.diff(ssd, 2)].index(min([i * 100 for i in np.diff(ssd, 2)]))

        ## plot
        if show_plot:
            fig, ax = plt.subplots()
            ax.plot(range(1, len(ssd) + 1), ssd)
            ax.axvline(k, ls='--', color="red", label="k = " + str(k))
            ax.set(title='The Elbow Method', xlabel='Number of clusters', ylabel="SSD Error")
            ax.legend()
            ax.grid(True)
            plt.show()

        return k

    @staticmethod
    def cluster_locations(df, lat='start_lat', lon='start_lon', method='kmeans', number_of_clusters=None,
                          show_plot=False, show_on_map=False):
        # define clustering method
        if method == 'kmeans':
            k = number_of_clusters
            if not k:
                k = Models.find_best_k_for_KMeans(df)
            model = cluster.KMeans(n_clusters=k, init='k-means++')

        X = df[[lat, lon]]

        ## clustering
        df_X = X
        df_X["cluster"] = model.fit_predict(X)

        ## find real centroids
        closest, distances = scipy_cluster.vq.vq(model.cluster_centers_, df_X.drop("cluster", axis=1).values)

        # define centroids
        df_X['centroids'] = 0
        for i in closest:
            df_X['centroids'].iloc[i] = 1

        ## add clustering info to the original dataset
        df.loc[:, ['cluster', 'centroids']] = df_X.loc[:, ['cluster', 'centroids']]
        # df['cluster'] = df_X['cluster']
        # df['centroids'] = df_X['centroids']

        ## plot
        if show_plot:
            fig, ax = plt.subplots()
            sns.scatterplot(x=lat, y=lon, data=df, palette=sns.color_palette("bright", k),
                            hue='cluster', size="centroids", size_order=[1, 0],
                            legend="brief", ax=ax).set_title('Clustering (k=' + str(k) + ')')

            centroids = model.cluster_centers_
            ax.scatter(centroids[:, 0], centroids[:, 1], s=50, c='black', marker="x")

        if show_on_map:
            Visualizer.plot_on_map(df)

        return df, model
