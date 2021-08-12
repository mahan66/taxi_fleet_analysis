import os
import datetime
from os.path import dirname, abspath, join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import folium
from IPython.core.display import display, HTML
import webbrowser

from sklearn import preprocessing, cluster
import scipy.cluster as scipy_cluster


# import minisom


class TaxiFleetAnalyser:
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    DATASET_DIR_NAME = 'taxicab_dataset'
    OUTPUT_DIR_NAME = 'output_files'
    TAXI_DATA_FILE_NAME = 'taxi_data.csv'
    RIDE_DATA_FILE_NAME = 'ride_data.csv'
    LATITUDE = 'start_lat'
    LONGITUDE = 'start_lon'

    def __init__(self):
        self.taxi_data = None
        self.periods = {1: 'Late Night', 2: 'Early Morning', 3: 'Morning', 4: 'Noon', 5: 'Evening', 6: 'Night'}

    def read_dataset(self, get_all_data=False, number_of_cabs=5, save_to_file=False,
                     taxi_data_file_name=TAXI_DATA_FILE_NAME):
        dataset_dir = join(TaxiFleetAnalyser.ROOT_DIR, TaxiFleetAnalyser.DATASET_DIR_NAME)

        df = []
        i = 0
        for file_name in os.listdir(dataset_dir):
            if file_name.endswith(".txt"):
                if not get_all_data and i == number_of_cabs:
                    break

                cur_file_path = join(dataset_dir, file_name)
                cur_df = pd.read_csv(cur_file_path, sep=' ', names=['latitude', 'longitude', 'occupancy', 'time'])
                cur_df['date_time'] = pd.to_datetime(cur_df['time'], unit='s')
                cur_df.drop('time', axis=1, inplace=True)

                taxi_title = file_name.split('.')[0]
                cur_df['taxi_title'] = taxi_title

                df.append(cur_df)
                i += 1

        df = pd.concat(df)
        self.taxi_data = df

        if save_to_file:
            output_dir = join(TaxiFleetAnalyser.ROOT_DIR, TaxiFleetAnalyser.OUTPUT_DIR_NAME)
            taxi_data_path = join(output_dir, taxi_data_file_name)
            df.to_csv(taxi_data_path, index=False)

        return self.taxi_data

    def haversine_distance(self, latitude1: float, longitude1: float, latitude2: float, longitude2: float) -> float:
        EARTH_RADIUS = 6367
        KM2Mile = 0.621371
        lon1, lat1, lon2, lat2 = map(np.radians, [longitude1, latitude1, longitude2, latitude2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

        c = 2 * np.arcsin(np.sqrt(a))

        mile = EARTH_RADIUS * c * KM2Mile

        return mile

    def add_features_for_each_ride(self, df):
        cur_ride = pd.DataFrame()

        cur_ride['taxi_title'] = [df.iloc[0]['taxi_title']]
        cur_ride['occupancy'] = [df.iloc[0]['occupancy']]

        cur_ride['start_lat'] = [df.iloc[0]['latitude']]
        cur_ride['start_lon'] = [df.iloc[0]['longitude']]

        cur_ride['end_lat'] = [df.iloc[-1]['latitude']]
        cur_ride['end_lon'] = [df.iloc[-1]['longitude']]

        cur_ride['start_time'] = [df.iloc[0]['date_time']]
        cur_ride['end_time'] = [df.iloc[-1]['date_time']]

        # distance (Mile)
        cur_ride['distance'] = [
            self.haversine_distance(df['latitude'], df['longitude'], df['latitude'].shift(),
                                    df['longitude'].shift()).sum()]

        # duration (Sec)
        cur_ride['duration'] = [df['date_time'].diff().dt.total_seconds().sum()]

        # speed (Mile/Hour)
        cur_ride['speed'] = cur_ride['distance'] / (cur_ride['duration'] / 3600.0)

        return cur_ride

    def extract_ride_data(self, df):
        df['ride_index'] = (df['occupancy'].shift() != df['occupancy']).cumsum()

        ride_list = []
        ride_list.append(df.groupby('ride_index').apply(self.add_features_for_each_ride))
        return pd.concat(ride_list)

    def postprocess_ride_data(self, df):
        ride_data = df.droplevel(level=[0, 2])
        ride_data = ride_data.set_index([list(range(1, ride_data.shape[0] + 1))])

        return ride_data

    def extract_all_ride_data(self, df, save_to_file=False, ride_data_file_name=RIDE_DATA_FILE_NAME):
        data_list = []
        df.append(df.groupby('taxi_title').apply(self.extract_ride_data))

        df = pd.concat(data_list)

        df = self.postprocess_ride_data(df)

        if save_to_file:
            output_dir = join(TaxiFleetAnalyser.ROOT_DIR, TaxiFleetAnalyser.OUTPUT_DIR_NAME)
            ride_data_path = join(output_dir, ride_data_file_name)
            df.to_csv(ride_data_path, index=False)

        return df

    def calculate_potential_emission_reduction(self, df, emission_per_mile=404, taxi_change_rate=0.15):
        number_of_distinct_taxies = len(df.groupby('taxi_title').size())
        sum_of_distances_of_all_empty_taxies_per_month = df[
            (df['occupancy'] == 0) & (df['distance'] != 0)].distance.sum()

        average_emission_per_taxi = sum_of_distances_of_all_empty_taxies_per_month / number_of_distinct_taxies * emission_per_mile

        monthly_average_emission_for_all_taxies = number_of_distinct_taxies * average_emission_per_taxi

        potenital_yearly_emission_reduction = (monthly_average_emission_for_all_taxies * (
                1 - (1 - taxi_change_rate) ** 12)) / taxi_change_rate

        print(
            'The potential for a yearly reduction in CO2 emissions, caused by empty roaming taxi cabs is about {0:.2f} metric tone.'
                .format(potenital_yearly_emission_reduction / (1000000)))
        return potenital_yearly_emission_reduction

    def get_hour_bin(self, hour):
        hour_bin_number = (hour % 24 + 4) // 4
        return hour_bin_number

    def postprocess_occupied_ride_data(self, df, quant_val=0.0001, sample_frac=0.02, show_plot=False):

        df = df[df['occupancy'] == 1]

        df1 = df[['start_lat', 'start_lon']]
        df1 = df1[(df1.quantile(quant_val) < df1) & (df1 < df1.quantile(1 - quant_val))]
        df1 = df1.dropna(how='any')

        df1.loc[:, 'taxi_title'] = df['taxi_title']
        df1.loc[:, 'distance'] = df['distance']
        df1.loc[:, 'duration'] = df['duration']
        df1.loc[:, 'day_of_month'] = df['start_time'].dt.day
        df1.loc[:, 'day_of_week'] = df['start_time'].dt.dayofweek
        df1.loc[:, 'hour'] = df['start_time'].dt.hour
        df1.loc[:, 'periods'] = self.get_hour_bin(df1['hour'])

        df1['periods'].replace(self.periods, inplace=True)

        df = df1

        df = df.sample(frac=sample_frac, random_state=1)

        if show_plot:
            X = np.array(df[[self.LONGITUDE, self.LATITUDE]])
            plt.scatter(X[:, 0], X[:, 1], alpha=0.2, s=50)
        return df

    def map_auto_open(self, map, file_name='map.html'):
        map_path = join(TaxiFleetAnalyser.ROOT_DIR, TaxiFleetAnalyser.OUTPUT_DIR_NAME, file_name)
        html_page = f'{map_path}'
        map.save(html_page)
        # open in browser.
        new = 2
        webbrowser.open(html_page, new=new)

    def plot_on_map(self, df, lat=LATITUDE, lon=LONGITUDE):
        color = "cluster"
        size = "size"
        popup = 'taxi_title'
        marker = "centroids"

        df[size] = 3
        data = df

        ## create color column
        lst_elements = sorted(list(df[color].unique()))
        lst_colors = ['#%06X' % np.random.randint(0, 0xFFFFFF) for i in range(len(lst_elements))]
        data["color"] = data[color].apply(lambda x: lst_colors[lst_elements.index(x)])

        ## create size column (scaled)
        scaler = preprocessing.MinMaxScaler(feature_range=(3, 15))
        data["size"] = scaler.fit_transform(data[size].values.reshape(-1, 1)).reshape(-1)

        ## initialize the map with the starting location
        map_ = folium.Map(location=[data[lat].mean(), data[lon].mean()], tiles="cartodbpositron", zoom_start=11)

        ## add points
        data.apply(lambda row: folium.CircleMarker(
            location=[row[lat], row[lon]], popup=row[popup],
            color=row["color"], fill=True,
            radius=row["size"]).add_to(map_), axis=1)

        ## add html legend
        legend_html = """<div style="position:fixed; bottom:10px; left:10px; border:2px solid black; z-index:9999; font-size:14px;">&nbsp;<b>""" + color + """:</b><br>"""
        for i in lst_elements:
            legend_html = legend_html + """&nbsp;<i class="fa fa-circle 
          fa-1x" style="color:""" + lst_colors[lst_elements.index(i)] + """">
          </i>&nbsp;""" + str(i) + """<br>"""
        legend_html = legend_html + """</div>"""
        map_.get_root().html.add_child(folium.Element(legend_html))
        ## add centroids marker
        lst_elements = sorted(list(df[marker].unique()))
        data[data[marker] == 1].apply(lambda row: folium.Marker(location=[row[lat], row[lon]],
                                                                popup=str(row[marker]), draggable=False,
                                                                icon=folium.Icon(color="black")).
                                      add_to(map_), axis=1)
        ## plot the map
        # display(map_)
        self.map_auto_open(map_)

    def find_best_k_for_KMeans(self, df, show_plot=False, lat=LATITUDE, lon=LONGITUDE, max_k=30):

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

    def cluster_locations(self, df, lat=LATITUDE, lon=LONGITUDE, method='kmeans', number_of_clusters=None,
                          show_plot=False, show_on_map=False):
        # define clustering method
        if method == 'kmeans':
            k = number_of_clusters
            if not k:
                k = self.find_best_k_for_KMeans(df)
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
            self.plot_on_map(df)

        return df, model

    def recommend_next_location(self, cur_latitude, cur_longitude, df=None, model=None, prediction_method='kmeans',
                                show_plot=False, show_on_map=False, sample_frac=0.02, date_time=None):

        df = self.postprocess_occupied_ride_data(df, sample_frac=sample_frac, show_plot=show_plot)

        if date_time:
            cur_hour = date_time.hour
            df = df[df['hour'] == cur_hour]

        if not model:
            df, model = self.cluster_locations(df, method=prediction_method, show_on_map=show_on_map,
                                               show_plot=show_plot)

        centroids = df.loc[df['centroids'] == 1].sort_values(by='cluster')
        number_of_centroids = centroids.shape[0]
        size_of_clusters = np.array([df[df['cluster'] == x].shape[0] for x in centroids['cluster']])

        number_of_recommendations = 3
        recommended_locations = []

        cur_lats = np.array([cur_latitude] * number_of_centroids, dtype=float)
        cur_lons = np.array([cur_longitude] * number_of_centroids, dtype=float)

        distances = self.haversine_distance(cur_lats, cur_lons, np.array(centroids[self.LATITUDE], dtype=float),
                                            np.array(centroids[self.LONGITUDE], dtype=float))

        closest_clusters = centroids.iloc[np.argsort(distances)][:number_of_recommendations]

        closest_locations = closest_clusters[['start_lat', 'start_lon']]
        closest_locations.loc[:, 'distance'] = distances[closest_clusters['cluster']]
        closest_locations.loc[:, 'density'] = size_of_clusters[closest_clusters['cluster']]

        closest_locations = closest_locations[[self.LATITUDE, self.LONGITUDE, 'distance', 'density']]

        return closest_locations, df, model

    def cluster_taxi_cabs(self, df, model=None, sample_frac=0.02, show_plot=False, prediction_method='kmeans',
                          show_on_map=False, date_time = None, hour_bin = None):

        if not model:
            df = self.postprocess_occupied_ride_data(df, sample_frac=sample_frac, show_plot=show_plot)
            if hour_bin:
                df = df[df['periods'] == hour_bin]

            df, model = self.cluster_locations(df, method=prediction_method, show_on_map=show_on_map,
                                               show_plot=show_plot)

        # cluster taxi cabs based on the number taxi cabs in the clusters of their pickup locations
        temp_df = pd.DataFrame()
        i = 0
        for k, cluster in df.groupby('cluster'):
            for taxi_title in cluster['taxi_title'].unique():
                temp_df.loc[i, 'cluster'] = k
                temp_df.loc[i, 'taxi_title'] = taxi_title
                temp_df.loc[i, 'count'] = cluster[cluster['taxi_title'] == taxi_title].shape[0]
                i += 1

        number_of_taxi_cabs_in_each_pickup_cluster = temp_df.sort_values(by=['cluster', 'count'], ascending=False)


        return number_of_taxi_cabs_in_each_pickup_cluster


def main():
    analyser = TaxiFleetAnalyser()

    ###  ------------- Read and Process Raw Taxi Data ----------------- ###
    ## Taxi Data
    # df = analyser.read_dataset(get_all_data=True, save_to_file=True)
    # taxi_data_path = join(analyser.ROOT_DIR, analyser.OUTPUT_DIR_NAME, analyser.TAXI_DATA_FILE_NAME)
    # taxi_data = pd.read_csv(taxi_data_path, parse_dates=['date_time'])
    ###  ------------------------------------------ ###

    ###  ------------- Extract Features from Taxi Data ----------------- ###
    # ride_data = analyser.extract_all_ride_data(taxi_data, save_to_file=True)
    ride_data_path = join(analyser.ROOT_DIR, analyser.OUTPUT_DIR_NAME, 'ride_data2.csv')
    ride_data = pd.read_csv(ride_data_path, parse_dates=['start_time', 'end_time'])
    ###  ------------------------------------------ ###

    ###  ------------- Question 1 ----------------- ###
    # To calculate the potential for a yearly reduction in CO2 emissions,
    # caused by the taxi cabs roaming without passengers.
    # In your calculation please assume that the taxicab fleet is changing at the rate of 15% per month
    # (from combustion engine-powered vehicles to electric vehicles).
    # Assume also that the average passenger vehicle emits about 404 grams of CO2 per mile.
    ###  ------------------------------------------ ###
    # potenital_yearly_emission_reduction = analyser.calculate_potential_emission_reduction(ride_data, taxi_change_rate=0.15)

    ###  ------------- Question 2 ----------------- ###
    # To build a predictor for taxi drivers, predicting the next place a passenger will hail a cab.
    ###  ------------------------------------------ ###
    recommended_locations, cur_df, model = analyser.recommend_next_location(37.75153, -122.39447, df=ride_data,
                                                                            show_plot=True, show_on_map=False,
                                                                            date_time = datetime.datetime.now())
    print(recommended_locations)

    ###  ------------- Question 3 ----------------- ###
    # (Bonus question) Identify clusters of taxi cabs that you find being relevant
    # from the taxi cab company point of view.
    ###  ------------------------------------------ ###
    taxi_clusters = analyser.cluster_taxi_cabs(ride_data, hour_bin = analyser.periods[3])
    print(taxi_clusters)


if __name__ == '__main__':
    main()
