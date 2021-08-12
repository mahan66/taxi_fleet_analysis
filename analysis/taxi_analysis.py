import os
import datetime
from os.path import dirname, abspath, join

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis.utils import Utils
from analysis.models import Models

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
            Utils.haversine_distance(df['latitude'], df['longitude'], df['latitude'].shift(),
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
        df1.loc[:, 'periods'] = Utils.get_hour_bin(df1['hour'])

        df1['periods'].replace(self.periods, inplace=True)

        df = df1

        df = df.sample(frac=sample_frac, random_state=1)

        if show_plot:
            X = np.array(df[[self.LONGITUDE, self.LATITUDE]])
            plt.scatter(X[:, 0], X[:, 1], alpha=0.2, s=50)
        return df

    def recommend_next_location(self, cur_latitude, cur_longitude, df=None, model=None, prediction_method='kmeans',
                                show_plot=False, show_on_map=False, sample_frac=0.02, date_time=None):

        df = self.postprocess_occupied_ride_data(df, sample_frac=sample_frac, show_plot=show_plot)

        if date_time:
            cur_hour = date_time.hour
            df = df[df['hour'] == cur_hour]

        if not model:
            df, model = Models.cluster_locations(df, method=prediction_method, show_on_map=show_on_map,
                                               show_plot=show_plot)

        centroids = df.loc[df['centroids'] == 1].sort_values(by='cluster')
        number_of_centroids = centroids.shape[0]
        size_of_clusters = np.array([df[df['cluster'] == x].shape[0] for x in centroids['cluster']])

        number_of_recommendations = 3
        recommended_locations = []

        cur_lats = np.array([cur_latitude] * number_of_centroids, dtype=float)
        cur_lons = np.array([cur_longitude] * number_of_centroids, dtype=float)

        distances = Utils.haversine_distance(cur_lats, cur_lons, np.array(centroids[self.LATITUDE], dtype=float),
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

            df, model = Models.cluster_locations(df, method=prediction_method, show_on_map=show_on_map,
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
