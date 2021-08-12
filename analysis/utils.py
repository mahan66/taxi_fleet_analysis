import numpy as np
from os.path import join, dirname, abspath
import webbrowser



class Utils:
    ROOT_DIR = dirname(dirname(abspath(__file__)))
    DATASET_DIR_NAME = 'taxicab_dataset'
    OUTPUT_DIR_NAME = 'output_files'
    @staticmethod
    def haversine_distance(latitude1: float, longitude1: float, latitude2: float, longitude2: float) -> float:
        EARTH_RADIUS = 6367
        KM2Mile = 0.621371
        lon1, lat1, lon2, lat2 = map(np.radians, [longitude1, latitude1, longitude2, latitude2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

        c = 2 * np.arcsin(np.sqrt(a))

        mile = EARTH_RADIUS * c * KM2Mile

        return mile

    @staticmethod
    def get_hour_bin(hour):
        hour_bin_number = (hour % 24 + 4) // 4
        return hour_bin_number

    @staticmethod
    def map_auto_open(map, file_name='map.html'):
        map_path = join(Utils.ROOT_DIR, Utils.OUTPUT_DIR_NAME, file_name)
        html_page = f'{map_path}'
        map.save(html_page)
        # open in browser.
        new = 2
        webbrowser.open(html_page, new=new)
