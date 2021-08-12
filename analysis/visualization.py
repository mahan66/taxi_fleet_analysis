from sklearn import preprocessing
from analysis.utils import Utils
import folium
import numpy as np

class Visualizer:
    @staticmethod
    def plot_on_map(df, lat='start_lat', lon='start_lon'):
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
        Utils.map_auto_open(map_)
