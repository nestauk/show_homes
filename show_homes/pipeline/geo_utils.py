import numpy as np
import requests
import json

from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist  # avoid naming confusion

import pandas as pd

from keplergl import KeplerGl
import yaml


def get_travel_distance_and_duration(coords_1, coords_2):

    lat_1, lon_1 = coords_1
    lat_2, lon_2 = coords_2

    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{lon_1},{lat_1};{lon_2},{lat_2}?overview=false"
        ""
    )

    routes = json.loads(r.content)
    if routes.get("routes") is None:
        return None
    route_1 = routes.get("routes")[0]

    duration = route_1["duration"] / 60
    distance = route_1["distance"] / 1000

    return duration, distance


def to_Cartesian(lat, lng):
    R = 6367  # radius of the Earth in kilometers

    x = R * np.cos(lat) * np.cos(lng)
    y = R * np.cos(lat) * np.sin(lng)
    z = R * np.sin(lat)
    return np.array((x, y, z)).T


def prepare_coords(df):

    coords_org = df[["LATITUDE", "LONGITUDE"]].to_numpy()

    coords = coords_org.copy()
    coords = np.deg2rad(coords)
    coords_cart = to_Cartesian(coords[:, 0], coords[:, 1])

    return coords_cart, coords_org


def create_output_map(connections_df, host_df, visitor_df, settings_string):

    config_file = "network_gradio_config.txt"
    with open(config_file, "r") as infile:
        config = infile.read()
        config = yaml.load(config, Loader=yaml.FullLoader)

    map_dict = {0.0: "no", 1.0: "yes"}

    host_df["Matched"] = host_df["Matched"].map(map_dict)
    visitor_df["Matched"] = visitor_df["Matched"].map(map_dict)

    new_row_1_host = pd.DataFrame(
        {
            "LATITUDE": -25.27729575389078,
            "LONGITUDE": 131.07289492391908,
            "Visitor matches (capped)": 0,
            "Visitor matches": 0,
            "Matched": "no",
        },
        index=[0],
    )

    new_row_2_host = pd.DataFrame(
        {
            "LATITUDE": -27.27729575389078,
            "LONGITUDE": 131.07289492391908,
            "Visitor matches (capped)": 0,
            "Visitor matches": 0,
            "Matched": "yes",
        },
        index=[0],
    )

    new_row_1_visitor = pd.DataFrame(
        {
            "LATITUDE": -25.27729575389078,
            "LONGITUDE": 131.07289492391908,
            "Host matches (capped)": 0,
            "Host matches": 0,
            "Matched": "no",
        },
        index=[0],
    )

    new_row_2_visitor = pd.DataFrame(
        {
            "LATITUDE": -27.27729575389078,
            "LONGITUDE": 131.07289492391908,
            "Host matches (capped)": 0,
            "Host matches": 0,
            "Matched": "yes",
        },
        index=[0],
    )

    host_df = pd.concat([new_row_1_host, new_row_2_host, host_df]).reset_index(
        drop=True
    )
    visitor_df = pd.concat(
        [new_row_1_visitor, new_row_2_visitor, visitor_df]
    ).reset_index(drop=True)

    print(host_df["Matched"].unique())
    print(visitor_df["Matched"].unique())
    network_map = KeplerGl(height=500, config=config)

    network_map.add_data(data=connections_df, name="Network")

    network_map.add_data(
        data=host_df[
            [
                "LATITUDE",
                "LONGITUDE",
                "Visitor matches (capped)",
                "Visitor matches",
                "Matched",
            ]
        ],
        name="Host homes",
    )

    network_map.add_data(
        data=visitor_df[
            [
                "LATITUDE",
                "LONGITUDE",
                "Host matches (capped)",
                "Host matches",
                "Matched",
            ]
        ],
        name="Visitor homes",
    )

    network_map.save_to_html(
        file_name="maps/Generated_network_map_{}.html".format(settings_string)
    )
