import numpy as np
import requests
import json

from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist  # avoid naming confusion


def get_travel_distance_and_duration(coords_1, coords_2):
    """Retrieve the travel distance and travel time for two coordinates.

    Args:
        coords_1 (tuple): Coordinates for source location (lat, long).
        coords_2 (tuple): Coordinates for target location (lat, long).

    Returns:
        duration (float): Travel duration in hours.
        distance (float): Travel distance in km.
    """

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
    """Convert latitude/longitude coordinates to Cartesian coordinate system.

    Args:
        lat (float): Latitude.
        lng (float): Longitude.

    Returns:
        np.array: Converted cartesian coordinates (x,y,z).
    """
    R = 6367  # radius of the Earth in kilometers

    x = R * np.cos(lat) * np.cos(lng)
    y = R * np.cos(lat) * np.sin(lng)
    z = R * np.sin(lat)

    return np.array((x, y, z)).T


def prepare_coords(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """

    coords_org = df[["LATITUDE", "LONGITUDE"]].to_numpy()
    coords = coords_org.copy()

    # Convert to Cartesian coordinates
    coords = np.deg2rad(coords)
    coords_cart = to_Cartesian(coords[:, 0], coords[:, 1])

    return coords_cart, coords_org
