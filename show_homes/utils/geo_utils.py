# File: show_homes/utils/geo_utils
"""
Utils for handling geographical data.

Project: Show homes
Author: Julia Suter
"""

import numpy as np
import requests
import json


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


def prepare_coords(coords):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """

    coords_org = coords.copy()

    # Convert to Cartesian coordinates
    coords = np.deg2rad(coords)
    coords_cart = to_Cartesian(coords[:, 0], coords[:, 1])

    return coords_cart, coords_org


def distance(coords_1, coords_2):
    """Lat/long input

    Args:
        coords_1 (_type_): _description_
        coords_2 (_type_): _description_
    """

    rad_coords_1 = np.radians(coords_1)
    rad_coords_2 = np.radians(coords_2)

    # Haversine formula
    dlon = rad_coords_2[:, 1] - rad_coords_1[:, 1]
    dlat = rad_coords_2[:, 0] - rad_coords_1[:, 0]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(rad_coords_1[:, 0])
        * np.cos(rad_coords_2[:, 0])
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.009

    return c * r

    # from scipy.spatial.distance import cdist
    # from geopy.distance import distance as geodist # avoid naming confusion
    # connections[:,4] = cdist(connections[:,:2], connections[:,2:4], lambda u, v: geodist(u, v).km)[np.eye(connections.shape[0], dtype=bool)].round(1) # you can choose unit here
