# File: show_homes/utils/geo_utils.py
"""
Utils for handling geographical data.

Project: Show homes
Author: Julia Suter
"""

import numpy as np
import requests
import json

from show_homes.config import config


def get_travel_distance_and_duration(coords_1, coords_2):
    """Retrieve the travel distance and travel time for two coordinates.

    Args:
        coords_1 (tuple): Coordinates for source location (lat, long).
        coords_2 (tuple): Coordinates for target location (lat, long).

    Returns:
        duration (float): Travel duration in hours.
        distance (float): Travel distance in km.
    """

    lat_1, lng_1 = coords_1
    lat_2, lng_2 = coords_2

    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{lng_1},{lat_1};{lng_2},{lat_2}?overview=false"
        ""
    )

    routes = json.loads(r.content)
    if routes.get("routes") is None:
        return None
    route_1 = routes.get("routes")[0]

    duration = route_1["duration"] / 60
    distance = route_1["distance"] / 1000

    return duration, distance


def to_Cartesian(coords):
    """Convert latitude/longitude coordinates to Cartesian coordinate system.

    Args:
        coords (np.array): Latitude, longitude as array of shape (n, 2).

    Returns:
        np.array: Converted cartesian coordinates (x,y,z).
    """

    coords = np.deg2rad(coords)
    lat, lng = coords[:, 0], coords[:, 1]

    R = config.EARTH_RADIUS

    x = R * np.cos(lat) * np.cos(lng)
    y = R * np.cos(lat) * np.sin(lng)
    z = R * np.sin(lat)

    return np.array((x, y, z)).T


def distance(coords_1, coords_2):
    """Compute distance in km for sets of pairs of coordinates (lat/lng).
    Spherical model so resulting in an error of up to about 0.5%
    but not relevant for shorter distances.

    Args:
        coords_1 (np.array): First set of coordinates in lat/lng format.
        coords_2 (np.array): Second set of coordinates in lat/lng format.

    Returns:
        float: Distance in km.
    """

    rad_coords_1 = np.radians(coords_1)
    rad_coords_2 = np.radians(coords_2)

    # Haversine formula
    dlng = rad_coords_2[:, 1] - rad_coords_1[:, 1]
    dlat = rad_coords_2[:, 0] - rad_coords_1[:, 0]

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(rad_coords_1[:, 0])
        * np.cos(rad_coords_2[:, 0])
        * np.sin(dlng / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return c * config.EARTH_RADIUS


# Alternative for computing distance with geodist and cdist (just saving if ever needed at a later stage)

# from scipy.spatial.distance import cdist
# from geopy.distance import distance as geodist # avoid naming confusion
# connects[:,4] = cdist(connects[:,:2], connects[:,2:4], lambda u, v: geodist(u, v).km)[np.eye(connects.shape[0], dtype=bool)].round(1)
