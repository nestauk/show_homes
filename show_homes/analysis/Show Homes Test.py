# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: show_homes
#     language: python
#     name: show_homes
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np

import requests
import json
from time import sleep

import asf_core_data

from asf_core_data.getters.epc import epc_data, data_batches
from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.getters.supplementary_data.geospatial import coordinates
from asf_core_data.pipeline.preprocessing import preprocess_epc_data, data_cleaning
from asf_core_data.utils.visualisation import easy_plotting

from asf_core_data.pipeline.data_joining import merge_install_dates

from asf_core_data.utils.visualisation import easy_plotting, kepler
from keplergl import KeplerGl

from asf_core_data import Path

# %%
# Enter the path to your local data dir
# Adjust data dir!!
LOCAL_DATA_DIR = "/Users/juliasuter/Documents/ASF_data"

# %%
# Takes a while!
# Run only if you have the data 'EPC_GB_preprocessed_and_deduplicated' at hand,
# no need to download it just for testing this line of code!
# You can also simply re-name 'EPC_Wales_preprocessed_and_deduplicated' which was created in the previous cell

EPC_FEAT_SELECTION = [
    "ADDRESS1",
    "ADDRESS2",
    "POSTCODE",
    #  "POSTTOWN",
    "LOCAL_AUTHORITY_LABEL",
    "CURRENT_ENERGY_RATING",
    "POTENTIAL_ENERGY_RATING",
    "TENURE",
    "HP_INSTALLED",
    "INSPECTION_DATE",
    "BUILT_FORM",
    "PROPERTY_TYPE",
    "CONSTRUCTION_AGE_BAND",
    "TRANSACTION_TYPE",
    # "MAIN_FUEL",
    "TOTAL_FLOOR_AREA",
    "HP_TYPE",
    "HP_INSTALLED",
]

epc_df = epc_data.load_preprocessed_epc_data(
    data_path=LOCAL_DATA_DIR, batch="newest", usecols=EPC_FEAT_SELECTION
)
epc_df.shape

# %%
epc_df["HP_INSTALLED"].value_counts(dropna=False)

# %%
manchester = epc_df.loc[epc_df["LOCAL_AUTHORITY_LABEL"] == "Manchester"]
manchester.shape

# %%
manchester.head()

# %%
manchester["HP_INSTALLED"].value_counts()


# %%
def clean_POSTCODE(postcode, level="sector", with_space=True):
    """Get POSTCODE as unit, district, sector or area.

    Args:
        postcode (str):  Raw postcode
        level (str, optional): Desired postcode level.
            Options: district, area, sector, unit. Defaults to "unit".
        with_space (bool, optional): Whether to add space after district. Defaults to True.

    Returns:
        str: Specific postcode level.
    """

    postcode = postcode.strip()
    postcode = postcode.upper()

    seperation = " " if with_space else ""

    if level == "area":
        return re.findall(r"([A-Z]+)", postcode)[0]

    else:
        part_1 = postcode[:-3].strip()
        part_2 = postcode[-3:].strip()

        if level == "district":
            return part_1
        elif level == "sector":
            return part_1 + seperation + part_2[0]
        elif level == "unit":
            return part_1 + seperation + part_2
        else:
            raise IOError(
                "Postcode level '{}' unknown. Please select 'area', 'district', 'sector' or 'unit'.".format(
                    level
                )
            )


coord_df = coordinates.get_postcode_coordinates(data_path=LOCAL_DATA_DIR)

coord_df.head()


print(manchester.shape)
manchester = pd.merge(manchester, coord_df, on=["POSTCODE"], how="left")
print(manchester.shape)

manchester["POSTCODE SECTOR"] = manchester["POSTCODE"].apply(clean_POSTCODE)

# %%
coord_df = coordinates.get_postcode_coordinates(data_path=LOCAL_DATA_DIR)
coord_df.head()

epc_df = pd.merge(epc_df, coord_df, on=["POSTCODE"], how="left")
print(epc_df.shape)

epc_df["POSTCODE SECTOR"] = epc_df["POSTCODE"].apply(clean_POSTCODE)

# %%
epc_df.shape

# %%
from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist  # avoid naming confusion


def get_travel_distance_and_duration(coords_1, coords_2):

    lat_1, lon_1 = coords_1
    lat_2, lon_2 = coords_2

    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{lon_1},{lat_1};{lon_2},{lat_2}?overview=false"
        ""
    )  # then you load the response using the json libray
    routes = json.loads(r.content)
    if routes.get("routes") is None:
        return None
    route_1 = routes.get("routes")[0]

    duration = route_1["duration"] / 60
    distance = route_1["distance"] / 1000

    return duration, distance


# %%
manchester[no_cond].shape

# %%
len(manchester["POSTCODE SECTOR"].unique())

# %%
epc_df = manchester

terraced = [
    "Enclosed Mid-Terrace",
    "Enclosed End-Terrace",
    "End-Terrace",
    "Mid-Terrace",
]
no_cond = ~epc_df["LATITUDE"].isna()
flat = epc_df["PROPERTY_TYPE"] == "Flat"
terraced_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"].isin(terraced)
)
detached_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"] == "Detached"
)
semi_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"] == "Semi-Detached"
)

dist_matrices = {}
dist_matrices_similar = {}

from collections import defaultdict

driving_dist = defaultdict(lambda: defaultdict(dict))
driving_dist_similar = defaultdict(lambda: defaultdict(dict))

# terraced_house: 'Terraced House', detached_house:'Detached House'}

cond_labels = ["All", "Flat", "Semi-detached house", "Detached House", "Terraced House"]


with open("./outputs/distances.csv", "a") as outfile:

    for i, conds in enumerate(
        [no_cond, flat, semi_house, detached_house, terraced_house]
    ):

        label = cond_labels[i]

        epc_df_sampled = epc_df.loc[~epc_df["HP_INSTALLED"]].sample(
            frac=1, random_state=42
        )
        hp_sampled = epc_df.loc[epc_df["HP_INSTALLED"]].sample(frac=1, random_state=42)

        print(epc_df_sampled.shape)
        print(hp_sampled.shape)

        print(label)
        print("-----")

        epc_df_sampled = epc_df_sampled.loc[conds]
        hp_sampled = hp_sampled.loc[~hp_sampled["LATITUDE"].isna()]
        print("# Properties:", epc_df_sampled.shape)

        prop_coords = epc_df_sampled[["LATITUDE", "LONGITUDE"]].to_numpy()
        hp_prop_coords = hp_sampled[["LATITUDE", "LONGITUDE"]].to_numpy()

        dist_matrix = cdist(
            prop_coords, hp_prop_coords, lambda u, v: geodist(u, v).km
        )  # you can choose unit here
        print(dist_matrix.shape)

        hp_prop_coords_similar_type = hp_sampled.loc[conds][
            ["LATITUDE", "LONGITUDE"]
        ].to_numpy()
        print("# similar HP Properties", hp_prop_coords_similar_type.shape)

        dist_matrix_similar = cdist(
            prop_coords, hp_prop_coords_similar_type, lambda u, v: geodist(u, v).km
        )  # you can choose unit here
        print(dist_matrix_similar.shape)

        print("Mean distance", np.min(dist_matrix, axis=1).mean())
        print(
            "Mean distance, similar property",
            np.min(dist_matrix_similar, axis=1).mean(),
        )

        dist_matrices[label] = dist_matrix
        dist_matrices_similar[label] = dist_matrix_similar

        for dist_label, matrix in zip(
            ["any", "similar"], [dist_matrix, dist_matrix_similar]
        ):

            distances = []
            durations = []

            failed = []
            print(dist_label)
            ind = np.argmin(matrix, axis=1)
            print("Indices", ind.shape)

            hp_property_coords = (
                hp_prop_coords_similar_type
                if dist_label == "similar"
                else hp_prop_coords
            )

            print("HP property numbers", hp_property_coords.shape)

            for i, coord_pair in enumerate(zip(prop_coords, hp_property_coords[ind])):

                if label == "All" and dist_label == "similar":
                    continue

                coords_1 = (coord_pair[0][0], coord_pair[0][1])
                coords_2 = (coord_pair[1][0], coord_pair[1][1])

                dist_duration = get_travel_distance_and_duration(coords_1, coords_2)
                if dist_duration is None:
                    failed.append(i)
                    continue
                distance, duration = dist_duration
                distances.append(distance)
                durations.append(duration)

                out = [
                    str(i),
                    label,
                    dist_label,
                    str(coords_1[0]),
                    str(coords_1[1]),
                    str(coords_2[0]),
                    str(coords_2[1]),
                    str(distance),
                    str(duration),
                ]
                out = ",".join(out)

                outfile.write(out)
                outfile.write("\n")

                sleep(1.5)

                if i < 5:
                    print(i)
                    print(distance, duration)

            if dist_label == "any":
                driving_dist[label]["distances"] = np.array(distances)
                driving_dist[label]["durations"] = np.array(durations)
            else:
                driving_dist_similar[label]["distances"] = np.array(distances)
                driving_dist_similar[label]["durations"] = np.array(durations)

            print("Distances:", np.array(distances).mean())
            print("Durations:", np.array(durations).mean())
            print("Failed:", failed)

# %%
epc_df = manchester

terraced = [
    "Enclosed Mid-Terrace",
    "Enclosed End-Terrace",
    "End-Terrace",
    "Mid-Terrace",
]
no_cond = ~epc_df["LATITUDE"].isna()
flat = epc_df["PROPERTY_TYPE"] == "Flat"
terraced_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"].isin(terraced)
)
detached_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"] == "Detached"
)
semi_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"] == "Semi-Detached"
)

dist_matrices = {}
dist_matrices_similar = {}

from collections import defaultdict

driving_dist = defaultdict(lambda: defaultdict(dict))
driving_dist_similar = defaultdict(lambda: defaultdict(dict))

postcode_dict = {}

# terraced_house: 'Terraced House', detached_house:'Detached House'}

cond_labels = ["Flat", "Semi-detached house", "Detached House", "Terraced House", "All"]


with open("./outputs/distances.csv", "a") as outfile:

    for i, conds in enumerate(
        [flat, semi_house, detached_house, terraced_house, no_cond]
    ):

        label = cond_labels[i]

        epc_df_sampled = epc_df.loc[~epc_df["HP_INSTALLED"]].sample(
            frac=1, random_state=42
        )
        hp_sampled = epc_df.loc[epc_df["HP_INSTALLED"]].sample(frac=1, random_state=42)

        print(epc_df_sampled.shape)
        print(hp_sampled.shape)

        print(label)
        print("-----")

        epc_df_sampled = epc_df_sampled.loc[conds]
        hp_sampled = hp_sampled.loc[~hp_sampled["LATITUDE"].isna()]
        epc_df_sampled = epc_df_sampled.loc[~epc_df_sampled["LATITUDE"].isna()]
        print("# Properties:", epc_df_sampled.shape)

        epc_df_sampled = epc_df_sampled.drop_duplicates(
            subset=["LONGITUDE", "LATITUDE"]
        )
        hp_sampled = hp_sampled.drop_duplicates(subset=["LONGITUDE", "LATITUDE"])

        prop_coords = epc_df_sampled[["LATITUDE", "LONGITUDE"]].to_numpy()
        hp_prop_coords = hp_sampled[["LATITUDE", "LONGITUDE"]].to_numpy()

        print(prop_coords.shape)
        print(hp_prop_coords.shape)

        dist_matrix = cdist(
            prop_coords, hp_prop_coords, lambda u, v: geodist(u, v).km
        )  # you can choose unit here
        print(dist_matrix.shape)

        hp_prop_coords_similar_type = hp_sampled.loc[conds][
            ["LATITUDE", "LONGITUDE"]
        ].to_numpy()
        print("# similar HP Properties", hp_prop_coords_similar_type.shape)

        dist_matrix_similar = cdist(
            prop_coords, hp_prop_coords_similar_type, lambda u, v: geodist(u, v).km
        )  # you can choose unit here
        print(dist_matrix_similar.shape)

        print("Mean distance", np.min(dist_matrix, axis=1).mean())
        print(
            "Mean distance, similar property",
            np.min(dist_matrix_similar, axis=1).mean(),
        )

        dist_matrices[label] = dist_matrix
        dist_matrices_similar[label] = dist_matrix_similar

        for dist_label, matrix in zip(
            ["any", "similar"], [dist_matrix, dist_matrix_similar]
        ):

            distances = []
            durations = []

            failed = []
            print(dist_label)
            ind = np.argmin(matrix, axis=1)
            print("Indices", ind.shape)

            hp_property_coords = (
                hp_prop_coords_similar_type
                if dist_label == "similar"
                else hp_prop_coords
            )

            print("HP property numbers", hp_property_coords.shape)

            print(prop_coords.shape)
            print(hp_property_coords[ind].shape)

            for i, coord_pair in enumerate(zip(prop_coords, hp_property_coords[ind])):

                if label == "All" and dist_label == "similar":
                    continue

                coords_1 = (coord_pair[0][0], coord_pair[0][1])
                coords_2 = (coord_pair[1][0], coord_pair[1][1])

                print(coords_1)

                print(matrix.shape)
                print(prop_coords.shape)
                print(ind.shape)
                print(np.min(matrix, axis=1)[i])

                distance = np.min(matrix, axis=1)[i]

                if distance is None:
                    failed.append(i)
                    continue

                postcode_dict[label, dist_label, coords_1] = (coords_2, distance)

                out = [
                    str(i),
                    label,
                    dist_label,
                    str(coords_1[0]),
                    str(coords_1[1]),
                    str(coords_2[0]),
                    str(coords_2[1]),
                    str(distance),
                    str("unknown"),
                ]
                out = ",".join(out)

                outfile.write(out)
                outfile.write("\n")


# %%
dist_df = pd.read_csv("./outputs/distances.csv")
dist_df.head()

# %%
dist_df = dist_df.loc[
    (dist_df["HP Property"] == "similar") & (dist_df["Property Type"] != "All")
]
dist_df.shape

# %%
dist_df["LAT_LONG"] = dist_df[["Latitude 1", "Longitude 1"]].apply(tuple, axis=1)
dist_df["NEAREST_HP"] = dist_df[
    ["Latitude 2", "Longitude 2", "Distance", "Property Type"]
].apply(tuple, axis=1)

dist_df["NEAREST_HP"]

# %%
coord_to_nearest_hp = dict(zip(dist_df["LAT_LONG"], dist_df["NEAREST_HP"]))

# %%
manchester["LAT_LONG"] = manchester[["LATITUDE", "LONGITUDE"]].apply(tuple, axis=1)

manchester["NEAREST_HP"] = manchester["LAT_LONG"].map(coord_to_nearest_hp)

# %%
manchester["NEAREST_HP"]
manchester[["HP_LATITUDE", "HP_LONGITUDE", "DIST", "PROP_TYPE"]] = pd.DataFrame(
    manchester["NEAREST_HP"].tolist(), index=manchester.index
)

# %%
manchester["BUILT_FORM"].value_counts()

# %%
from ipywidgets import interact


@interact(
    max_dist=(0, 10),
    prop_type=["Flat", "Semi-detached house", "Detached House", "Terraced House"],
)
def within_reach(max_dist, prop_type):

    subset = manchester.loc[manchester["PROP_TYPE"] == prop_type]

    subset["HP_WITHIN_REACH"] = np.where(subset.DIST < max_dist, True, False)

    print(subset["HP_WITHIN_REACH"].value_counts(dropna=False, normalize=True))

    import yaml

    config_file = "distances.txt"
    with open(config_file, "r") as infile:
        config = infile.read()
        config = yaml.load(config, Loader=yaml.FullLoader)

    manchester_map = KeplerGl(height=500, config=config)

    manchester_map.add_data(
        data=subset[["LATITUDE", "LONGITUDE", "HP_LATITUDE", "HP_LONGITUDE", "DIST"]],
        name="distances",
    )

    manchester_map.add_data(
        data=subset[["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST"]],
        name="within reach",
    )

    manchester_map


# %%
manchester["HP_WITHIN_REACH"] = manchester.DIST < 3

# %%
manchester["HP_WITHIN_REACH"].value_counts(dropna=False)

manchester["HP_WITHIN_REACH"] = manchester["HP_WITHIN_REACH"].astype(str)

# %%
manchester.shape

# %%
manchester["HP_WITHIN_REACH"].value_counts()

manchester_red = manchester.loc[manchester["HP_WITHIN_REACH"] == "False"]
manchester_red.shape

# %%
## from keplergl import KeplerGl
import yaml

config_file = "distances.txt"
with open(config_file, "r") as infile:
    config = infile.read()
    config = yaml.load(config, Loader=yaml.FullLoader)

manchester_map = KeplerGl(height=500, config=config)

manchester_map.add_data(
    data=manchester[["LATITUDE", "LONGITUDE", "HP_LATITUDE", "HP_LONGITUDE", "DIST"]],
    name="distances",
)

manchester_map.add_data(
    data=manchester[["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST"]],
    name="within reach",
)


manchester_map.add_data(
    data=manchester_red[["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST"]],
    name="radius",
)


manchester_map.add_data(
    data=manchester.loc[manchester["PROPERTY_TYPE"] == "Flat"][
        [
            "LATITUDE",
            "LONGITUDE",
            "HP_WITHIN_REACH",
            "DIST",
            "HP_LATITUDE",
            "HP_LONGITUDE",
        ]
    ],
    name="Flat",
)

# manchester_map.add_data(
#     data=manchester.loc[manchester['PROPERTY_TYPE'] == 'Detached House'][["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST", "HP_LATITUDE", "HP_LONGITUDE",]], name="Detached House")


# manchester_map.add_data(
#     data=manchester.loc[manchester['PROPERTY_TYPE'] == 'Semi-detached House'][["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST", "HP_LATITUDE", "HP_LONGITUDE",]], name="Semi-detached House")

# manchester_map.add_data(
#     data=manchester.loc[manchester['PROPERTY_TYPE'] == 'Terraced House'][["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST", "HP_LATITUDE", "HP_LONGITUDE",]], name="Terraced House")


# manchester_map.add_data(
#     data=manchester_red.loc[manchester_red['PROPERTY_TYPE'] == 'Flat'][["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST", "HP_LATITUDE", "HP_LONGITUDE",]], name="Flat (reach)")

# manchester_map.add_data(
#     data=manchester_red.loc[manchester_red['PROPERTY_TYPE'] == 'Detached House'][["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST", "HP_LATITUDE", "HP_LONGITUDE",]], name="Detached House (reach)")


# manchester_map.add_data(
#     data=manchester_red.loc[manchester_red['PROPERTY_TYPE'] == 'Semi-detached House'][["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST", "HP_LATITUDE", "HP_LONGITUDE",]], name="Semi-detached House (reach) ")

# manchester_map.add_data(
#     data=manchester_red.loc[manchester_red['PROPERTY_TYPE'] == 'Terraced House'][["LATITUDE", "LONGITUDE", "HP_WITHIN_REACH", "DIST", "HP_LATITUDE", "HP_LONGITUDE",]], name="Terraced House (reach)")


manchester_map

# %%
with open("distances.txt", "w") as outfile:
    outfile.write(str(manchester_map.config))

manchester_map.save_to_html(file_name="Distances.html")

# %%
manchester.to_csv("manchester.csv")

# %%
import geopandas as gpd
from shapely.geometry import Polygon

manchester_red = manchester_red[manchester_red["LATITUDE"].notna()]
manchester_red = manchester_red[manchester_red["LONGITUDE"].notna()]


lat_point_list = manchester_red["LATITUDE"]
lon_point_list = manchester_red["LONGITUDE"]

polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
crs = {"init": "epsg:4326"}
polygon = gpd.GeoDataFrame(index=[0], crs=crs, geometry=[polygon_geom])

polygon.to_file(filename="polygon.geojson", driver="GeoJSON")
polygon.to_file(filename="polygon.shp", driver="ESRI Shapefile")

# %%
polygon.shape

# %%
## from keplergl import KeplerGl
import yaml

# config_file = 'distances.txt'
# with open(config_file, "r") as infile:
#     config = infile.read()
#     config = yaml.load(config, Loader=yaml.FullLoader)

hp_map = KeplerGl(height=500)  # , config=config)

hp_map.add_data(data=polygon[["geometry"]], name="geo")

hp_map

# %%
import numpy as np
from scipy.spatial import KDTree

pts = np.array(
    [
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
        (1, 3),
        (2, 3),
        (3, 3),
        (4, 3),
        (1, 4),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
)

# Build tree
tree = KDTree(pts)

# Query tree for points within distance=2 of (1,1)
idx = tree.query_ball_point([1, 1], r=2)
print(idx)

# %%
# Get coordinates of the indices
print(tree.data[idx])

# %%
from scipy import spatial

import numpy as np

A = np.random.random((10, 2)) * 100

A

# %%
A = np.array(((51.511312, -0.107522), (51.519544, -0.128649), (51.532503, -0.124413)))

# %%
A

# %%
pt = [6, 30]  # <-- the point to find

A[spatial.KDTree(A).query(pt)[1]]  # <-- the nearest point

# %%
distance, index = spatial.KDTree(A).query(pt)

# %%
distance

# %%
index

# %%
A[index]

# %%
points = [[6, 10], [30, 30]]

points = np.array(
    ((51.533538, -0.125389), (51.511173, -0.104874), (51.729344, -0.119406))
)

distance, index = spatial.KDTree(A).query(points)

# %%
distance

# %%
index

# %%
import numpy as np


def deg2rad(degree):
    rad = degree * 2 * np.pi / 360
    return rad


def distToKM(x):
    R = 6367  # earth radius
    gamma = 2 * np.arcsin(
        deg2rad(x / (2 * R))
    )  # compute the angle of the isosceles triangle
    dist = 2 * R * np.sin(gamma / 2)  # compute the side of the triangle
    return dist


distToKM(1501.59637685)
# 26.207800812050056


# %%
distToKM(distance[0])

# %%
distToKM(distance[1])

# %%
distToKM(distance[2])

# %%
np.deg2rad(distance[2])

# %%
deg2rad(distance[2])

# %%
epc_df.columns

# %%
coords = manchester[["LATITUDE", "LONGITUDE"]].to_numpy()
coords = epc_df[["LATITUDE", "LONGITUDE"]].to_numpy()

# %%
hp_coords = coords[:5]
hp_coords

# %%
no_hp_coords = coords[500:510]
no_hp_coords

# %%
from scipy import spatial

# %%
epc_df[["LATITUDE", "LONGITUDE"]][10:15].values

# %%
coords[500:550]

# %% [markdown]
# ## Get nearest HP home by coords

# %%
epc_df.shape

# %%
hp_coords = (
    epc_df[["LATITUDE", "LONGITUDE"]]
    .sample(frac=1)
    .reset_index(drop=True)[10:15]
    .values
)
non_hp_coords = (
    epc_df[["LATITUDE", "LONGITUDE"]]
    .sample(frac=1)
    .reset_index(drop=True)
    .to_numpy()[500:510]
)

original_hp_coords = hp_coords.copy()
original_non_hp_coords = non_hp_coords.copy()

hp_coords = np.deg2rad(hp_coords)
non_hp_coords = np.deg2rad(non_hp_coords)


def to_Cartesian(lat, lng):
    R = 6367  # radius of the Earth in kilometers

    x = R * np.cos(lat) * np.cos(lng)
    y = R * np.cos(lat) * np.sin(lng)
    z = R * np.sin(lat)
    return np.array((x, y, z)).T


# %%
hp_coords = to_Cartesian(hp_coords[:, 0], hp_coords[:, 1])
non_hp_coords = to_Cartesian(non_hp_coords[:, 0], non_hp_coords[:, 1])

hp_coords.shape

# %%
tree = spatial.KDTree(hp_coords)

distances, indices = tree.query(non_hp_coords)
distances

# %%
original_non_hp_coords, original_hp_coords[indices]

# %%
# epc_df = manchester

terraced = [
    "Enclosed Mid-Terrace",
    "Enclosed End-Terrace",
    "End-Terrace",
    "Mid-Terrace",
]
no_cond = ~epc_df["LATITUDE"].isna()
flat = epc_df["PROPERTY_TYPE"] == "Flat"
terraced_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"].isin(terraced)
)
detached_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"] == "Detached"
)
semi_house = (epc_df["PROPERTY_TYPE"] == "House") & (
    epc_df["BUILT_FORM"] == "Semi-Detached"
)

dist_matrices = {}
dist_matrices_similar = {}

from collections import defaultdict

driving_dist = defaultdict(lambda: defaultdict(dict))
driving_dist_similar = defaultdict(lambda: defaultdict(dict))

postcode_dict = {}

# terraced_house: 'Terraced House', detached_house:'Detached House'}

cond_labels = ["Flat", "Semi-detached house", "Detached House", "Terraced House", "All"]


with open("./outputs/distances.csv", "a") as outfile:

    for i, conds in enumerate(
        [flat, semi_house, detached_house, terraced_house, no_cond]
    ):

        label = cond_labels[i]

        epc_df_sampled = epc_df.loc[~epc_df["HP_INSTALLED"]].sample(
            frac=1, random_state=42
        )
        hp_sampled = epc_df.loc[epc_df["HP_INSTALLED"]].sample(frac=1, random_state=42)

        print(epc_df_sampled.shape)
        print(hp_sampled.shape)

        print(label)
        print("-----")

        epc_df_sampled = epc_df_sampled.loc[conds]
        hp_sampled = hp_sampled.loc[~hp_sampled["LATITUDE"].isna()]
        epc_df_sampled = epc_df_sampled.loc[~epc_df_sampled["LATITUDE"].isna()]
        print("# Properties:", epc_df_sampled.shape)

        epc_df_sampled = epc_df_sampled.drop_duplicates(
            subset=["LONGITUDE", "LATITUDE"]
        )
        hp_sampled = hp_sampled.drop_duplicates(subset=["LONGITUDE", "LATITUDE"])

        prop_coords = epc_df_sampled[["LATITUDE", "LONGITUDE"]].to_numpy()
        hp_prop_coords = hp_sampled[["LATITUDE", "LONGITUDE"]].to_numpy()

        print(prop_coords.shape)
        print(hp_prop_coords.shape)

        dist_matrix = cdist(
            prop_coords, hp_prop_coords, lambda u, v: geodist(u, v).km
        )  # you can choose unit here
        print(dist_matrix.shape)

        hp_prop_coords_similar_type = hp_sampled.loc[conds][
            ["LATITUDE", "LONGITUDE"]
        ].to_numpy()
        print("# similar HP Properties", hp_prop_coords_similar_type.shape)

        dist_matrix_similar = cdist(
            prop_coords, hp_prop_coords_similar_type, lambda u, v: geodist(u, v).km
        )  # you can choose unit here
        print(dist_matrix_similar.shape)

        print("Mean distance", np.min(dist_matrix, axis=1).mean())
        print(
            "Mean distance, similar property",
            np.min(dist_matrix_similar, axis=1).mean(),
        )

        dist_matrices[label] = dist_matrix
        dist_matrices_similar[label] = dist_matrix_similar

        for dist_label, matrix in zip(
            ["any", "similar"], [dist_matrix, dist_matrix_similar]
        ):

            distances = []
            durations = []

            failed = []
            print(dist_label)
            ind = np.argmin(matrix, axis=1)
            print("Indices", ind.shape)

            hp_property_coords = (
                hp_prop_coords_similar_type
                if dist_label == "similar"
                else hp_prop_coords
            )

            print("HP property numbers", hp_property_coords.shape)

            print(prop_coords.shape)
            print(hp_property_coords[ind].shape)

            for i, coord_pair in enumerate(zip(prop_coords, hp_property_coords[ind])):

                if label == "All" and dist_label == "similar":
                    continue

                coords_1 = (coord_pair[0][0], coord_pair[0][1])
                coords_2 = (coord_pair[1][0], coord_pair[1][1])

                print(coords_1)

                print(matrix.shape)
                print(prop_coords.shape)
                print(ind.shape)
                print(np.min(matrix, axis=1)[i])

                distance = np.min(matrix, axis=1)[i]

                if distance is None:
                    failed.append(i)
                    continue

                postcode_dict[label, dist_label, coords_1] = (coords_2, distance)

                out = [
                    str(i),
                    label,
                    dist_label,
                    str(coords_1[0]),
                    str(coords_1[1]),
                    str(coords_2[0]),
                    str(coords_2[1]),
                    str(distance),
                    str("unknown"),
                ]
                out = ",".join(out)

                outfile.write(out)
                outfile.write("\n")


# %%
# terraced = ["Enclosed Mid-Terrace", "Enclosed End-Terrace", "End-Terrace", "Mid-Terrace"]
# no_cond = ~epc_df['LATITUDE'].isna()
# flat = epc_df['PROPERTY_TYPE'] == 'Flat'
# terraced_house = ((epc_df['PROPERTY_TYPE'] == 'House') & (epc_df['BUILT_FORM'].isin(terraced)))
# detached_house = ((epc_df['PROPERTY_TYPE'] == 'House') & (epc_df['BUILT_FORM'] == 'Detached'))
# semi_house = ((epc_df['PROPERTY_TYPE'] == 'House') & (epc_df['BUILT_FORM'] == 'Semi-Detached'))

# dist_matrices = {}
# dist_matrices_similar = {}

# from collections import defaultdict
# from scipy import spatial

# driving_dist = defaultdict(lambda: defaultdict(dict))
# driving_dist_similar = defaultdict(lambda: defaultdict(dict))

# cond_labels = ['Flat', 'Semi-detached house', 'Detached House', 'Terraced House']


# with open('./outputs/distances.csv', 'a') as outfile:


#     for i, conds in enumerate([flat, semi_house, detached_house, terraced_house]):

#         label = cond_labels[i]

#         epc_df = epc_df[epc_df['LATITUDE'].notna()]
#         epc_df = epc_df[epc_df['LONGITUDE'].notna()]

#         epc_df_sampled = epc_df.loc[~epc_df['HP_INSTALLED']].sample(frac=1, random_state=42)
#         hp_sampled = epc_df.loc[epc_df['HP_INSTALLED']].sample(frac=1, random_state=42)

#         print(epc_df_sampled.shape)
#         print(hp_sampled.shape)

#         print(label)
#         print('-----')

#         epc_df_sampled = epc_df_sampled.loc[conds]
#         hp_sampled = hp_sampled.loc[~hp_sampled['LATITUDE'].isna()]
#         print('# Properties:', epc_df_sampled.shape)


#         prop_coords_org = epc_df_sampled[['LATITUDE', 'LONGITUDE']].to_numpy()
#         hp_prop_coords_org = hp_sampled[['LATITUDE', 'LONGITUDE']].to_numpy()
#         hp_prop_coords_similar_type_org = hp_sampled.loc[conds][['LATITUDE', 'LONGITUDE']].to_numpy()

#         prop_coords = np.deg2rad(prop_coords_org.copy())
#         hp_prop_coords = np.deg2rad(hp_prop_coords_org.copy())
#         hp_prop_coords_similar_type = np.deg2rad(hp_prop_coords_similar_type_org.copy())

#         prop_coords = to_Cartesian(prop_coords[:,0],prop_coords[:,1])
#         hp_prop_coords = to_Cartesian(hp_prop_coords[:,0],hp_prop_coords[:,1])
#         hp_prop_coords_similar_type = to_Cartesian(hp_prop_coords_similar_type[:,0],hp_prop_coords_similar_type[:,1])

#         hp_prop_dict = {'any': hp_prop_coords, 'similar': hp_prop_coords_similar_type}
#         coords_dict = {'any': hp_prop_coords_org, 'similar': hp_prop_coords_similar_type_org}

#         for version in ['any', 'similar']:

#             failed = []

#             hp_coords = hp_prop_dict[version]
#             hp_coords_org = coords_dict[version]

#             print('now computing tree...')
#             tree = spatial.KDTree(hp_coords)
#             distances, indices = tree.query(prop_coords)

#             #indices = np.where(indices==hp_coords.shape[0], None, indices)

#             print('Done')

#             for i, coord_pair in enumerate(zip(prop_coords_org, hp_coords_org[indices])):

#                 if label == 'All' and version == 'similar':
#                     continue

#                 coords_1 = (coord_pair[0][0],coord_pair[0][1])
#                 coords_2 = (coord_pair[1][0],coord_pair[1][1])


#                 dist_duration = get_travel_distance_and_duration(coords_1, coords_2)


#                 if dist_duration is None:
#                     failed.append(i)
#                     continue

#                 distance, duration = dist_duration


#                 out = [str(i), label,version,str(coords_1[0]),str(coords_1[1]),str(coords_2[0]),str(coords_2[1]),str(distance),str(duration)]
#                 out = ','.join(out)

#                 outfile.write(out)
#                 outfile.write('\n')

#                 sleep(1.5)

#                 if i < 5:
#                     print(i)
#                     print(distance, duration)


#             print('Failed:', failed)
