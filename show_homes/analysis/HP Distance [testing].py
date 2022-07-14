#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")

from asf_core_data.getters.epc import epc_data
from asf_core_data.pipeline.preprocessing import (
    preprocess_epc_data,
    feature_engineering,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from geopy import distance
from asf_core_data import PROJECT_DIR

from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist  # avoid naming confusion

from scipy import spatial

import matplotlib.pyplot as plt

from ipywidgets import interact


# In[316]:


EPC_FEAT_SELECTION = [
    "ADDRESS1",
    "ADDRESS2",
    "POSTCODE",
    "POSTTOWN",
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
    "MAIN_FUEL",
    "TOTAL_FLOOR_AREA",
    "HP_TYPE",
]


# In[317]:


epc_df = epc_data.load_preprocessed_epc_data(
    version="preprocessed_dedupl", usecols=EPC_FEAT_SELECTION
)


# In[318]:


epc_df.head()


# In[319]:


epc_df.loc[(epc_df["POSTCODE"].str.startswith("SW84") & epc_df["HP_INSTALLED"])].shape


# In[320]:


import pandas

pandas.set_option("display.max_rows", None)


# In[323]:


epc_df.loc[(epc_df["POSTCODE"].str.startswith("SW84") & epc_df["HP_INSTALLED"])][
    ["ADDRESS1", "ADDRESS2", "POSTCODE", "HP_TYPE", "CONSTRUCTION_AGE_BAND"]
].head(200)


# In[34]:


epc_df["BUILT_FORM"].value_counts()


# In[5]:


epc_df = feature_engineering.get_postcode_coordinates(epc_df)
epc_df["COORDS"] = list(zip(epc_df["LATITUDE"], epc_df["LONGITUDE"]))


# In[164]:


epc_df["COORDS"]


# In[273]:


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


# In[277]:


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

cond_labels = [
    "Semi-detached house",
    "Flat",
    "Terraced House",
    "Detached House",
    "All",
]

dist_matrices = {}
dist_matrices_similar = {}

from collections import defaultdict

driving_dist = defaultdict(lambda: defaultdict(dict))
driving_dist_similar = defaultdict(lambda: defaultdict(dict))

# label_dict = {semi_house:'Semi-detached House', no_cond:'All', flat:'Flat',
#          terraced_house: 'Terraced House', detached_house:'Detached House'}

cond_labels = ["Flat", "Semi-detached house", "Terraced House", "Detached House", "All"]


for i, conds in enumerate([flat, semi_house, terraced_house, detached_house, no_cond]):

    label = cond_labels[i]

    epc_df_sampled = epc_df.loc[~epc_df["HP_INSTALLED"]].sample(
        frac=1, random_state=42
    )[:10000]
    hp_sampled = epc_df.loc[epc_df["HP_INSTALLED"]].sample(frac=1, random_state=42)[
        :1000
    ]

    print(label)
    print("-----")

    epc_df_sampled = epc_df_sampled.loc[conds]
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
    print("Mean distance, similar property", np.min(dist_matrix_similar, axis=1).mean())

    dist_matrices[label] = dist_matrix
    dist_matrices_similar[label] = dist_matrix_similar

    np.save(label + "_any.npy", dist_matrix)
    np.save(label + "_similar.npy", dist_matrix_similar)


# In[276]:


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

cond_labels = [
    "Semi-detached house",
    "Flat",
    "Terraced House",
    "Detached House",
    "All",
]

dist_matrices = {}
dist_matrices_similar = {}

from collections import defaultdict

driving_dist = defaultdict(lambda: defaultdict(dict))
driving_dist_similar = defaultdict(lambda: defaultdict(dict))

# label_dict = {semi_house:'Semi-detached House', no_cond:'All', flat:'Flat',
#          terraced_house: 'Terraced House', detached_house:'Detached House'}

cond_labels = ["Terraced House", "Detached House", "All"]  #'Semi-detached house',


with open("distances.csv", "a") as outfile:

    for i, conds in enumerate([terraced_house, detached_house, no_cond]):  # semi_house,

        label = cond_labels[i]

        epc_df_sampled = epc_df.loc[~epc_df["HP_INSTALLED"]].sample(
            frac=1, random_state=42
        )[:10000]
        hp_sampled = epc_df.loc[epc_df["HP_INSTALLED"]].sample(frac=1, random_state=42)[
            :1000
        ]

        print(label)
        print("-----")

        epc_df_sampled = epc_df_sampled.loc[conds]
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

                if dist_label == "any" and label == "Terraced House" and i <= 2188:
                    continue

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


# In[188]:


sample_dist_matrix = dist_matrices_same[0]


# In[204]:


print(np.min(sample_dist_matrix, axis=1).shape)

ind = np.argmin(sample_dist_matrix, axis=1)

no_hp_coords = prop_coords
hp_coords = hp_prop_coords[ind]


# In[208]:


len(np.unique(ind))


# In[210]:


ind


# In[215]:


coords_combos = zip(prop_coords, hp_prop_coords[ind])

for coord_pair in coords_combos:

    print(coord_pair[0], coord_pair[1])


# In[240]:


def get_travel_distance_and_duration(coords_1, coords_2):

    lat_1, lon_1 = coords_1
    lat_2, lon_2 = coords_2

    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{lon_1},{lat_1};{lon_2},{lat_2}?overview=false"
        ""
    )  # then you load the response using the json libray
    routes = json.loads(r.content)
    route_1 = routes.get("routes")[0]

    duration = route_1["duration"] / 60
    distance = route_1["distance"] / 1000

    return duration, distance


from time import sleep

distances = []
durations = []

for coord_pair in list(zip(prop_coords, hp_prop_coords[ind]))[:2]:

    coords_1 = (coord_pair[0][0], coord_pair[0][1])
    coords_2 = (coord_pair[1][0], coord_pair[1][1])

    print(coords_1, coords_2)
    distance, duration = get_travel_distance_and_duration(coords_1, coords_2)
    distances.append(distance)
    durations.append(duration)

    sleep(2)

print(distances)
print(durations)


# In[148]:


distances = pd.DataFrame(dist_matrices).T
distances.columns = cond_labels
distances.head()
distances.plot(kind="density")  # or pd.Series()


# In[282]:


distances


# In[280]:


cond_labels = [
    "All",
    "Flat",
    "Semi-detached House",
    "Detached House",
    "Terraced House",
]


for label in cond_labels:
    sns.kdeplot(data=distances[label], fill=True, common_norm=False, alpha=0.1)
plt.legend(cond_labels)
plt.xlabel("Distance [km]")
plt.title("Air distance to nearest HP property")
plt.xlim(-5, 50)
plt.savefig("Air distance to nearest HP property.png")
plt.show()


# In[281]:


cond_labels = [
    "All",
    "Flat",
    "Detached House",
    "Semi-detached house",
    "Terraced House",
]


for label in cond_labels:
    sns.kdeplot(data=same_distances[label], fill=True, common_norm=False, alpha=0.1)
plt.legend(cond_labels)
plt.xlabel("Distance [km]")
plt.title("Air distance to nearest similar HP property")
plt.xlim(-5, 50)
plt.savefig("Air distance to nearest similar HP property.png")

plt.show()


# In[263]:


# 51.31143039897616, 0.03468077124949037
# 51.99290880326267, 0.39142219451868565

# 50.957578196663334, -3.7449509358869957
# 51.7500054896761, -3.9756638249275205

# 56.23502558061421, -4.059261153368452
# 55.909265295815224, -4.261134931278913

import requests
import json  # call the OSMR API

r = requests.get(
    f"http://router.project-osrm.org/route/v1/car/{-4.059261153368452},{56.23502558061421};{-4.059261153368452},{56.23502558061421}?overview=false"
    ""
)  # then you load the response using the json libray
# by default you get only one alternative so you access 0-th element of the `routes`routes = json.loads(r.content)
routes = json.loads(r.content)
route_1 = routes.get("routes")[0]


# In[264]:


route_1["duration"]


# In[232]:


def get_travel_distance_and_duration(coords_1, coords_2):

    lat_1, lon_1 = coords_1
    lat_2, lon_2 = coords_2

    r = requests.get(
        f"http://router.project-osrm.org/route/v1/car/{lon_1},{lat_1};{lon_2},{lat_2}?overview=false"
        ""
    )  # then you load the response using the json libray
    routes = json.loads(r.content)
    route_1 = routes.get("routes")[0]

    duration = route_1["duration"] / 60
    distance = route_1["distance"] / 1000

    return duration, distance


# In[233]:


coords_1 = (56.23502558061421, -4.059261153368452)
coords_2 = (55.909265295815224, -4.261134931278913)


duration, distance = get_travel_distance_and_duration(coords_1, coords_2)


# In[234]:


duration, distance


# In[ ]:


prop_coords = epc_df_sampled[["LATITUDE", "LONGITUDE"]].to_numpy()
hp_prop_coords = hp_sampled[["LATITUDE", "LONGITUDE"]].to_numpy()

dist_matrix = cdist(
    prop_coords, hp_prop_coords, lambda u, v: geodist(u, v).km
)  # you can choose unit here
print(dist_matrix.shape)


# In[71]:


np.min(dist_matrix, axis=1).mean()


# In[65]:


hp_prop_coords_same_type = hp_sampled.loc[conds][["LATITUDE", "LONGITUDE"]].to_numpy()
print(hp_prop_coords_same_type.shape)

dist_matrix_same = cdist(
    prop_coords, hp_prop_coords_same_type, lambda u, v: geodist(u, v).km
)  # you can choose unit here
print(dist_matrix_same.shape)


# In[72]:


np.min(dist_matrix_same, axis=1).mean()


# In[48]:


semi_detached_house_same = dist_matrix_same.copy()


# In[14]:


np.save(
    "distribution_matrix.npy",
    dist_matrix,
)


# In[ ]:


# flat_same = dist_matrix_same.copy()
# detached_house_same = dist_matrix_same.copy()
# semi_detached_house_same = dist_matrix_same.copy()


# In[44]:


plt.hist(
    [np.min(detached_house_same, axis=1), np.min(flat_same, axis=1)],
    bins=100,
    histtype="step",
)


# In[ ]:


mask = np.argmin(dist_matrix, axis=1)

for i, j in zip(range(dist_matrix.shape[0]), mask):

    print(epc_df_sampled.iloc[i]["POSTCODE"])
    print(hp_sampled.iloc[j]["POSTCODE"])
    print()


# In[ ]:


np.min(dist_matrix, axis=1)


# In[ ]:


epc_df_sampled["POSTCODE"][100:200]


# In[ ]:


hp_sampled["POSTCODE"][100:200]


# ----
#
# ####  Random Things I Tried
#
# ---

# In[ ]:


import pgeocode

dist = pgeocode.GeoDistance("GB")
nomi = pgeocode.Nominatim("GB")
dist.query_postal_code("OX28", "WN2")


# In[ ]:


nomi.query_postal_code("OX28")["longitude"]


# In[ ]:


from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist  # avoid naming confusion

sc_distx = cdist(
    all_coords[:3], all_HP_coords[:3], lambda u, v: geodist(u, v).km
)  # you can choose unit here


# In[ ]:


postcode_samples = "NW9"
nomi.query_postal_code(postcode_samples)["latitude"], nomi.query_postal_code(
    postcode_samples
)["longitude"]

postcode_samples = "NW9 blablabla"
nomi.query_postal_code(postcode_samples)["latitude"], nomi.query_postal_code(
    postcode_samples
)["longitude"]


# In[ ]:


postcodes_longs = [nomi.query_postal_code(pc)["longitude"] for pc in postcodes]
postcodes_lats = [nomi.query_postal_code(pc)["latitude"] for pc in postcodes]
all_coords = np.array(list(zip(postcodes_lats, postcodes_longs)))
print(all_coords.shape)
all_coords = all_coords[~np.isnan(all_coords).any(axis=1)]
print(all_coords.shape)


# In[ ]:


postcodes_longs = [nomi.query_postal_code(pc)["longitude"] for pc in HP_postcodes]
postcodes_lats = [nomi.query_postal_code(pc)["latitude"] for pc in HP_postcodes]
print(all_HP_coords.shape)
all_HP_coords = np.array(list(zip(postcodes_lats, postcodes_longs)))
all_HP_coords = all_HP_coords[~np.isnan(all_HP_coords).any(axis=1)]
print(all_HP_coords.shape)


# In[ ]:


from scipy import spatial

dist_matrix = scipy.spatial.distance.cdist(all_coords, all_HP_coords)


# In[ ]:


np.min(spatial.distance.cdist(all_coords, all_HP_coords), axis=0)


# In[ ]:


from scipy.spatial.distance import cdist
from geopy.distance import distance as geodist  # avoid naming confusion

sc_dist = cdist(
    all_coords, all_HP_coords, lambda u, v: geodist(u, v).km
)  # you can choose unit here


# In[ ]:


vfunc = np.vectorize(distance)
import itertools

vfunc(itertools.product(postcodes, HP_postcodes))


# In[ ]:


coords = epc_df[["LONGITUDE", "LATITUDE"]].to_numpy()
print(coords.shape)

dist.query_postal_code("BS4", "CO15")


# In[ ]:


from scipy import spatial

dist_matrix = scipy.spatial.distance.cdist(coords, coords)


# In[ ]:


newport_ri = (41.49008, -71.312796)
cleveland_oh = (41.499498, -81.695391)
print(distance.distance(newport_ri, cleveland_oh).km)


# In[ ]:
