# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: epc_data_analysis
#     language: python
#     name: epc_data_analysis
# ---

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("total_distances.csv")
df.head()

# %%
flat_any = df.loc[(df["Property Type"] == "Flat") & (df["HP Property"] == "any")]
semi_detached_any = df.loc[
    (df["Property Type"] == "Semi-detached house") & (df["HP Property"] == "any")
]
detached_any = df.loc[
    (df["Property Type"] == "Detached House") & (df["HP Property"] == "any")
]
terraced_any = df.loc[
    (df["Property Type"] == "Terraced House") & (df["HP Property"] == "any")
]
all_any = df.loc[(df["Property Type"] == "All") & (df["HP Property"] == "any")]

flat_similar = df.loc[
    (df["Property Type"] == "Flat") & (df["HP Property"] == "similar")
]
semi_detached_similar = df.loc[
    (df["Property Type"] == "Semi-detached house") & (df["HP Property"] == "similar")
]
detached_similar = df.loc[
    (df["Property Type"] == "Detached House") & (df["HP Property"] == "similar")
]
terraced_similar = df.loc[
    (df["Property Type"] == "Terraced House") & (df["HP Property"] == "similar")
]
all_similar = df.loc[(df["Property Type"] == "All") & (df["HP Property"] == "similar")]

# %%
print("All")
print("-----")
print(all_any.shape[0])
print(round(all_any["Distance"].mean(), 1))
print(round(all_any["Duration"].mean(), 1))
print()
print(round(all_similar["Distance"].mean(), 1))
print(round(all_similar["Duration"].mean(), 1))
print()
print("Flat")
print("-----")
print(flat_any.shape[0])
print(round(flat_any["Distance"].mean(), 1))
print(round(flat_any["Duration"].mean(), 1))
print()
print(flat_similar.shape[0])
print(round(flat_similar["Distance"].mean(), 1))
print(round(flat_similar["Duration"].mean(), 1))
print()
print("Detached House")
print("-----")
print(detached_any.shape[0])
print(round(detached_any["Distance"].mean(), 1))
print(round(detached_any["Duration"].mean(), 1))
print()
print(detached_similar.shape[0])
print(round(detached_similar["Distance"].mean(), 1))
print(round(detached_similar["Duration"].mean(), 1))
print()
print("Semi-detached House")
print("-----")
print(semi_detached_any.shape[0])
print(round(semi_detached_any["Distance"].mean(), 1))
print(round(semi_detached_any["Duration"].mean(), 1))
print()
print(semi_detached_similar.shape[0])
print(round(semi_detached_similar["Distance"].mean(), 1))
print(round(semi_detached_similar["Duration"].mean(), 1))
print()
print("Terraced House")
print("-----")
print(terraced_any.shape[0])
print(round(terraced_any["Distance"].mean(), 1))
print(round(terraced_any["Duration"].mean(), 1))
print()
print(terraced_similar.shape[0])
print(round(terraced_similar["Distance"].mean(), 1))
print(round(terraced_similar["Duration"].mean(), 1))
print()


# %%
print("All")
print("-----")
print(all_any.shape[0])
print(round(all_any["Distance"].median(), 1))
print(round(all_any["Duration"].median(), 1))
print()
print(round(all_similar["Distance"].median(), 1))
print(round(all_similar["Duration"].median(), 1))
print()
print("Flat")
print("-----")
print(flat_any.shape[0])
print(round(flat_any["Distance"].median(), 1))
print(round(flat_any["Duration"].median(), 1))
print()
print(flat_similar.shape[0])
print(round(flat_similar["Distance"].median(), 1))
print(round(flat_similar["Duration"].median(), 1))
print()
print("Detached House")
print("-----")
print(detached_any.shape[0])
print(round(detached_any["Distance"].median(), 1))
print(round(detached_any["Duration"].median(), 1))
print()
print(detached_similar.shape[0])
print(round(detached_similar["Distance"].median(), 1))
print(round(detached_similar["Duration"].median(), 1))
print()
print("Semi-detached House")
print("-----")
print(semi_detached_any.shape[0])
print(round(semi_detached_any["Distance"].median(), 1))
print(round(semi_detached_any["Duration"].median(), 1))
print()
print(semi_detached_similar.shape[0])
print(round(semi_detached_similar["Distance"].median(), 1))
print(round(semi_detached_similar["Duration"].median(), 1))
print()
print("Terraced House")
print("-----")
print(terraced_any.shape[0])
print(round(terraced_any["Distance"].median(), 1))
print(round(terraced_any["Duration"].median(), 1))
print()
print(terraced_similar.shape[0])
print(round(terraced_similar["Distance"].median(), 1))
print(round(terraced_similar["Duration"].median(), 1))
print()

# %%
distances = {
    "Flat": flat_any,
    "Semi-detached House": semi_detached_any,
    "Detached House": detached_any,
    "Terraced House": terraced_any,
    "All": all_any,
}

distances_similar = {
    "Flat": flat_similar,
    "Semi-detached House": semi_detached_similar,
    "Detached House": detached_similar,
    "Terraced House": terraced_similar,
    "All": all_any,
}

# %%
cond_labels = [
    "All",
    "Flat",
    "Terraced House",
    "Detached House",
    "Semi-detached House",
]


for label in cond_labels:
    sns.kdeplot(
        data=distances[label]["Distance"], fill=True, common_norm=False, alpha=0.1
    )
plt.legend(cond_labels)
plt.xlabel("Distance [km]")
plt.title("Distance to nearest HP property")
plt.xlim(-5, 75)
plt.savefig("Distance to nearest HP property.png", dpi=300)
plt.show()

for label in cond_labels:
    sns.kdeplot(
        data=distances_similar[label]["Distance"],
        fill=True,
        common_norm=False,
        alpha=0.1,
    )
plt.legend(cond_labels)
plt.xlabel("Distance [km]")
plt.title("Distance to nearest similar HP property")
plt.xlim(-5, 75)
plt.savefig("Distance to nearest similar HP property.png", dpi=300)
plt.show()

# %%

cond_labels = ["All", "Flat", "Terraced House", "Detached House", "Semi-detached House"]

for label in cond_labels:
    sns.kdeplot(
        data=distances[label]["Duration"], fill=True, common_norm=False, alpha=0.1
    )
plt.legend(cond_labels)
plt.xlabel("Travel Time [min]")
plt.title("Travel Time to nearest HP property")
plt.xlim(-5, 75)
plt.savefig("Travel Time to nearest HP property.png", dpi=300)
plt.show()

for label in cond_labels:
    sns.kdeplot(
        data=distances_similar[label]["Duration"],
        fill=True,
        common_norm=False,
        alpha=0.1,
    )
plt.legend(cond_labels)
plt.xlabel("Travel Time [min]")
plt.title("Travel Time to nearest similar HP property")
plt.xlim(-5, 75)
plt.savefig("Travel Time to nearest similar HP property.png", dpi=300)
plt.show()

# %%
cond_labels = ["All", "Flat", "Terraced House", "Detached House", "Semi-detached House"]

for label in cond_labels:
    dist_matrix = np.load(label + "_any.npy")
    dist_matrix = np.min(dist_matrix, axis=1)
    sns.kdeplot(data=dist_matrix, fill=True, common_norm=False, alpha=0.1)
plt.legend(cond_labels)
plt.xlabel("Distance [km]")
plt.title("Air distance to nearest HP property")
plt.xlim(-5, 75)
plt.savefig("Air distance to nearest HP property.png", dpi=300)
plt.show()

for label in cond_labels:
    dist_matrix_similar = np.load(label + "_similar.npy")
    dist_matrix_similar = np.min(dist_matrix_similar, axis=1)
    sns.kdeplot(data=dist_matrix_similar, fill=True, common_norm=False, alpha=0.1)

plt.legend(cond_labels)
plt.xlabel("Distance [km]")
plt.title("Air distance to nearest similar HP property")
plt.xlim(-5, 75)
plt.savefig("Air distance to nearest similar HP property.png", dpi=300)
plt.show()

# %%
from epc_data_analysis.config.kepler.kepler_config import (
    MAPS_CONFIG_PATH,
    MAPS_OUTPUT_PATH,
)
from epc_data_analysis.config.kepler import kepler_config as kepler

from keplergl import KeplerGl

config = kepler.get_config(kepler.MAPS_CONFIG_PATH + "distance.txt")


flat_similar = flat_similar.round({"Distance": 1, "Duration": 1})
semi_detached_similar = semi_detached_similar.round({"Distance": 1, "Duration": 1})
detached_similar = detached_similar.round({"Distance": 1, "Duration": 1})
terraced_similar = terraced_similar.round({"Distance": 1, "Duration": 1})

flat_similar.rename(columns={"Distance": "Distance [in km]"}, inplace=True)
semi_detached_similar.rename(columns={"Distance": "Distance [in km]"}, inplace=True)
detached_similar.rename(columns={"Distance": "Distance [in km]"}, inplace=True)
terraced_similar.rename(columns={"Distance": "Distance [in km]"}, inplace=True)


distance = KeplerGl(height=500, config=config)

distance.add_data(
    data=flat_similar[
        [
            "Latitude 1",
            "Longitude 1",
            "Latitude 2",
            "Longitude 2",
            "Distance [in km]",
            "Duration",
        ]
    ],
    name="Flat",
)

distance.add_data(
    data=semi_detached_similar[
        [
            "Latitude 1",
            "Longitude 1",
            "Latitude 2",
            "Longitude 2",
            "Distance [in km]",
            "Duration",
        ]
    ],
    name="Semi-detached House",
)

distance.add_data(
    data=detached_similar[
        [
            "Latitude 1",
            "Longitude 1",
            "Latitude 2",
            "Longitude 2",
            "Distance [in km]",
            "Duration",
        ]
    ],
    name="Detached House",
)

distance.add_data(
    data=terraced_similar[
        [
            "Latitude 1",
            "Longitude 1",
            "Latitude 2",
            "Longitude 2",
            "Distance [in km]",
            "Duration",
        ]
    ],
    name="Terraced House",
)

# distance.add_data(
#  data=all_similar[["Latitude 1", "Longitude 1", "Latitude 2", "Longitude 2", 'Distance', 'Duration']], name="All")

distance


# %%
kepler.save_config(distance, MAPS_CONFIG_PATH + "distance.txt")

distance.save_to_html(file_name=MAPS_OUTPUT_PATH + "Distances_similar.html")

# %%
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

epc_df = epc_data.load_preprocessed_epc_data(
    version="preprocessed_dedupl", usecols=EPC_FEAT_SELECTION
)

epc_df = feature_engineering.get_postcode_coordinates(epc_df)
epc_df["COORDS"] = list(zip(epc_df["LATITUDE"], epc_df["LONGITUDE"]))


# %%


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


with open("distances.csv", "a") as outfile:

    for i, conds in enumerate(
        [no_cond, flat, semi_house, detached_house, terraced_house]
    ):

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
