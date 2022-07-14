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


from asf_core_data.pipeline.data_joining import merge_install_dates

from asf_core_data.utils.visualisation import easy_plotting, kepler

import ipywidgets
from ipywidgets import interact, FloatSlider, fixed

from scipy import spatial

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
    "UPRN",
]

epc_df = epc_data.load_preprocessed_epc_data(
    data_path=LOCAL_DATA_DIR, batch="newest", usecols=EPC_FEAT_SELECTION
)
epc_df.shape

# %%
epc_df_with_MCS = merge_install_dates.manage_hp_install_dates(epc_df)
epc_df_with_MCS.head()

# %%
epc_df_with_MCS.to_csv("epc_df_with_mcs.csv")

# %%
from asf_core_data.getters.supplementary_data.deprivation import imd_data

imd_df = imd_data.get_gb_imd_data(data_path=LOCAL_DATA_DIR)

# %%
imd_df.head()

# %%
epc_df = imd_data.merge_imd_with_other_set(
    imd_df, epc_df_with_MCS, postcode_label="Postcode"
)

# %%
epc_df.shape

# %%
visitor_cond = (
    (epc_df["TENURE"] == "owner-occupied")
    & (epc_df["CURRENT_ENERGY_RATING"].isin(["E", "D", "C"]))
    & (epc_df["IMD Decile"] >= 5)
    & (~epc_df["HP_INSTALLED"])
)
###  | ((epc_df['PROPERTY_TYPE'] == 'Flat')
# & (epc_df['IMD Decile'] >=5)
# & (~epc_df['HP_INSTALLED'])))
visitor_cond

visitor_epc_df = epc_df.loc[visitor_cond]

# %%
visitor_epc_df.shape

# %%
visitor_epc_df["HP_INSTALLED"].value_counts(dropna=False, normalize=True)

# %%
epc_df["HP_INSTALLED"].value_counts(dropna=False, normalize=True)

# %%

# %%
epc_df["HP_INSTALLED"].value_counts(dropna=False, normalize=False)

# %%
easy_plotting.plot_subcategory_distribution(
    visitor_epc_df,
    "BUILT_FORM",
    Path(LOCAL_DATA_DIR) / "outputs/figures/",
    plot_title="Property types for potential visitor homes",
    y_label="",
    x_label="",
    y_ticklabel_type="m",
    x_tick_rotation=45,
)

# %%
easy_plotting.plot_subcategory_distribution(
    epc_df,
    "BUILT_FORM",
    Path(LOCAL_DATA_DIR) / "outputs/figures/",
    plot_title="Property types for potential visitor homes",
    y_label="",
    x_label="",
    y_ticklabel_type="m",
    x_tick_rotation=45,
)

# %%

# %%
from asf_core_data.utils.visualisation import easy_plotting

easy_plotting.plot_subcats_by_other_subcats(
    epc_df,
    "HP_INSTALLED",
    "BUILT_FORM",
    Path(LOCAL_DATA_DIR) / "outputs/figures/",
    plotting_colors="viridis",
    legend_loc="outside",
    plot_title="HP status by built form",
)

# %%
coord_df = coordinates.get_postcode_coordinates(data_path=LOCAL_DATA_DIR)
coord_df.head()
coord_df["POSTCODE"] = coord_df["POSTCODE"].str.replace(" ", "")

epc_df = epc_df.rename(columns={"Postcode": "POSTCODE"})
epc_df.drop(columns=["LONGITUDE", "LATITUDE"], inplace=True)
epc_df = pd.merge(epc_df, coord_df, on=["POSTCODE"], how="left")
print(epc_df.shape)

# %%
epc_df.head()

# %%
hp_props = epc_df.loc[epc_df["HP_INSTALLED"]]
hp_props.shape

# %%
visitor_epc_df = epc_df.loc[visitor_cond]
print(visitor_epc_df.shape)

# %%
hp_props["LONGITUDE"].value_counts()

# %%
from keplergl import KeplerGl
import yaml

# network_df = pd.read_csv("network.csv")

# config_file = 'network_config.txt'
# with open(config_file, "r") as infile:
#     config = infile.read()
#     config = yaml.load(config, Loader=yaml.FullLoader)

# hp_props['LONGITUDE'] = hp_props['LONGITUDE'].astype('float')
# visitor_epc_df['LONGITUDE'] = visitor_epc_df['LONGITUDE'].astype('float')
# hp_props['LATITUDE'] = hp_props['LATITUDE'].astype('float')
# visitor_epc_df['LATITUDE'] = visitor_epc_df['LATITUDE'].astype('float')

network_map = KeplerGl(height=500)  # , config=config)

network_map.add_data(data=hp_props[["LONGITUDE", "LATITUDE"]], name="Heat pump homes")

network_map.add_data(
    data=visitor_epc_df[["LONGITUDE", "LATITUDE"]], name="Visitor homes"
)

network_map

# %%
network_map.save_to_html(file_name="network_test2.html")

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

    return distance, duration


def to_Cartesian(lat, lng):
    R = 6367  # radius of the Earth in kilometers

    x = R * np.cos(lat) * np.cos(lng)
    y = R * np.cos(lat) * np.sin(lng)
    z = R * np.sin(lat)
    return np.array((x, y, z)).T


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

# %%
df = epc_df

terraced = [
    "Enclosed Mid-Terrace",
    "Enclosed End-Terrace",
    "End-Terrace",
    "Mid-Terrace",
]
no_cond = ~df["LATITUDE"].isna()
flat = df["PROPERTY_TYPE"] == "Flat"
terraced_house = (df["PROPERTY_TYPE"] == "House") & (df["BUILT_FORM"].isin(terraced))
detached_house = (df["PROPERTY_TYPE"] == "House") & (df["BUILT_FORM"] == "Detached")
semi_house = (df["PROPERTY_TYPE"] == "House") & (df["BUILT_FORM"] == "Semi-Detached")

property_types = ["Flat", "Semi-detached house", "Detached House", "Terraced House"]
cond_dict = {
    "Flat": flat,
    "Semi-detached house": semi_house,
    "Detached House": detached_house,
    "Terraced House": terraced_house,
}

df = df[df["LATITUDE"].notna()]
df = df[df["LONGITUDE"].notna()]

# %%
import scipy

scipy.__version__


# %%
def prepare_coords(df):

    coords_org = df[["LATITUDE", "LONGITUDE"]].to_numpy()

    coords = coords_org.copy()

    coords = np.deg2rad(coords)
    coords = to_Cartesian(coords[:, 0], coords[:, 1])

    return coords, coords_org


from collections import defaultdict


local_authorities = df["LOCAL_AUTHORITY_LABEL"].unique()

host_ratio_widget = FloatSlider(min=0.0, max=100.0, step=1, value=1)
visitor_ratio_widget = FloatSlider(min=0.0, max=100.0, step=1, value=5)

v_max_widget = FloatSlider(min=0, max=50, step=1, value=10)
n_open_days_widget = FloatSlider(min=0, max=50, step=1, value=3)
d_max_widget = FloatSlider(min=0, max=50, step=1, value=25)

LA_widget = ipywidgets.SelectMultiple(
    options=local_authorities,
    value=[],
    # rows=10,
    description="Local authorities",
    disabled=False,
)


@interact(
    df=fixed(df),
    property_type=property_types,
    same_prop_type=[True, False],
    host_ratio=host_ratio_widget,
    vistor_ratio=visitor_ratio_widget,
    v_max=v_max_widget,
    n_open_days=n_open_days_widget,
    d_max=d_max_widget,
    local_auth=LA_widget,
)
def compute_network_measure(
    df,
    property_type,
    same_prop_type,
    host_ratio,
    vistor_ratio,
    v_max,
    n_open_days,
    d_max,
    local_auth,
):

    label = property_type
    conds = cond_dict[property_type]

    v_max = v_max * n_open_days
    host_ratio = host_ratio / 100
    vistor_ratio = vistor_ratio / 100

    non_hp_samples = df.loc[~df["HP_INSTALLED"] & conds]
    hp_samples = df.loc[df["HP_INSTALLED"]]
    hp_same_type_samples = df.loc[df["HP_INSTALLED"] & conds]

    if local_auth:
        non_hp_samples = non_hp_samples.loc[
            non_hp_samples["LOCAL_AUTHORITY_LABEL"].isin(local_auth)
        ]

    print("Before subsampling:")
    print("# Props without HPs: {}".format(non_hp_samples.shape[0]))
    print("# Props with HPs: {}".format(hp_samples.shape[0]))
    print("# Similar props with with HPs: {}".format(hp_same_type_samples.shape[0]))

    n_non_hp_samples = int(non_hp_samples.shape[0] * vistor_ratio)
    n_hp_samples = int(hp_samples.shape[0] * host_ratio)
    n_hp_same_type_samples = int(hp_same_type_samples.shape[0] * host_ratio)

    non_hp_samples = non_hp_samples.sample(frac=1, random_state=42)[:n_non_hp_samples]
    hp_samples = hp_samples.sample(frac=1, random_state=42)[:n_hp_samples]
    hp_same_type_samples = hp_same_type_samples.sample(frac=1, random_state=42)[
        :n_hp_same_type_samples
    ]

    print()
    print("After subsampling:")
    print("# Props without HPs: {}".format(non_hp_samples.shape[0]))
    print("# Props with HPs: {}".format(hp_samples.shape[0]))
    print("# Similar props with with HPs: {}".format(hp_same_type_samples.shape[0]))

    non_hp_coords, non_hp_coords_org = prepare_coords(non_hp_samples)
    hp_coords, hp_coords_org = prepare_coords(hp_samples)
    sim_hp_coords, sim_hp_coords_org = prepare_coords(hp_same_type_samples)

    hp_prop_dict = {"any": hp_coords, "similar": sim_hp_coords}
    org_coords_dict = {"any": hp_coords_org, "similar": sim_hp_coords_org}

    version = "similar" if same_prop_type else "any"

    failed = []

    hp_set_coords = hp_prop_dict[version]
    hp_set_coords_org = org_coords_dict[version]

    tree = spatial.KDTree(hp_set_coords)
    distances, indices = tree.query(non_hp_coords)

    v_collection = []
    host_array = np.zeros((indices.shape[0]))

    host_tree = spatial.KDTree(hp_set_coords)
    visitor_tree = spatial.KDTree(non_hp_coords)
    v_counts_old = host_tree.count_neighbors(visitor_tree, r=d_max, cumulative=False)
    match_indices = host_tree.query_ball_tree(visitor_tree, r=d_max)

    # print(match_indices)
    v_counts = np.array([[len(x) for x in match_indices]])
    match_indices = list(set([x for xs in match_indices for x in xs]))
    capacity_array = np.zeros((indices.shape[0]))
    capacity_array[match_indices] = 1

    original_v_counts = v_counts.copy()

    over_cap_ratio = np.count_nonzero(v_counts > v_max) / v_counts.shape[1] * 100
    v_counts[v_counts >= v_max] = v_max

    v_data = np.zeros((v_counts.shape[1], 4))
    v_data[:, 0] = v_counts
    v_data[:, 1] = original_v_counts
    v_data[:, 2] = hp_set_coords_org[:, 0]
    v_data[:, 3] = hp_set_coords_org[:, 1]

    for ind in set(indices):

        same_ind = np.where(indices == ind)
        all_distances = distances[same_ind]

        all_distances = all_distances[all_distances < d_max]
        v = all_distances.shape[0]
        v = v if v <= v_max else v_max
        host_array[same_ind] = v
        v_collection.append(v)

    max_vis_dict = defaultdict(int)

    visitor_array = np.zeros((non_hp_coords.shape[0]))

    for i in range(non_hp_coords.shape[0]):

        closest_ind = indices[i]
        if max_vis_dict[closest_ind] < v_max:
            max_vis_dict[closest_ind] += 1
            visitor_array[i] = 1

    # print('Host array', host_array)
    # print('Visitor array', visitor_array)

    capacity = np.sum(v_counts) / n_non_hp_samples
    coverage = capacity_array.sum() / capacity_array.shape[0]

    print()
    print("Results {}".format(property_type))
    print("=========")
    print("Capacity: {}%".format(round(capacity * 100, 2)))
    print("Coverage: {}%".format(round(coverage * 100, 2)))
    print("Over cap ratio: {}%".format(round(over_cap_ratio)))

    network = np.zeros((indices.shape[0], 7))
    network[:, 0] = non_hp_coords_org[:, 0]
    network[:, 1] = non_hp_coords_org[:, 1]
    network[:, 2] = hp_set_coords_org[indices][:, 0]
    network[:, 3] = hp_set_coords_org[indices][:, 1]
    network[:, 4] = distances
    network[:, 5] = host_array
    network[:, 6] = capacity_array

    v_df = pd.DataFrame(
        v_data, columns=["Counts", "Original counts", "LATITUDE", "LONGITUDE"]
    )
    v_df.to_csv("v_data.csv")

    network_df = pd.DataFrame(
        network,
        columns=[
            "non-HP LATITUDE",
            "non-HP LONGITUDE",
            "HP LATITUDE",
            "HP LONGITUDE",
            "DISTANCE",
            "# Connections",
            "Coverage",
        ],
    )

    network_df["Coverage"] = network_df["Coverage"].astype("bool")

    network_df.to_csv("network.csv")


# %%
from keplergl import KeplerGl
import yaml

network_df = pd.read_csv("network.csv")
v_data = pd.read_csv("v_data.csv")

config_file = "network_config.txt"
with open(config_file, "r") as infile:
    config = infile.read()
    config = yaml.load(config, Loader=yaml.FullLoader)

network_map = KeplerGl(height=500, config=config)

network_map.add_data(
    data=network_df[
        [
            "non-HP LATITUDE",
            "non-HP LONGITUDE",
            "HP LATITUDE",
            "HP LONGITUDE",
            "DISTANCE",
        ]
    ],
    name="Network",
)

network_map.add_data(
    data=network_df[["non-HP LATITUDE", "non-HP LONGITUDE", "Coverage"]],
    name="Coverage",
)

# network_map.add_data(
#     data=network_df[['HP LATITUDE', 'HP LONGITUDE',  '# Connections']],
#     name="Connections")

network_map.add_data(data=v_data[["LATITUDE", "LONGITUDE", "Counts"]], name="Counts")

network_map.add_data(
    data=v_data[["LATITUDE", "LONGITUDE", "Original counts"]], name="Original counts"
)


network_map

# %%

# %%
with open("network_config.txt", "w") as outfile:
    outfile.write(str(network_map.config))

network_map.save_to_html(file_name="Network_detached.html")

# %%
from keplergl import KeplerGl
import yaml

network_df = pd.read_csv("network.csv")

config_file = "network_config.txt"
with open(config_file, "r") as infile:
    config = infile.read()
    config = yaml.load(config, Loader=yaml.FullLoader)

network_map = KeplerGl(height=500, config=config)

network_map.add_data(
    data=network_df[["non-HP LATITUDE", "non-HP LONGITUDE", "Coverage"]],
    name="Coverage",
)

# network_map.add_data(
#     data=network_df[['HP LATITUDE', 'HP LONGITUDE',  '# Connections']],
#     name="Connections")


network_map

# %% [markdown]
# ![Screenshot%202022-07-04%20at%2013.55.39.png](attachment:Screenshot%202022-07-04%20at%2013.55.39.png)

# %% [markdown]
# ![Screenshot%202022-07-04%20at%2013.59.32.png](attachment:Screenshot%202022-07-04%20at%2013.59.32.png)

# %%
