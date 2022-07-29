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

import geopy

import requests
import json
from time import sleep
import random

import gradio as gr


from keplergl import KeplerGl
import yaml
import asf_core_data

from asf_core_data.getters.epc import epc_data, data_batches
from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.getters.supplementary_data.geospatial import coordinates
from asf_core_data.pipeline.preprocessing import preprocess_epc_data, data_cleaning
from asf_core_data.utils.visualisation import easy_plotting
from asf_core_data.pipeline.data_joining import merge_install_dates


from asf_core_data.pipeline.data_joining import merge_install_dates

from asf_core_data.utils.visualisation import easy_plotting, kepler

from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.utils.visualisation import easy_plotting


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
epc_df_with_MCS = pd.read_csv("epc_df_with_mcs.csv")

# %%
# epc_df_with_MCS.to_csv('epc_df_with_mcs.csv')

# %%
imd_df = imd_data.get_gb_imd_data(data_path=LOCAL_DATA_DIR)

epc_df = imd_data.merge_imd_with_other_set(
    imd_df, epc_df_with_MCS, postcode_label="Postcode"
)

# %%
visitor_cond = (
    (epc_df["TENURE"] == "owner-occupied")
    & (epc_df["CURRENT_ENERGY_RATING"].isin(["E", "D", "C"]))
    & (epc_df["IMD Decile"] >= 5)
    & (~epc_df["HP_INSTALLED"])
)

visitor_epc_df = epc_df.loc[visitor_cond]

# %%
visitor_epc_df["HP_INSTALLED"].value_counts(dropna=False, normalize=True)

# %%
epc_df["HP_INSTALLED"].value_counts(dropna=False, normalize=True)

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

# %%
epc_df = epc_df.rename(columns={"Postcode": "POSTCODE"})
# epc_df.drop(columns=['LONGITUDE', 'LATITUDE'], inplace=True)
epc_df = pd.merge(epc_df, coord_df, on=["POSTCODE"], how="left")
print(epc_df.shape)

# %%
hp_props = epc_df.loc[epc_df["HP_INSTALLED"]]
hp_props.shape

# %%
visitor_epc_df = epc_df.loc[visitor_cond]
print(visitor_epc_df.shape)

# %%
from keplergl import KeplerGl
import yaml

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

    v_max = int(v_max * n_open_days)
    host_ratio = host_ratio / 100
    vistor_ratio = vistor_ratio / 100

    local_auth_str = local_auth if local_auth != "all GB (not recommended)" else "ßßGB"
    print(local_auth)

    settings_string = (
        property_type
        + str(same_prop_type)
        + str(host_ratio)
        + str(vistor_ratio)
        + str(v_max)
        + str(d_max)
        + local_auth_str
    )

    non_hp_samples = df.loc[~df["HP_INSTALLED"] & conds]
    hp_samples = df.loc[df["HP_INSTALLED"]]
    hp_same_type_samples = df.loc[df["HP_INSTALLED"] & conds]

    if local_auth != "all GB (not recommended)":
        non_hp_samples = non_hp_samples.loc[
            non_hp_samples["LOCAL_AUTHORITY_LABEL"].isin([local_auth])
        ]
        # hp_samples = non_hp_samples.loc[hp_samples['LOCAL_AUTHORITY_LABEL'].isin(local_auth)]
        # hp_same_type_samples = hp_same_type_samples.loc[hp_samples['LOCAL_AUTHORITY_LABEL'].isin(local_auth)]

    print("Before subsampling:")
    print("# Props without HPs: {}".format(non_hp_samples.shape[0]))
    print("# Props with HPs: {}".format(hp_samples.shape[0]))
    print("# Similar props with with HPs: {}".format(hp_same_type_samples.shape[0]))

    before = "Overall situation:\n# Props without HPs: {}\n# Props with HPs: {}\n# Similar props with with HPs: {}".format(
        non_hp_samples.shape[0], hp_samples.shape[0], hp_same_type_samples.shape[0]
    )
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
    print("# Similar props with HPs: {}".format(hp_same_type_samples.shape[0]))

    after = "After subsampling:\n# Props without HPs: {}\n# Props with HPs: {}\n# Similar props with HPs: {}".format(
        non_hp_samples.shape[0], hp_samples.shape[0], hp_same_type_samples.shape[0]
    )
    non_hp_coords, non_hp_coords_org = prepare_coords(non_hp_samples)
    hp_coords, hp_coords_org = prepare_coords(hp_samples)
    sim_hp_coords, sim_hp_coords_org = prepare_coords(hp_same_type_samples)

    print(local_auth)
    if local_auth != "all GB (not recommended)":
        out_of_la_all = np.array(
            ~hp_samples["LOCAL_AUTHORITY_LABEL"].isin([local_auth])
        )
        out_of_la_sim = np.array(
            ~hp_same_type_samples["LOCAL_AUTHORITY_LABEL"].isin([local_auth])
        )
        out_of_la_dict = {"any": out_of_la_all, "similar": out_of_la_sim}

    hp_prop_dict = {"any": hp_coords, "similar": sim_hp_coords}
    org_coords_dict = {"any": hp_coords_org, "similar": sim_hp_coords_org}

    version = "similar" if same_prop_type else "any"

    hp_set_coords = hp_prop_dict[version]
    hp_set_coords_org = org_coords_dict[version]
    n_hp_props = hp_set_coords_org.shape[0]

    tree = spatial.KDTree(hp_set_coords)  # remove
    distances, indices = tree.query(non_hp_coords)  # remove

    v_collection = []  # remove
    host_array = np.zeros((indices.shape[0]))  # remove

    host_tree = spatial.KDTree(hp_set_coords)
    visitor_tree = spatial.KDTree(non_hp_coords)
    v_counts_old = host_tree.count_neighbors(
        visitor_tree, r=d_max, cumulative=False
    )  # remove
    match_indices = host_tree.query_ball_tree(visitor_tree, r=d_max)

    host_opts_org = np.array([[len(x) for x in match_indices]])[0]

    filter_host_data = False
    zero_game = False

    if not [x for xs in match_indices for x in xs]:

        vistor_opts_org = np.zeros(n_non_hp_samples).astype(int)
        vistor_opts = np.zeros(n_non_hp_samples).astype(int)
        visitor_match = np.zeros((n_non_hp_samples)).astype(int)

        host_opts_org = np.zeros(n_hp_props).astype(int)
        host_opts = np.zeros(n_hp_props).astype(int)

        n_countable_hp_props = host_opts_org.shape[0]
        zero_game = True

    else:

        org_indices_counts = np.array([x for xs in match_indices for x in xs])[0]

        unique_values, unique_counts = np.unique(org_indices_counts, return_counts=True)
        vistor_opts_org = np.zeros(n_non_hp_samples)
        vistor_opts_org[unique_values] = unique_counts

        v_counts = np.array([[len(x) for x in match_indices]])  # remove
        original_v_counts = v_counts.copy()  # remove

        upd_match_indices = []
        for i, host_matches in enumerate(match_indices):
            upd_match_indices.append(
                random.sample(host_matches, len(host_matches))[:v_max]
            )

        #             v_counts = np.array([[len(x) for x in upd_match_indices]])[0] # remove all
        #             upd_match_indices_counts = np.array([x for xs in upd_match_indices for x in xs]) # remove all
        #             upd_match_indices_unique = list(set([x for xs in upd_match_indices for x in xs])) # remove all
        #             coverage_array =  np.zeros((n_non_hp_samples)) # remove all
        #             coverage_array[upd_match_indices_unique] = 1 # remove all

        host_opts = np.array([[len(x) for x in upd_match_indices]])[0]

        indices_counts = np.array([x for xs in upd_match_indices for x in xs])
        unique_values, unique_counts = np.unique(indices_counts, return_counts=True)
        vistor_opts = np.zeros(n_non_hp_samples)
        vistor_opts[unique_values] = unique_counts

        upd_match_indices_unique = np.unique(indices_counts)
        visitor_match = np.zeros((n_non_hp_samples))
        visitor_match[upd_match_indices_unique] = 1

        print(visitor_match.shape)

        print(host_opts.shape)
        print(host_opts_org.shape)

        if local_auth != "all GB (not recommended)":
            out_of_la = out_of_la_dict[version]
            host_w_match = host_opts > 0
            n_countable_hp_props = (~out_of_la | host_w_match).sum()
            show_home_filter = ~out_of_la | host_w_match
            filter_host_data = True
        else:
            n_countable_hp_props = host_opts_org.shape[0]

    over_cap_ratio = (
        np.count_nonzero(host_opts_org >= v_max) / n_countable_hp_props * 100
    )

    #         print(v_counts.shape)

    #         mask = np.where(v_counts==0)                 # filter out values larger than 5
    #         print(mask)

    #         print(hp_set_coords_org.shape)
    #         no_match_hosts = hp_set_coords_org[mask]
    #         print(no_match_hosts.shape)

    #         #if no_match_hosts.shape[0]:
    #         no_match_hosts = pd.DataFrame(no_match_hosts, columns = ['LATITUDE', 'LONGITUDE'])
    #         no_match_hosts.to_csv('unmatched_hosts.csv')

    # over_cap_ratio = np.count_nonzero(original_v_counts > v_max)/original_v_counts.shape[1]*100

    # v_counts[v_counts >= v_max] = v_max
    host_opts[host_opts >= v_max] = v_max

    connections = np.zeros((n_hp_props * v_max, 5))

    counter = 0

    for i in range(n_hp_props):

        m = host_opts[i]

        for j in range(m):
            connections[counter, :2] = hp_set_coords_org[i]
            connections[counter, 2:4] = non_hp_coords_org[upd_match_indices[i][j]]

            connections[counter, 4] = round(
                geopy.distance.distance(
                    hp_set_coords_org[i], non_hp_coords_org[upd_match_indices[i][j]]
                ).km,
                1,
            )

            counter += 1

    connections = connections[~np.all(connections == 0, axis=1)]

    connections_df = pd.DataFrame(
        connections,
        columns=[
            "LATITUDE source",
            "LONGITUDE source",
            "LATITUDE target",
            "LONGITUDE target",
            "Distance",
        ],
    )

    if filter_host_data and not zero_game:

        host_data = np.zeros((n_countable_hp_props, 5))
        host_data[:, 0] = hp_set_coords_org[show_home_filter, 0]
        host_data[:, 1] = hp_set_coords_org[show_home_filter, 1]
        host_data[:, 2] = host_opts[show_home_filter]
        host_data[:, 3] = host_opts_org[show_home_filter]
        host_data[:, 4] = host_w_match[show_home_filter]

    else:

        host_data = np.zeros((n_hp_props, 5))
        host_data[:, 0] = hp_set_coords_org[:, 0]
        host_data[:, 1] = hp_set_coords_org[:, 1]
        host_data[:, 2] = host_opts
        host_data[:, 3] = host_opts_org
        host_data[:, 4] = host_opts > 0

    host_data_df = pd.DataFrame(
        host_data,
        columns=[
            "LATITUDE",
            "LONGITUDE",
            "Visitor matches (capped)",
            "Visitor matches",
            "Host coverage",
        ],
    )

    visitor_data = np.zeros((n_non_hp_samples, 5))
    visitor_data[:, 0] = non_hp_coords_org[:, 0]
    visitor_data[:, 1] = non_hp_coords_org[:, 1]
    visitor_data[:, 2] = vistor_opts
    visitor_data[:, 3] = vistor_opts_org
    visitor_data[:, 4] = visitor_match

    visitor_data_df = pd.DataFrame(
        visitor_data,
        columns=[
            "LATITUDE",
            "LONGITUDE",
            "Host matches (capped)",
            "Host matches",
            "Visitor coverage",
        ],
    )

    visitor_data_df.to_csv("visitor_df.csv")
    host_data_df.to_csv("host_df.csv")
    connections_df.to_csv("connections_df.csv")
    #         for ind in set(indices):  # remove

    #             same_ind = np.where(indices == ind)
    #             all_distances = distances[same_ind]

    #             all_distances = all_distances[all_distances < d_max]
    #             v = all_distances.shape[0]
    #             v = v if v <= v_max else v_max
    #             host_array[same_ind] = v
    #             v_collection.append(v)

    #         max_vis_dict = defaultdict(int)

    #         visitor_array = np.zeros((non_hp_coords.shape[0]))

    #         for i in range(non_hp_coords.shape[0]):

    #             closest_ind = indices[i]
    #             if max_vis_dict[closest_ind] < v_max:
    #                 max_vis_dict[closest_ind] +=1
    #                 visitor_array[i] = 1

    # print('Host array', host_array)
    # print('Visitor array', visitor_array)

    capacity = np.sum(host_opts) / n_non_hp_samples
    coverage = visitor_match.sum() / visitor_match.shape[0]

    capacity = round(capacity * 100, 2)
    coverage = round(coverage * 100, 2)
    over_cap_ratio = round(over_cap_ratio)

    print()
    print("Results {}".format(property_type))
    print("=========")
    print("Capacity:\t {}%".format(capacity))
    print("Coverage: {}%".format(coverage))
    print("Over cap ratio: {}%".format(over_cap_ratio))

    local_auth_output = (
        " in " + local_auth if local_auth != "all GB (not recommended)" else " in GB"
    )

    output = "Network for {}s{}\n=========\nCapacity:\t{}%\nCoverage:\t{}%\nOver cap ratio:\t{}%".format(
        property_type, local_auth_output, capacity, coverage, over_cap_ratio
    )

    output = before + "\n\n" + after + "\n\n" + output
    print(output)

    #         network = np.zeros((indices.shape[0],7))
    #         network[:,0] = non_hp_coords_org[:,0]
    #         network[:,1] = non_hp_coords_org[:,1]
    #         network[:,2] = hp_set_coords_org[indices][:,0]
    #         network[:,3] = hp_set_coords_org[indices][:,1]
    #         network[:,4] = distances
    #         network[:,5] = host_array
    #         network[:,6] = coverage_array

    #         v_df = pd.DataFrame(v_data, columns = ['Counts', 'Original counts', 'LATITUDE', "LONGITUDE"])
    #         v_df.to_csv('v_data.csv')

    #         network_df = pd.DataFrame(network, columns=['non-HP LATITUDE', 'non-HP LONGITUDE',
    #                                                     'HP LATITUDE', 'HP LONGITUDE',
    #                                                     'DISTANCE',
    #                                                     '# Connections', 'Coverage',

    #                                                    ])

    #         network_df['Coverage'] = network_df['Coverage'].astype('bool')

    #         network_df.to_csv("network.csv")

    #         network = np.zeros((sources.shape[0], 4))
    #         network[:,0] = sources[:,0]
    #         network[:,1] = sources[:,1]
    #         network[:,2] = targets[:,1]
    #         network[:,3] = sources[:,2]

    #         network_df = pd.DataFrame(network, columns=['non-HP LATITUDE con', 'non-HP LONGITUDE con',
    #                                                     'HP LATITUDE con', 'HP LONGITUDE con',

    #                                                    ])

    create_output_map(connections_df, host_data_df, visitor_data_df, settings_string)
    # map_ = '<html><body><p>You can check out the map <a href="file/Network_for_gradio.html">here</a>.</p></body></html>'
    map_ = '<iframe src="file/maps/Generated_network_map_{}.html" style="border:0px #ffffff none;" name="myiFrame" scrolling="no" frameborder="1" marginheight="0px" marginwidth="0px" height="600px" width="800px" allowfullscreen></iframe>'.format(
        settings_string
    )

    return output, map_


# %%
# host_df = pd.read_csv('host_df.csv')
# visitor_df = pd.read_csv('visitor_df.csv')
# # connections_df = pd.read_csv('connections_df.csv')


# %%
def create_output_map(connections_df, host_df, visitor_df, settings_string):

    config_file = "network_gradio_config.txt"
    with open(config_file, "r") as infile:
        config = infile.read()
        config = yaml.load(config, Loader=yaml.FullLoader)

    network_map = KeplerGl(height=500, config=config)

    network_map.add_data(data=connections_df, name="Network")

    network_map.add_data(
        data=host_df[
            [
                "LATITUDE",
                "LONGITUDE",
                "Visitor matches (capped)",
                "Visitor matches",
                "Host coverage",
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
                "Visitor coverage",
            ]
        ],
        name="Visitor homes",
    )

    network_map.save_to_html(
        file_name="maps/Generated_network_map_{}.html".format(settings_string)
    )


# network_map

# network_map.save_to_html(
# file_name="Network_for_gradio.html")

# %%

host_df = pd.read_csv("host_df.csv")
visitor_df = pd.read_csv("visitor_df.csv")
connections_df = pd.read_csv("connections_df.csv")


config_file = "network_gradio_config.txt"
with open(config_file, "r") as infile:
    config = infile.read()
    config = yaml.load(config, Loader=yaml.FullLoader)

network_map = KeplerGl(height=500)  # , config=config)

network_map.add_data(data=connections_df, name="Network")


network_map.add_data(
    data=host_df[
        [
            "LATITUDE",
            "LONGITUDE",
            "Visitor matches (capped)",
            "Visitor matches",
            "Host coverage",
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
            "Visitor coverage",
        ]
    ],
    name="Visitor homes",
)

network_map.save_to_html(file_name="Network_for_gradio.html")


network_map


# %%
with open("network_gradio_config.txt", "w") as outfile:
    outfile.write(str(network_map.config))

network_map.save_to_html(file_name="Network_for_gradio.html")

# %%
local_authorities = list(df["LOCAL_AUTHORITY_LABEL"].unique())
local_authorities = [
    "Greenwich",
    "Manchester",
    "Glasgow City",
    "Orkney Islands",
    "all GB (not recommended)",
]

demo = gr.Interface(
    fn=compute_network_measure,
    inputs=[
        gr.inputs.Radio(
            property_types, label="Property Type", default="Detached House"
        ),
        gr.inputs.Radio(
            [True, False], label="Show home of same property", default=True
        ),
        gr.inputs.Slider(0, 100, default=1, step=1, label="Host ratio (%)"),
        gr.inputs.Slider(0, 100, default=5, step=1, label="Visitor ratio (%)"),
        gr.inputs.Slider(1, 50, default=10, step=1, label="Max visitors"),
        gr.inputs.Slider(1, 50, default=3, step=1, label="Open days"),
        gr.inputs.Slider(1, 50, default=25, step=1, label="Max distance"),
        gr.inputs.Radio(
            local_authorities, default="Manchester", label="Local authorities"
        ),
    ],
    outputs=["text", "html"],
    title="Network of Show Homes",
    allow_screenshot=True,
)


demo.launch(share=True)

# %%
example = '<html><body><p>You can check out the map <a href="Network_detached.html">here</a>.</p></body></html>'


from IPython.core.display import display, HTML

display(HTML(example))

# %%
