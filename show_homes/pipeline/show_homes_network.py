import numpy as np
from scipy import spatial
import random
import geopy
import pandas as pd
import gradio as gr

from show_homes import PROJECT_DIR
from show_homes.config import config
from show_homes.pipeline import geo_utils, kepler_maps


prop_type_dict = {
    "Flat": (
        ["Flat", "Maisonette"],
        [
            "Semi-Detached",
            "Mid-Terrace",
            "Detached",
            "End-Terrace",
            "unknown",
            "Encloded End-Terrace",
            "Enclosed Mid-Terrace",
        ],
    ),
    "Terraced House": (
        ["House", "Bungalow", "Park home"],
        ["Enclosed Mid-Terrace", "Enclosed End-Terrace", "End-Terrace", "Mid-Terrace"],
    ),
    "Detached House": (["House", "Bungalow", "Park home"], ["Detached"]),
    "Semi-detached House": (["House", "Bungalow", "Park home"], ["Semi-Detached"]),
    # "Any": (["Flat", "Maisonette", "House", "Bungalow", "Park home"], ['Semi-Detached', 'Mid-Terrace', 'Detached', 'End-Terrace',
    #         'unknown', 'Encloded End-Terrace', 'Enclosed Mid-Terrace'])
}


def get_network_homes_for_prop_type(df, property_type):
    """Filter data by property type and return a filtered set
    for show homes and visitor homes.

    Args:
        df (pd.DataFrame): EPC records.
        property_type (str): Property type to filter for.

    Raises:
        NotImplementedError: If given unknown property type.

    Returns:
        tuple: Filtered visitor homes, show homes and similar show homes
    """

    # print(df['PROPERTY_TYPE'].value_counts())
    # print(df['BUILT_FORM'].value_counts())

    # prop_vis_hdfomes = visitor_homes[(visitor_homes['PROPERTY_TYPE'] == prop_type_dict[prop_type][0]) &visitor_homes['BUILT_FORM'].isin(prop_type_dict[prop_type][1])]

    if property_type == "Any":
        filter_cond = ~df["LATITUDE"].isna()
    else:
        filter_cond = df["PROPERTY_TYPE"].isin(prop_type_dict[property_type][0]) & (
            df["BUILT_FORM"].isin(prop_type_dict[property_type][1])
        )

    # if property_type == "Any":
    #     filter_cond = ~df["LATITUDE"].isna()

    # elif property_type == "Flat":
    #     filter_cond = df["PROPERTY_TYPE"] == "Flat"

    # elif property_type == "Terraced House":

    #     filter_cond = (df["PROPERTY_TYPE"] == "House") & (
    #         df["BUILT_FORM"].isin(
    #             [
    #                 "Enclosed Mid-Terrace",
    #                 "Enclosed End-Terrace",
    #                 "End-Terrace",
    #                 "Mid-Terrace",
    #             ]
    #         )
    #     )

    # elif property_type == "Detached House":
    #     filter_cond = (df["PROPERTY_TYPE"] == "House") & (
    #         df["BUILT_FORM"] == "Detached"
    #     )

    # elif property_type == "Semi-detached House":
    #     filter_cond = (df["PROPERTY_TYPE"] == "House") & (
    #         df["BUILT_FORM"] == "Semi-Detached"
    #     )
    # else:
    #     raise NotImplementedError(
    #         "'{}' is not a valid property type.".format(property_type)
    #     )

    # Exclude vistitor homes with unrealistic EPC and within high deprivation areas
    visitor_cond = (
        (df["TENURE"] == "owner-occupied")
        & (df["CURRENT_ENERGY_RATING"].isin(["E", "D", "C"]))
        & (df["IMD Decile"] >= 5)
        & (~df["HP_INSTALLED"])
    )

    visitor_homes = df.loc[~df["HP_INSTALLED"] & filter_cond & visitor_cond]
    host_homes = df.loc[df["HP_INSTALLED"]]
    host_homes_similar = df.loc[df["HP_INSTALLED"] & filter_cond]

    return visitor_homes, host_homes, host_homes_similar


def get_samples(
    visitor_homes,
    host_homes,
    host_ratio,
    visitor_ratio,
    local_auth,
    d_max,
    verbose=False,
):
    """Randomly select samples for visitor homes host/show homes, considering the constraint
    of local authority if necessary.

    Args:
        visitor_homes (df.DatFrame): Visitor homes.
        host_homes (df.DatFrame): Host/show homes.
        host_ratio (int): Ratio for how many show homes to consider.
        visitor_ratio (int): Ratio for how many visitor homes to consider.
        local_auth (str): Local authority to filter by.
        d_max (int): Maximum acceptable driving distance for visitors.
        verbose (bool, optional): Whether to print number of houses before and after sampling. Defaults to False.

    Returns:
        tuple: visitor_homes, host_homes, n_visitor_samples, n_host_samples, pre_samp_text, post_samp_text,
    """

    # Only use visitor homes from within LA
    if local_auth != "GB":
        visitor_homes = visitor_homes[
            visitor_homes["LOCAL_AUTHORITY_LABEL"] == local_auth
        ]

    # Compute number of visitor homes and sample randomly
    n_original_visitor_homes = visitor_homes.shape[0]
    n_visitor_samples = int(n_original_visitor_homes * visitor_ratio)
    visitor_homes = visitor_homes.sample(frac=1, random_state=42)[:n_visitor_samples]

    # Get coordinates (we only need Cartesian)
    coords_dict = get_coordinates(visitor_homes, host_homes)

    # Get show homes that are likely within reach (both inside and outside of LA area)
    if local_auth != "GB":

        # Which properties could be within reach (2 * max distance from randomly selected hosue in LA)
        host_tree = spatial.KDTree(coords_dict["host cart"])
        within_reach_idx = host_tree.query_ball_point(
            coords_dict["visitor cart"][0], d_max * 2
        )

        visitor_homes = visitor_homes.sample(frac=1, random_state=42)[
            :n_visitor_samples
        ]
        same_la_idx = np.where(host_homes["LOCAL_AUTHORITY_LABEL"] == local_auth)[
            0
        ].tolist()

        merged_idx = list(set(within_reach_idx + same_la_idx))
        host_homes = host_homes.iloc[merged_idx]

    # Compute number of show homes and sample randomly
    n_original_show_homes = host_homes.shape[0]
    n_host_samples = int(n_original_show_homes * host_ratio)
    host_homes = host_homes.sample(frac=1, random_state=42)[:n_host_samples]

    if verbose:

        print("Before subsampling:")
        print("# Props without HPs: {}".format(n_original_visitor_homes))
        print("# Props with HPs: {}".format(n_original_show_homes))
        print()
        print("After subsampling:")
        print("# Props without HPs: {}".format(visitor_homes.shape[0]))
        print("# Props with HPs: {}".format(host_homes.shape[0]))

    pre_samp_text = (
        "Before subsampling:\n# Props without HPs: {}\n# Props with HPs: {}".format(
            n_original_visitor_homes, n_original_show_homes
        )
    )
    post_samp_text = (
        "After subsampling:\n# Props without HPs: {}\n# Props with HPs: {}".format(
            visitor_homes.shape[0], host_homes.shape[0]
        )
    )

    return (
        visitor_homes,
        host_homes,
        pre_samp_text,
        post_samp_text,
    )


def get_coordinates(visitor_homes, host_homes):

    # Get coordinates (cartesian and original)
    visitor_coords_cart, visitor_coords_org = geo_utils.prepare_coords(visitor_homes)
    host_coords_cart, host_coords_org = geo_utils.prepare_coords(host_homes)

    coords_dict = {
        "visitor cart": visitor_coords_cart,
        "visitor org": visitor_coords_org,
        "host cart": host_coords_cart,
        "host org": host_coords_org,
    }

    # return visitor_coords_cart, visitor_coords_org, host_coords_cart, host_coords_org
    return coords_dict


def get_host_visitor_options(host_vis_match_idx, n_visitor_samples):

    # How many matches each host gets
    host_opts = np.array([[len(x) for x in host_vis_match_idx]])[0]

    # All matches for all hosts in one flattened list
    flattened_match_idx = np.array([x for xs in host_vis_match_idx for x in xs])

    # Which visitor homes are matched, and how often
    matched_vis_home_idx, matched_vis_home_counts = np.unique(
        flattened_match_idx, return_counts=True
    )

    visitor_opts = np.zeros((n_visitor_samples))
    visitor_opts[matched_vis_home_idx] = matched_vis_home_counts

    return host_opts, visitor_opts


def get_host_visitor_matches(coords_dict, d_max, v_max):

    n_visitor_samples = coords_dict["visitor cart"].shape[0]
    n_host_samples = coords_dict["host cart"].shape[0]

    # Host and visitor tree
    host_tree = spatial.KDTree(coords_dict["host cart"])
    visitor_tree = spatial.KDTree(coords_dict["visitor cart"])

    # Nested list: for each host, which visitor idx are matches
    host_vis_match_idx = host_tree.query_ball_tree(visitor_tree, r=d_max)

    # Catch any matches
    if [x for xs in host_vis_match_idx for x in xs]:

        host_opts_pre_cap, visitor_opts_pre_cap = get_host_visitor_options(
            host_vis_match_idx, n_visitor_samples
        )

        # This is sadly the simplest way to randomly sample without triggering an error if there are less than 5 to begin with and without creating a new variable
        capped_match_idx = [
            random.sample(host_matches, len(host_matches))[:v_max]
            for host_matches in host_vis_match_idx
        ]
        host_opts_post_cap, visitor_opts_post_cap = get_host_visitor_options(
            capped_match_idx, n_visitor_samples
        )
        n_valid_hp_props = host_opts_pre_cap.shape[0]

    # If there are no matched, fill in with zeros
    else:
        visitor_opts_pre_cap = np.zeros(n_visitor_samples).astype(int)
        visitor_opts_post_cap = np.zeros(n_visitor_samples).astype(int)

        host_opts_post_cap = np.zeros(n_host_samples).astype(int)
        host_opts_pre_cap = np.zeros(n_host_samples).astype(int)

        n_valid_hp_props = n_host_samples
        capped_match_idx = []

    connections = np.zeros((n_valid_hp_props * v_max, 5))

    counter = 0
    for i in range(host_opts_post_cap.shape[0]):
        m = host_opts_post_cap[i]

        for j in range(m):

            dist = round(
                geopy.distance.distance(
                    coords_dict["host org"][i],
                    coords_dict["visitor org"][capped_match_idx[i][j]],
                ).km,
                1,
            )

            connections[counter, :2] = coords_dict["host org"][i]
            connections[counter, 2:4] = coords_dict["visitor org"][
                capped_match_idx[i][j]
            ]
            connections[counter, 4] = dist

            counter += 1

    # Remove all zero rows
    connections = connections[~np.all(connections == 0, axis=1)]

    host_data = [n_valid_hp_props, host_opts_post_cap, host_opts_pre_cap]
    visitor_data = [n_visitor_samples, visitor_opts_post_cap, visitor_opts_pre_cap]

    host_df, visitor_df, connections_df = get_options_and_connections_as_df(
        connections, coords_dict, host_data, visitor_data
    )

    host_data.append(host_df)
    visitor_data.append(visitor_df)

    return host_data, visitor_data, connections_df


def get_options_and_connections_as_df(
    connections, coords_dict, host_data, visitor_data, save_df=True
):

    n_valid_hp_props, host_opts_post_cap, host_opts_pre_cap = host_data
    n_visitor_samples, visitor_opts_post_cap, visitor_opts_pre_cap = visitor_data

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

    host_data = np.zeros((n_valid_hp_props, 5))
    host_data[:, 0:2] = coords_dict["host org"][:, 0:2]
    host_data[:, 2] = host_opts_post_cap
    host_data[:, 3] = host_opts_pre_cap
    host_data[:, 4] = host_opts_post_cap > 0

    host_df = pd.DataFrame(
        host_data,
        columns=[
            "LATITUDE",
            "LONGITUDE",
            "Visitor matches (capped)",
            "Visitor matches",
            "Matched",
        ],
    )

    visitor_data = np.zeros((n_visitor_samples, 5))
    visitor_data[:, 0:2] = coords_dict["visitor org"][:, 0:2]
    visitor_data[:, 2] = visitor_opts_post_cap
    visitor_data[:, 3] = visitor_opts_pre_cap
    visitor_data[:, 4] = visitor_opts_post_cap > 0

    visitor_df = pd.DataFrame(
        visitor_data,
        columns=[
            "LATITUDE",
            "LONGITUDE",
            "Host matches (capped)",
            "Host matches",
            "Matched",
        ],
    )

    if save_df:
        host_df.to_csv(config.HOST_VIS_CON_OUT_DATA_PATH / "host_df.csv")
        visitor_df.to_csv(config.HOST_VIS_CON_OUT_DATA_PATH / "visitor_df.csv")
        connections_df.to_csv(config.HOST_VIS_CON_OUT_DATA_PATH / "connections_df.csv")

    return host_df, visitor_df, connections_df


def handle_any_to_same_edge_case(visitor_homes, host_homes, d_max, v_max):

    connections = []
    host_dfs = []
    visitor_dfs = []
    network_metrics = {}

    total_n_visitor_samples = 0
    total_n_host_samples = 0

    total_visitor_options = 0
    total_host_options = 0

    total_visitor_matches = 0
    total_host_matches = 0

    over_cap_hosts = 0

    # under construction
    for prop_type in [
        "Flat",
        "Semi-detached House",
        "Detached House",
        "Terraced House",
    ]:

        print(prop_type)

        prop_vis_homes = visitor_homes[
            (visitor_homes["PROPERTY_TYPE"].isin(prop_type_dict[prop_type][0]))
            & visitor_homes["BUILT_FORM"].isin(prop_type_dict[prop_type][1])
        ]

        prop_host_homes = host_homes[
            (host_homes["PROPERTY_TYPE"].isin(prop_type_dict[prop_type][0]))
            & host_homes["BUILT_FORM"].isin(prop_type_dict[prop_type][1])
        ]

        print(prop_host_homes.shape)
        print(prop_vis_homes.shape)

        # prop_vis_homes = visitor_homes[visitor_homes['PROPERTY_TYPE'] == prop_type]
        # prop_host_homes = host_homes[host_homes['PROPERTY_TYPE'] == prop_type]

        if prop_vis_homes.shape[0] == 0 or prop_host_homes.shape[0] == 0:
            total_n_visitor_samples = prop_vis_homes.shape[0]
            total_n_host_samples = prop_host_homes.shape[0]
            print("nothing found")
            continue

        # Get coordinates (cartesian and original)
        coords_dict = get_coordinates(prop_vis_homes, prop_host_homes)

        host_data, visitor_data, connections_df = get_host_visitor_matches(
            coords_dict, d_max, v_max
        )

        network_metrics = compute_network_metrics(
            host_data, visitor_data, v_max, verbose=False
        )

        connections.append(connections_df)
        host_dfs.append(host_data[-1])
        visitor_dfs.append(visitor_data[-1])

        total_n_visitor_samples += network_metrics["visitor samples"]
        total_n_host_samples += network_metrics["host samples"]

        total_visitor_matches += network_metrics["visitor matches"]
        total_host_matches += network_metrics["host matches"]

        total_visitor_options += network_metrics["visitor options summed"]
        total_host_options += network_metrics["host options summed"]

        over_cap_hosts += network_metrics["over cap hosts"]

    if connections:
        connections_df = pd.concat(connections)
        host_df = pd.concat(host_dfs)
        visitor_df = pd.concat(visitor_dfs)

        network_metrics["visitor samples"] = total_n_visitor_samples
        network_metrics["host samples"] = total_n_host_samples
        network_metrics["capacity visitor"] = (
            total_visitor_options / total_n_visitor_samples
        )
        network_metrics["capacity host"] = total_host_options / total_n_host_samples
        network_metrics["coverage visitor"] = (
            total_visitor_options / total_n_visitor_samples
        )
        network_metrics["coverage host"] = total_host_options / total_n_host_samples
        network_metrics["over cap ratio"] = over_cap_hosts / total_n_host_samples * 100

    else:
        connections_df = pd.DataFrame(
            {},
            columns=[
                "LATITUDE source",
                "LONGITUDE source",
                "LATITUDE target",
                "LONGITUDE target",
                "Distance",
            ],
        )

        host_df = pd.DataFrame(
            {},
            columns=[
                "LATITUDE",
                "LONGITUDE",
                "Vistor matches (capped)",
                "Vistor matches",
                "Matched",
            ],
        )
        visitor_df = pd.DataFrame(
            {},
            columns=[
                "LATITUDE",
                "LONGITUDE",
                "Host matches (capped)",
                "Host matches",
                "Matched",
            ],
        )

        network_metrics["capacity visitor"] = 0
        network_metrics["capacity host"] = 0
        network_metrics["coverage visitor"] = 0
        network_metrics["coverage host"] = 0
        network_metrics["over cap ratio"] = 0

    return host_df, visitor_df, connections_df, network_metrics


def compute_network_metrics(host_data, visitor_data, v_max, verbose=False):

    n_valid_hp_props, host_opts_post_cap, host_opts_pre_cap, _ = host_data
    n_visitor_samples, visitor_opts_post_cap, _, _ = visitor_data

    # print(n_visitor_samples)
    # print(n_valid_hp_props)

    over_cap_ratio = (
        np.count_nonzero(host_opts_pre_cap >= v_max) / n_valid_hp_props * 100
    )

    host_matches = host_opts_post_cap > 0
    visitor_matches = visitor_opts_post_cap > 0
    capacity_host = np.sum(host_opts_post_cap) / n_valid_hp_props
    capacity_visitor = np.sum(visitor_opts_post_cap) / n_visitor_samples
    coverage_host = host_matches.sum() / host_matches.shape[0]
    coverage_visitor = visitor_matches.sum() / visitor_matches.shape[0]

    # print('visitor samples', n_visitor_samples)
    # print('valid host props', n_valid_hp_props)
    # print('visitors', visitor_matches.shape[0])
    # print('hosts', host_matches.shape[0])

    capacity_host = round(capacity_host, 2)
    capacity_visitor = round(capacity_visitor, 2)
    coverage_host = round(coverage_host * 100, 2)
    coverage_visitor = round(coverage_visitor * 100, 2)
    over_cap_ratio = round(over_cap_ratio)

    if verbose:
        print()
        print("Results")
        print("=========")
        print("Host capacity:\t {}".format(capacity_host))
        print("Visitor capacity:\t {}".format(capacity_visitor))
        print("Visitor coverage: {}%".format(coverage_visitor))
        print("Host coverage: {}%".format(coverage_host))
        print("Over cap ratio: {}%".format(over_cap_ratio))

    return {
        "capacity host": capacity_host,
        "capacity visitor": capacity_visitor,
        "coverage host": coverage_host,
        "coverage visitor": coverage_visitor,
        "over cap ratio": over_cap_ratio,
        "visitor matches": visitor_matches.sum(),
        "host matches": host_matches.sum(),
        "visitor options summed": np.sum(visitor_opts_post_cap),
        "host options summed": np.sum(host_opts_post_cap),
        "visitor samples": n_visitor_samples,
        "host samples": n_valid_hp_props,
        "over cap hosts": np.count_nonzero(host_opts_pre_cap >= v_max),
    }


def prepare_textual_output(
    local_auth, property_type, network_metrics, pre_samp_text, post_samp_text
):

    # Prepare text
    local_auth_output = " in " + local_auth if local_auth != "GB" else " in GB"
    property_type = property_type if property_type != "Any" else "any propertie"

    output = "Network for {}s{}\n=========\nAverage visitor matches for show homes:\t{}\nAverage host matches for visitor homes:\t\t{}\nHost Coverage:\t\t{}%\nVisitor Coverage:\t{}%\nOver cap ratio:\t\t\t{}%".format(
        property_type,
        local_auth_output,
        network_metrics["capacity host"],
        network_metrics["capacity visitor"],
        network_metrics["coverage host"],
        network_metrics["coverage visitor"],
        network_metrics["over cap ratio"],
    )

    text_output = pre_samp_text + "\n\n" + post_samp_text + "\n\n" + output
    return text_output


def compute_network_measure(
    df,
    property_type,
    same_prop_type,
    host_ratio,
    visitor_ratio,
    v_max,
    n_open_days,
    d_max,
    local_auth,
    verbose=True,
):

    v_max = int(v_max * n_open_days)
    host_ratio = host_ratio / 100
    visitor_ratio = visitor_ratio / 100

    version = "similar" if same_prop_type else "any"

    settings_string = "{}_{}_{}_{}_{}_{}_{}".format(
        property_type,
        str(same_prop_type),
        str(host_ratio),
        str(visitor_ratio),
        str(v_max),
        str(d_max),
        local_auth,
    )

    if verbose:
        print("Settings\n************")
        print("Area:", local_auth)
        print("Property type:", property_type)
        print("Visitor ratio:", visitor_ratio)
        print("Host ratio:", host_ratio)
        print("Maximum visitors:", v_max)
        print("Maximum distance:", d_max)
        print("Same property type or not:", str(same_prop_type))
        print()

    visitor_homes, host_homes, host_homes_similar = get_network_homes_for_prop_type(
        df, property_type
    )

    host_homes = host_homes if version == "any" else host_homes_similar

    # # Sample visitor and host homes
    sample_outputs = get_samples(
        visitor_homes,
        host_homes,
        host_ratio,
        visitor_ratio,
        local_auth,
        d_max,
        verbose=True,
    )

    visitor_homes, host_homes, pre_samp_text, post_samp_text = sample_outputs

    if property_type == "Any" and same_prop_type:

        (
            host_df,
            visitor_df,
            connections_df,
            network_metrics,
        ) = handle_any_to_same_edge_case(visitor_homes, host_homes, d_max, v_max)

    else:
        # Get coordinates (cartesian and original)
        coords_dict = get_coordinates(visitor_homes, host_homes)

        host_data, visitor_data, connections_df = get_host_visitor_matches(
            coords_dict, d_max, v_max
        )

        network_metrics = compute_network_metrics(
            host_data, visitor_data, v_max, verbose=True
        )

        host_df = host_data[-1]
        visitor_df = visitor_data[-1]

    text_output = prepare_textual_output(
        local_auth, property_type, network_metrics, pre_samp_text, post_samp_text
    )

    kepler_maps.create_output_map(connections_df, host_df, visitor_df, settings_string)
    kepler_map = '<iframe src="file/maps/Generated_network_map_{}.html" style="border:0px #ffffff none;" name="myiFrame" scrolling="no" frameborder="1" marginheight="0px" marginwidth="0px" height="600px" width="800px" allowfullscreen></iframe>'.format(
        settings_string
    )

    return text_output, kepler_map


def main():

    df = pd.read_csv(config.PROJECT_DIR / "epc_for_show_homes.csv")
    # show_homes_network.compute_network_measure(df, 'Detached House', True, 1,5, 5, 6, 30, 'GB', verbose=True)

    def G(
        property_type,
        same_prop_type,
        host_ratio,
        visitor_ratio,
        v_max,
        n_open_days,
        d_max,
        local_auth,
        verbose=True,
    ):

        return compute_network_measure(
            df,
            property_type,
            same_prop_type,
            host_ratio,
            visitor_ratio,
            v_max,
            n_open_days,
            d_max,
            local_auth,
            verbose=True,
        )

    property_types = [
        "Flat",
        "Semi-detached house",
        "Detached House",
        "Terraced House",
        "Any",
    ]
    local_authorities = sorted(list(df["LOCAL_AUTHORITY_LABEL"].unique())) + ["GB"]

    demo = gr.Interface(
        fn=G,
        inputs=[
            gr.inputs.Radio(
                property_types, label="Property Type", default="Detached House"
            ),
            gr.inputs.Radio(
                [True, False], label="Show home of same property", default=True
            ),
            gr.inputs.Slider(0, 100, default=1, step=1, label="Host ratio (%)"),
            gr.inputs.Slider(0, 100, default=5, step=1, label="Visitor ratio (%)"),
            gr.inputs.Slider(1, 50, default=5, step=1, label="Max visitors"),
            gr.inputs.Slider(
                1, 50, default=6, step=1, label="Number of slots/open days"
            ),
            gr.inputs.Slider(1, 50, default=35, step=1, label="Max distance"),
            gr.inputs.Dropdown(
                local_authorities, default="Manchester", label="Local authorities"
            ),
        ],
        title="Network of Show Homes",
        outputs=["text", "html"],
    )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
