import numpy as np
from scipy import spatial
import random
import geopy
import pandas as pd
import gradio as gr


from show_homes.pipeline import geo_utils


def filter_by_property_type(df, property_type):

    no_cond = ~df["LATITUDE"].isna()
    flat = df["PROPERTY_TYPE"] == "Flat"
    terraced = [
        "Enclosed Mid-Terrace",
        "Enclosed End-Terrace",
        "End-Terrace",
        "Mid-Terrace",
    ]
    terraced_house = (df["PROPERTY_TYPE"] == "House") & (
        df["BUILT_FORM"].isin(terraced)
    )
    detached_house = (df["PROPERTY_TYPE"] == "House") & (df["BUILT_FORM"] == "Detached")
    semi_house = (df["PROPERTY_TYPE"] == "House") & (
        df["BUILT_FORM"] == "Semi-Detached"
    )

    all_conds = [flat, terraced_house, detached_house, semi_house]

    visitor_cond = (
        (df["TENURE"] == "owner-occupied")
        & (df["CURRENT_ENERGY_RATING"].isin(["E", "D", "C"]))
        & (df["IMD Decile"] >= 5)
        & (~df["HP_INSTALLED"])
    )

    cond_dict = {
        "Flat": flat,
        "Semi-detached house": semi_house,
        "Detached House": detached_house,
        "Terraced House": terraced_house,
        "Any": no_cond,
    }

    conds = cond_dict[property_type]

    visitor_homes = df.loc[~df["HP_INSTALLED"] & conds & visitor_cond]
    host_homes = df.loc[df["HP_INSTALLED"]]
    host_homes_similar = df.loc[df["HP_INSTALLED"] & conds]

    return visitor_homes, host_homes, host_homes_similar, all_conds


def get_samples(
    visitor_homes, host_homes, host_ratio, visitor_ratio, local_auth, version, d_max
):

    # Sample visitor and host homes
    if local_auth != "GB":
        visitor_homes = visitor_homes[
            visitor_homes["LOCAL_AUTHORITY_LABEL"] == local_auth
        ]
    n_visitor_samples = int(visitor_homes.shape[0] * visitor_ratio)
    n_original_visitor_homes = visitor_homes.shape[0]
    visitor_homes = visitor_homes.sample(frac=1, random_state=42)[:n_visitor_samples]

    # Get coordinates (cartesian and original)
    visitor_coords_cart, _, host_coords_cart, _ = get_coordinates(
        visitor_homes, host_homes
    )

    if local_auth != "GB":

        # Host tree
        host_tree = spatial.KDTree(host_coords_cart)
        idx_1 = host_tree.query_ball_point(visitor_coords_cart[0], d_max * 2)

        visitor_homes = visitor_homes.sample(frac=1, random_state=42)[
            :n_visitor_samples
        ]
        idx_2 = np.where(host_homes["LOCAL_AUTHORITY_LABEL"] == local_auth)[0].tolist()

        merged_idx = list(set(idx_1 + idx_2))
        host_homes = host_homes.iloc[merged_idx]

    n_original_show_homes = host_homes.shape[0]
    n_host_samples = int(n_original_show_homes * host_ratio)
    host_homes = host_homes.sample(frac=1, random_state=42)[:n_host_samples]

    print("After subsampling:")
    print("# Props without HPs: {}".format(n_original_visitor_homes))
    print("# Props with HPs: {}".format(n_original_show_homes))

    print("Before subsampling:")
    print("# Props without HPs: {}".format(visitor_homes.shape[0]))
    print("# Props with HPs: {}".format(host_homes.shape[0]))

    before_text = (
        "Before subsampling:\n# Props without HPs: {}\n# Props with HPs: {}".format(
            n_original_visitor_homes, n_original_show_homes
        )
    )
    after_text = (
        "After subsampling:\n# Props without HPs: {}\n# Props with HPs: {}".format(
            visitor_homes.shape[0], host_homes.shape[0]
        )
    )

    return (
        visitor_homes,
        host_homes,
        n_visitor_samples,
        n_host_samples,
        before_text,
        after_text,
    )


def get_coordinates(visitor_homes, host_homes):

    print(visitor_homes.shape)

    # Get coordinates (cartesian and original)
    visitor_coords_cart, visitor_coords_org = geo_utils.prepare_coords(visitor_homes)

    host_coords_cart, host_coords_org = geo_utils.prepare_coords(host_homes)

    # coords_dict = {'visitor_cartesian': visitor_coords_cart,
    #                'visitor_org_coords' : visitor_coords_org,
    #                'host_cartesian': host_coords_cart,
    #                'host_org_coords': host_coords_org }

    return visitor_coords_cart, visitor_coords_org, host_coords_cart, host_coords_org


def get_host_visitor_matches(host_vis_match_idx, n_visitor_samples):

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

    if verbose:
        print(local_auth)
        print(visitor_ratio)
        print(property_type)
        print(host_ratio)
        print(v_max)
        print(d_max)
        print(property_type)
        print(str(same_prop_type))

    settings_string = "{}_{}_{}_{}_{}_{}_{}".format(
        property_type,
        str(same_prop_type),
        str(host_ratio),
        str(visitor_ratio),
        str(v_max),
        str(d_max),
        local_auth,
    )

    visitor_homes, host_homes, host_homes_similar, all_conds = filter_by_property_type(
        df, property_type
    )

    host_homes = host_homes if version == "any" else host_homes_similar

    # Visitors only for given local authority
    # if local_auth != 'GB':
    #     visitor_homes = visitor_homes.loc[visitor_homes['LOCAL_AUTHORITY_LABEL'] == local_auth]
    # non_hp_samples = non_hp_samples.loc[non_hp_samples['LOCAL_AUTHORITY_LABEL'].isin([local_auth])]
    # hp_samples = non_hp_samples.loc[hp_samples['LOCAL_AUTHORITY_LABEL'].isin(local_auth)]
    # hp_same_type_samples = hp_same_type_samples.loc[hp_samples['LOCAL_AUTHORITY_LABEL'].isin(local_auth)]

    # print("Before subsampling:")
    # print("# Props without HPs: {}".format(visitor_homes.shape[0]))
    # print("# Props with HPs: {}".format(host_homes.shape[0]))
    # print("# Similar props with with HPs: {}".format(host_homes_similar.shape[0]))

    # before_text =  'Overall situation:\n# Props without HPs: {}\n# Props with HPs: {}\n# Similar props with with HPs: {}'.format(visitor_homes.shape[0], host_homes.shape[0], host_homes_similar.shape[0])

    # # Sample visitor and host homes
    (
        visitor_homes,
        host_homes,
        n_visitor_samples,
        n_host_samples,
        before_text,
        after_text,
    ) = get_samples(
        visitor_homes, host_homes, host_ratio, visitor_ratio, local_auth, version, d_max
    )

    # print()
    # print('After subsampling:')
    # print('# Props without HPs: {}'.format(visitor_homes.shape[0]))
    # print('# Props with HPs: {}'.format(host_homes.shape[0]))
    # print('# Similar props with HPs: {}'.format(host_homes_similar.shape[0]))

    # after = 'After subsampling:\n# Props without HPs: {}\n# Props with HPs: {}\n# Similar props with HPs: {}'.format(visitor_homes.shape[0], host_homes.shape[0],                                                                                              host_homes_similar.shape[0])

    # Get coordinates (cartesian and original)
    (
        visitor_coords_cart,
        visitor_coords_org,
        host_coords_cart,
        host_coords_org,
    ) = get_coordinates(visitor_homes, host_homes)

    # Host and visitor tree
    host_tree = spatial.KDTree(host_coords_cart)
    visitor_tree = spatial.KDTree(visitor_coords_cart)

    # Nested list: for each host, which visitor idx are matches
    host_vis_match_idx = host_tree.query_ball_tree(visitor_tree, r=d_max)

    # Catch any matches
    if [x for xs in host_vis_match_idx for x in xs]:

        # This is sadly the simplest way to randomly sample without triggering an error if there are less than 5 to begin with
        # and without creating a new variable
        host_opts_pre_cap, visitor_opts_pre_cap = get_host_visitor_matches(
            host_vis_match_idx, n_visitor_samples
        )

        capped_match_idx = [
            random.sample(host_matches, len(host_matches))[:v_max]
            for host_matches in host_vis_match_idx
        ]
        host_opts_post_cap, visitor_opts_post_cap = get_host_visitor_matches(
            capped_match_idx, n_visitor_samples
        )
        n_valid_hp_props = host_opts_pre_cap.shape[0]

        # if local_auth == 'GB' and same_prop_type:

        #     for prop_type in all_conds:
        #         idx = show_homes[prop_type]
        #     np.where(host_homes['LOCAL_AUTHORITY_LABEL'] == local_auth)[0].tolist()

        # if local_auth != 'GB':

        #     if version == 'any':
        #         out_of_la = np.array(~(host_homes['LOCAL_AUTHORITY_LABEL'] == local_auth))
        #     else:
        #         out_of_la = np.array(~(host_homes_similar['LOCAL_AUTHORITY_LABEL']== local_auth))

        #     la_hosts = ~out_of_la
        #     matched_hosts = host_opts_post_cap > 0
        #     n_valid_hp_props = (la_hosts| matched_hosts).sum() # only counts props within LA or with match
        #     within_la_or_matched = (la_hosts| matched_hosts)

    else:
        visitor_opts_pre_cap = np.zeros(n_visitor_samples).astype(int)
        visitor_opts_post_cap = np.zeros(n_visitor_samples).astype(int)

        host_opts_post_cap = np.zeros(n_host_samples).astype(int)
        host_opts_pre_cap = np.zeros(n_host_samples).astype(int)

        n_valid_hp_props = n_host_samples
        capped_match_idx = []

    over_cap_ratio = (
        np.count_nonzero(host_opts_pre_cap >= v_max) / n_valid_hp_props * 100
    )
    connections = np.zeros((n_valid_hp_props * v_max, 5))

    counter = 0

    for i in range(host_opts_post_cap.shape[0]):

        m = host_opts_post_cap[i]

        # if m == 0:
        #     continue

        for j in range(m):

            dist = round(
                geopy.distance.distance(
                    host_coords_org[i], visitor_coords_org[capped_match_idx[i][j]]
                ).km,
                1,
            )

            connections[counter, :2] = host_coords_org[i]
            connections[counter, 2:4] = visitor_coords_org[capped_match_idx[i][j]]
            connections[counter, 4] = dist

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

    # if local_auth != 'GB':
    #     # only consider hosts that were matched or are within LA
    #     host_coords_org = host_coords_org[within_la_or_matched, :]
    #     host_opts_pre_cap = host_opts_pre_cap[within_la_or_matched]
    #     host_opts_post_cap = host_opts_post_cap[within_la_or_matched]

    host_data = np.zeros((n_valid_hp_props, 5))
    host_data[:, 0:2] = host_coords_org[:, 0:2]
    host_data[:, 2] = host_opts_post_cap
    host_data[:, 3] = host_opts_pre_cap
    host_data[:, 4] = host_opts_post_cap > 0

    host_data_df = pd.DataFrame(
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
    visitor_data[:, 0:2] = visitor_coords_org[:, 0:2]
    visitor_data[:, 2] = visitor_opts_post_cap
    visitor_data[:, 3] = visitor_opts_pre_cap
    visitor_data[:, 4] = visitor_opts_post_cap > 0

    visitor_data_df = pd.DataFrame(
        visitor_data,
        columns=[
            "LATITUDE",
            "LONGITUDE",
            "Host matches (capped)",
            "Host matches",
            "Matched",
        ],
    )

    visitor_data_df.to_csv("visitor_df.csv")
    host_data_df.to_csv("host_df.csv")
    connections_df.to_csv("connections_df.csv")

    visitor_matches = visitor_opts_post_cap > 0
    host_matches = host_opts_post_cap > 0
    capacity_host = np.sum(host_opts_post_cap) / n_valid_hp_props
    capacity_visitor = np.sum(visitor_opts_post_cap) / n_visitor_samples
    coverage_visitor = visitor_matches.sum() / visitor_matches.shape[0]
    coverage_host = host_matches.sum() / host_matches.shape[0]

    capacity_host = round(capacity_host, 2)
    capacity_visitor = round(capacity_visitor, 2)
    coverage_visitor = round(coverage_visitor * 100, 2)
    coverage_host = round(coverage_host * 100, 2)
    over_cap_ratio = round(over_cap_ratio)

    print()
    print("Results {}".format(property_type))
    print("=========")
    print("Host capacity:\t {}".format(capacity_host))
    print("Visitor capacity:\t {}".format(capacity_visitor))
    print("Visitor coverage: {}%".format(coverage_visitor))
    print("Host coverage: {}%".format(coverage_host))
    print("Over cap ratio: {}%".format(over_cap_ratio))

    local_auth_output = " in " + local_auth if local_auth != "GB" else " in GB"

    property_type = property_type if property_type != "Any" else "any propertie"

    output = "Network for {}s{}\n=========\nAverage visitor matches for show homes:\t{}\nAverage host matches for visitor homes:\t{}\n\nVisitor Coverage:\t{}%\nHost Coverage:\t{}%\nOver cap ratio:\t{}%".format(
        property_type,
        local_auth_output,
        capacity_host,
        capacity_visitor,
        coverage_visitor,
        coverage_host,
        over_cap_ratio,
    )

    output = before_text + "\n\n" + after_text + "\n\n" + output
    print(output)

    geo_utils.create_output_map(
        connections_df, host_data_df, visitor_data_df, settings_string
    )
    kepler_map = '<iframe src="file/maps/Generated_network_map_{}.html" style="border:0px #ffffff none;" name="myiFrame" scrolling="no" frameborder="1" marginheight="0px" marginwidth="0px" height="600px" width="800px" allowfullscreen></iframe>'.format(
        settings_string
    )

    return output, kepler_map


def main():

    df = pd.read_csv("../analysis/epc_for_show_homes.csv")
    # df = pd.read_csv('analysis/orkney.csv')
    df = df[df["LOCAL_AUTHORITY_LABEL"] == "Orkney Islands"]

    # df.to_csv('analysis/orkney.csv')
    df = df[df["LATITUDE"].notna()]
    df = df[df["LONGITUDE"].notna()]

    property_types = [
        "Flat",
        "Semi-detached house",
        "Detached House",
        "Terraced House",
        "Any",
    ]
    compute_network_measure(
        df, "Detached House", True, 1, 5, 5, 1, 30, "Orkney Islands", verbose=True
    )

    # local_authorities = list(df['LOCAL_AUTHORITY_LABEL'].unique())
    # #local_authorities = ['Manchester', 'Glasgow City', 'Orkney Islands', 'Edinburgh', 'GB']

    # demo = gr.Interface(
    #     fn=compute_network_measure,
    #     inputs=[default=df, gr.inputs.Radio(property_types, label='Property Type', default='Detached House'),
    #             gr.inputs.Radio([True, False], label='Show home of same property', default=True),
    #             gr.inputs.Slider(0, 100, default=1, step=1, label='Host ratio (%)'),
    #             gr.inputs.Slider(0, 100, default=5, step=1, label='Visitor ratio (%)'),
    #             gr.inputs.Slider(1, 50, default=5, step=1, label='Max visitors'),
    #             gr.inputs.Slider(1, 50, default=6, step=1, label='Number of slots/open days'),
    #             gr.inputs.Slider(1, 50, default=35, step=1, label='Max distance'),
    #             gr.inputs.Dropdown(local_authorities, default='Manchester', label='Local authorities')],
    #     outputs=["text", "html"],
    #     title='Network of Show Homes',)

    # demo.launch(share=True)


if __name__ == "__main__":
    main()
