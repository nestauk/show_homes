import yaml

import pandas as pd
from keplergl import KeplerGl

from show_homes import PROJECT_DIR

MODEL_MAP_OUTPUT_PATH = PROJECT_DIR / "show_homes/analysis/maps/"
KEPLER_MODEL_NET_CONFIG = PROJECT_DIR / "outputs/maps/config/network_gradio_config.txt"


def add_fake_hosts_and_visitors(host_df, visitor_df):
    """Add two fake hosts and visitors to map (located in Australia)
    to account for the edge case of all hosts or all visitors being
    matched/unmatched, resulting in inconsistent color assingments to
    'yes' and 'no' categories in the Kepler maps.

    Args:
        host_df (pandas.DataFrame): All hosts.
        visitor_df (pandas.DataFrame): All visitors.

    Returns:
        host_df (pandas.DataFrame): Hosts + 2 fake entries.
        visitor_df (pandas.DataFrame): Visitors + 2 fake entries.
    """

    fake_host_1 = pd.DataFrame(
        {
            "LATITUDE": -25.27729575389078,
            "LONGITUDE": 131.07289492391908,
            "Visitor matches (capped)": 0,
            "Visitor matches": 0,
            "Matched": "no",
        },
        index=[0],
    )

    fake_host_2 = pd.DataFrame(
        {
            "LATITUDE": -27.27729575389078,
            "LONGITUDE": 131.07289492391908,
            "Visitor matches (capped)": 0,
            "Visitor matches": 0,
            "Matched": "yes",
        },
        index=[0],
    )

    fake_visitor_1 = pd.DataFrame(
        {
            "LATITUDE": -25.27729575389078,
            "LONGITUDE": 131.07289492391908,
            "Host matches (capped)": 0,
            "Host matches": 0,
            "Matched": "no",
        },
        index=[0],
    )

    fake_visitor_2 = pd.DataFrame(
        {
            "LATITUDE": -27.27729575389078,
            "LONGITUDE": 131.07289492391908,
            "Host matches (capped)": 0,
            "Host matches": 0,
            "Matched": "yes",
        },
        index=[0],
    )

    host_df = pd.concat([fake_host_1, fake_host_2, host_df]).reset_index(drop=True)
    visitor_df = pd.concat([fake_visitor_1, fake_visitor_2, visitor_df]).reset_index(
        drop=True
    )

    return host_df, visitor_df


def create_output_map(connections_df, host_df, visitor_df, settings_string):

    with open(KEPLER_MODEL_NET_CONFIG, "r") as infile:
        config = infile.read()
        config = yaml.load(config, Loader=yaml.FullLoader)

    # Map Booleans/ints to categories for better readability
    map_dict = {0.0: "no", 1.0: "yes"}
    host_df["Matched"] = host_df["Matched"].map(map_dict)
    visitor_df["Matched"] = visitor_df["Matched"].map(map_dict)

    # Add fake coordinates in Australia to make sure all categories are included.
    # Otherwise, color map might not be consistent across different exampels.
    host_df, visitor_df = add_fake_hosts_and_visitors(host_df, visitor_df)

    network_map = KeplerGl(height=500, config=config)

    network_map.add_data(data=connections_df, name="Network")

    network_map.add_data(
        data=host_df[
            [
                "LATITUDE",
                "LONGITUDE",
                "Visitor matches (capped)",
                "Visitor matches",
                "Matched",
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
                "Matched",
            ]
        ],
        name="Visitor homes",
    )

    map_name = "Generated_network_map_{}.html".format(settings_string)

    network_map.save_to_html(file_name=MODEL_MAP_OUTPUT_PATH / map_name)
