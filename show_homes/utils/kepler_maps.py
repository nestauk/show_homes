# File: show_homes/utils/kepler_maps.py
"""
Create Kepler map for show home models.

Project: Show homes
Author: Julia Suter
"""

import yaml
import pandas as pd
from keplergl import KeplerGl

from show_homes.config import config


def add_fake_hosts_and_visitors(visitor_df, host_df):
    """Add two fake hosts and visitors to map (located in Australia)
    to account for the edge case of all hosts or all visitors being
    matched/unmatched, resulting in inconsistent color assingments to
    'yes' and 'no' categories in the Kepler maps.

    Args:
        visitor_df (pandas.DataFrame): All visitors.
        host_df (pandas.DataFrame): All hosts.

    Returns:
        visitor_df (pandas.DataFrame): Visitors + 2 fake entries.
        host_df (pandas.DataFrame): Hosts + 2 fake entries.
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

    return visitor_df, host_df


def create_output_map(
    connections_df,
    visitor_df,
    host_df,
    settings_string,
    map_name=config.GRADIO_OUT_MAP_NAME,
):
    """Create an output map for the modelled show home network.

    Args:
        connections_df (pd.DataFrame): Data about connections between visitor and host homes.
        visitor_df (pd.DataFrame): Visitor homes.
        host_df (pd.DataFrame): Host homes.
        settings_string (str): String describing settings/parameters.
    """

    with open(config.GRADIO_KEPLER_CONFIG, "r") as infile:
        kepler_config = infile.read()
        kepler_config = yaml.load(kepler_config, Loader=yaml.FullLoader)

    # Map Booleans/ints to categories for better readability
    map_dict = {0.0: "no", 1.0: "yes"}
    visitor_df["Matched"] = visitor_df["Matched"].map(map_dict)
    host_df["Matched"] = host_df["Matched"].map(map_dict)

    # Add fake coordinates in Australia to make sure all categories are included.
    # Otherwise, color map might not be consistent across different exampels.
    visitor_df, host_df = add_fake_hosts_and_visitors(visitor_df, host_df)

    network_map = KeplerGl(height=500, config=kepler_config)

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

    map_name = map_name.format(settings_string)
    network_map.save_to_html(file_name=config.GRADIO_OUT_MAPS_PATH / map_name)
