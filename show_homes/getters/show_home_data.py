# File: show_homes/getters/show_home_data.py
"""
Generate and load data relevant for show home project.

Project: Show homes
Author: Julia Suter
"""

import pandas as pd

from asf_core_data.getters.epc import epc_data
from asf_core_data.pipeline.data_joining import merge_install_dates
from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.getters.supplementary_data.geospatial import coordinates
from asf_core_data.getters import data_download

from show_homes.config import config


def generate_data_for_show_homes_network(data_path="S3"):
    """Generate the necessary data for the showhome network.

    Args:
        data_path (str, optional): Path to data directory: local dir or on S3 bucket. Defaults to "S3".
    """

    EPC_FEAT_SELECTION = [
        "POSTCODE",
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
        "TOTAL_FLOOR_AREA",
        "HP_TYPE",
        "UPRN",
    ]

    epc_df = epc_data.load_preprocessed_epc_data(
        data_path=data_path,
        batch="newest",
        usecols=EPC_FEAT_SELECTION,
        version="preprocessed",
    )

    # Complete with MCS installation dates
    epc_df_with_MCS = merge_install_dates.manage_hp_install_dates(epc_df)
    epc_df_with_MCS.to_csv(config.SHOW_HOME_DATA_OUT_DATA_PATH / "epc_df_with_mcs.csv")

    # Add IMD data
    imd_df = imd_data.get_gb_imd_data()

    epc_df = imd_data.merge_imd_with_other_set(
        imd_df, epc_df_with_MCS, postcode_label="POSTCODE"
    )

    # Add coordinates
    coord_df = coordinates.get_postcode_coordinates()
    coord_df["POSTCODE"] = coord_df["POSTCODE"].str.replace(" ", "")
    epc_df = pd.merge(epc_df, coord_df, on=["POSTCODE"], how="left")

    epc_df.loc[
        epc_df["BUILT_FORM"].isin(
            [
                "Enclosed Mid-Terrace",
                "Enclosed End-Terrace",
                "End-Terrace",
                "Mid-Terrace",
            ]
        ),
        "BUILT_FORM",
    ] = "Terraced"
    epc_df = epc_df.loc[~epc_df["LATITUDE"].isna()]

    epc_df.to_csv(config.SHOW_HOME_DATA_OUT_DATA_PATH / "epc_for_show_homes.csv")


def get_show_home_data():
    """Load show home data, either from local directory or download data folder.

    Returns:
        pd.DataFrame: Data relevant for show home model.
    """

    data_file = config.SHOW_HOME_DATA_OUT_DATA_PATH / "epc_for_show_homes.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
    else:
        print("Downloading show homes data...")
        config.SHOW_HOME_DATA_OUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
        data_download.download_s3_folder(
            "inputs/data",
            config.SHOW_HOME_DATA_OUT_DATA_PATH.parent.parent,
            bucket_name="asf-show-homes",
        )

        df = pd.read_csv(data_file)

    return df


if __name__ == "__main__":

    # Also works with data_path = 'S3', but takes much longer (easily over 1h)
    # With local data, takes about 10min
    generate_data_for_show_homes_network(data_path=config.LOCAL_DATA_DIR)
