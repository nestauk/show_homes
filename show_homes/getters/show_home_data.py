# File: show_homes/getters/show_home_data.py
"""
Generate and load data relevant for show home project.

Project: Show homes
Author: Julia Suter
"""

import pandas as pd

from asf_core_data.getters.epc import epc_data
from asf_core_data.pipeline.data_joining import install_date_computation
from asf_core_data.pipeline.preprocessing import feature_engineering
from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.getters.data_getters import data_download, download_core_data, save_to_s3

from show_homes.config import config


def generate_data_for_show_homes_network(data_path="S3", save_locally=True):
    """Generate the necessary data for the showhome network.

    Args:
        data_path (str, optional): Path to data directory: local dir or on S3 bucket. Defaults to "S3".
        save_locally (bool, optional): Save the outputs in the local data dir instead of on the S3 bucket. 
            Make sure to define LOCAL_DATA_DIR in config. Defaults to True.
    """

    epc_df = epc_data.load_preprocessed_epc_data(
        data_path=data_path,
        batch="newest",
        usecols=config.EPC_FEAT_SELECTION,
        version="preprocessed",
    )

    # Complete with MCS installation dates
    epc_df = install_date_computation.compute_hp_install_date(epc_df)

    if save_locally:
        epc_df.to_csv(config.SHOW_HOME_DATA_OUT_DATA_PATH / "epc_df_with_mcs.csv")
    else:
        save_to_s3('asf-show-homes', epc_df, '/inputs/data/epc_df_with_mcs.csv')

    # Add coordinates
    epc_df = feature_engineering.get_postcode_coordinates(epc_df) 

    # Remove records without coordinates
    epc_df = epc_df.loc[~epc_df["LATITUDE"].isna()]

    # Add IMD data
    imd_df = imd_data.get_imd_data()

    epc_df = imd_data.merge_imd_with_other_set(
        imd_df, epc_df, postcode_label="POSTCODE"
    )

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


    if save_locally:
        epc_df.to_csv(config.SHOW_HOME_DATA_OUT_DATA_PATH / "epc_for_show_homes.csv")
    else:
        save_to_s3('asf-show-homes', epc_df, '/inputs/data/epc_for_show_homes.csv')


def get_show_home_data():
    """Load show home data.
    If data is already in local data dir, it will be loaded from there. 
    Otherwise it will be downloaded. 
    Make sure to define LOCAL_DATA_DIR in the config file either way.

    Returns:
        pd.DataFrame: Data relevant for show home model.
    """

    data_file = config.SHOW_HOME_DATA_OUT_DATA_PATH / "epc_for_show_homes.csv"
    if data_file.exists():
        df = pd.read_csv(data_file)
    else:
        print("Downloading show homes data to {}...".format(config.SHOW_HOME_DATA_OUT_DATA_PATH))
        config.SHOW_HOME_DATA_OUT_DATA_PATH.mkdir(parents=True, exist_ok=True)
        data_download.download_s3_folder(
            "inputs/data",
            config.SHOW_HOME_DATA_OUT_DATA_PATH.parent.parent,
            bucket_name="asf-show-homes",
        )

        df = pd.read_csv(data_file)

    return df


if __name__ == "__main__":

    # Example code for generating show homes data

    # Use this to download the most recent EPC batch to your local folder
    # You can comment this out if loading directly from S3
    download_core_data('epc_preprocessed', config.LOCAL_DATA_DIR, batch="newest")

    # Remember to update LOCAL_DATA_DIR in config file
    # - When giving local data dir as data_path, it takes <10min
    # - When giving "S3" as data_path, it can easily take over 1h.
    generate_data_for_show_homes_network(data_path=config.LOCAL_DATA_DIR)
