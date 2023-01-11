import pandas as pd

from asf_core_data.getters.epc import epc_data
from asf_core_data.pipeline.data_joining import merge_install_dates
from asf_core_data.getters.supplementary_data.deprivation import imd_data
from asf_core_data.getters.supplementary_data.geospatial import coordinates

from show_homes import config


def generate_data_for_show_homes_network(data_path="S3"):

    EPC_FEAT_SELECTION = [
        "ADDRESS1",
        "ADDRESS2",
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
        "HP_INSTALLED",
        "UPRN",
    ]

    epc_df = epc_data.load_preprocessed_epc_data(
        data_path=data_path,  # also works with data_path = 'S3', but takes much longer
        batch="newest",
        usecols=EPC_FEAT_SELECTION,
        version="preprocessed",
    )

    # Complete with MCS installation dates
    epc_df_with_MCS = merge_install_dates.manage_hp_install_dates(epc_df)
    epc_df_with_MCS.to_csv("epc_df_with_mcs.csv")

    # Add IMD data
    imd_df = imd_data.get_gb_imd_data()

    epc_df = imd_data.merge_imd_with_other_set(
        imd_df, epc_df_with_MCS, postcode_label="POSTCODE"
    )

    # Add coordinates
    coord_df = coordinates.get_postcode_coordinates()
    coord_df["POSTCODE"] = coord_df["POSTCODE"].str.replace(" ", "")
    epc_df = pd.merge(epc_df, coord_df, on=["POSTCODE"], how="left")

    epc_df.to_csv("epc_for_show_homes.csv")


if __name__ == "__main__":

    LOCAL_DATA_DIR = config["LOCAL_DATA_DIR"]
    generate_data_for_show_homes_network(data_path=LOCAL_DATA_DIR)
