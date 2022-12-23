LOCAL_DATA_DIR = "/Users/juliasuter/Documents/ASF_data"

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
    data_path=LOCAL_DATA_DIR,
    batch="newest",
    usecols=EPC_FEAT_SELECTION,
    version="preprocessed",
)


epc_df_with_MCS = merge_install_dates.manage_hp_install_dates(epc_df)

epc_df_with_MCS.to_csv("epc_df_with_mcs.csv")
epc_df_with_MCS.head()


imd_df = imd_data.get_gb_imd_data(data_path=LOCAL_DATA_DIR)

epc_df = imd_data.merge_imd_with_other_set(
    imd_df, epc_df_with_MCS, postcode_label="Postcode"
)

coord_df = coordinates.get_postcode_coordinates(data_path=LOCAL_DATA_DIR)
coord_df.head()
coord_df["POSTCODE"] = coord_df["POSTCODE"].str.replace(" ", "")

epc_df = epc_df.rename(columns={"Postcode": "POSTCODE"})
# epc_df.drop(columns=['LONGITUDE', 'LATITUDE'], inplace=True)
epc_df = pd.merge(epc_df, coord_df, on=["POSTCODE"], how="left")
print(epc_df.shape)

epc_df.to_csv("epc_for_show_homes.csv")
