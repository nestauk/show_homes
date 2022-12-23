# %%
# %load_ext autoreload
# %autoreload 2

import pandas as pd
from keplergl import KeplerGl
import yaml

host_df = pd.read_csv("../pipeline/host_df.csv")
visitor_df = pd.read_csv("../pipeline/visitor_df.csv")
connections_df = pd.read_csv("../pipeline/connections_df.csv")


map_dict = {0.0: "no", 1.0: "yes"}

host_df["Host coverage"] = host_df["Host coverage"].map(map_dict)
visitor_df["Visitor coverage"] = visitor_df["Visitor coverage"].map(map_dict)

host_df.rename(columns={"Host coverage": "Matched"}, inplace=True)
visitor_df.rename(columns={"Visitor coverage": "Matched"}, inplace=True)

print(visitor_df.head())

new_row_1_host = pd.DataFrame(
    {
        "LATITUDE": -25.27729575389078,
        "LONGITUDE": 131.07289492391908,
        "Visitor matches (capped)": 0,
        "Visitor matches": 0,
        "Matched": "no",
    },
    index=[0],
)

new_row_2_host = pd.DataFrame(
    {
        "LATITUDE": -27.27729575389078,
        "LONGITUDE": 131.07289492391908,
        "Visitor matches (capped)": 0,
        "Visitor matches": 0,
        "Matched": "yes",
    },
    index=[0],
)

new_row_1_visitor = pd.DataFrame(
    {
        "LATITUDE": -25.27729575389078,
        "LONGITUDE": 131.07289492391908,
        "Host matches (capped)": 0,
        "Host matches": 0,
        "Matched": "no",
    },
    index=[0],
)

new_row_2_visitor = pd.DataFrame(
    {
        "LATITUDE": -27.27729575389078,
        "LONGITUDE": 131.07289492391908,
        "Host matches (capped)": 0,
        "Host matches": 0,
        "Matched": "yes",
    },
    index=[0],
)


host_df = pd.concat([new_row_1_host, new_row_2_host, host_df]).reset_index(drop=True)
visitor_df = pd.concat([new_row_1_visitor, new_row_2_visitor, visitor_df]).reset_index(
    drop=True
)

host_df.drop(columns="Unnamed: 0", inplace=True)
visitor_df.drop(columns="Unnamed: 0", inplace=True)

print("---")
print(visitor_df.head(10))


config_file = "network_gradio_config.txt"
# config_file = "network_config.txt"
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
            "Matched",
        ]
    ],
    name="Host homes",
)


network_map.add_data(
    data=visitor_df[
        ["LATITUDE", "LONGITUDE", "Host matches (capped)", "Host matches", "Matched"]
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


import pandas as pd
from keplergl import KeplerGl
import yaml

host_df = pd.read_csv("../pipeline/host_df.csv")
visitor_df = pd.read_csv("../pipeline/visitor_df.csv")
connections_df = pd.read_csv("../pipeline/connections_df.csv")

map_dict = {0.0: "no", 1.0: "yes"}

# host_df['Host coverage'] = host_df['Host coverage'].map(map_dict)
# visitor_df['Visitor coverage'] = visitor_df['Visitor coverage'].map(map_dict)


config_file = "network_gradio_config.txt"
# config_file = "network_config.txt"
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

network_map.save_to_html(file_name="Network_for_gradio.html")


network_map

# %%
import numpy as np
from scipy import spatial
import random
import geopy
import pandas as pd
import gradio as gr


from show_homes.pipeline import geo_utils
from show_homes.pipeline import show_homes_network

df = pd.read_csv("epc_for_show_homes.csv")
# df = pd.read_csv('../analysis/orkney.csv')


# %%
show_homes_network.compute_network_measure(
    df, "Detached House", True, 1, 5, 5, 6, 30, "GB", verbose=True
)


# %%

# import functools

# G = functools.partial(show_homes_network.compute_network_measure, df)


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

    return show_homes_network.compute_network_measure(
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
    # fn=show_homes_network.compute_network_measure,
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
        gr.inputs.Slider(1, 50, default=6, step=1, label="Number of slots/open days"),
        gr.inputs.Slider(1, 50, default=35, step=1, label="Max distance"),
        gr.inputs.Dropdown(
            local_authorities, default="Manchester", label="Local authorities"
        ),
    ],
    outputs=["text", "html"],
    title="Network of Show Homes",
)

demo.launch(share=True)

# %%
