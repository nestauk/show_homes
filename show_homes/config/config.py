from show_homes import PROJECT_DIR
from pathlib import Path

LOCAL_DATA_DIR = Path("/Users/juliasuter/Documents/ASF_data")

FIG_OUT_PATH = PROJECT_DIR / "outputs/figures"

DIST_DUR_OUT_DATA_PATH = PROJECT_DIR / "outputs/data/dist_and_durations"
DIST_MATRIX_OUT_DATA_PATH = PROJECT_DIR / "outputs/data/dist_matrices"
HOST_VIS_CON_OUT_DATA_PATH = PROJECT_DIR / "outputs/data/host_vis_connections"
SHOW_HOME_DATA_OUT_DATA_PATH = PROJECT_DIR / "outputs/data/show_home_data"

KEPLER_CONFIG_PATH = PROJECT_DIR / "outputs/maps/config"
GRADIO_KEPLER_CONFIG = PROJECT_DIR / KEPLER_CONFIG_PATH / "network_gradio_config.txt"
MAPS_OUT_PATH = PROJECT_DIR / "outputs/maps"

# Has to be in same folder or subfolder as script
GRADIO_OUT_MAPS_PATH = PROJECT_DIR / "show_homes/analysis/maps"
GRADIO_OUT_MAP_NAME = "Generated_network_map_{}.html"
