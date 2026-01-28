# Dependencies
import plotly.express as px

data_dir = 'DATA' # Data Directory 
main_df = f'{data_dir}/df.parquet' 


# Visualization Settings
SAMPLE_FEATURES_VIZ = 25
RANDOM_SAMPLING = 10_000
VIZ_SAMPLING = 3_000
DEFAULT_FACETS = 10

# Colors
LOW_COLOR = '#9B1C1C'
MEDIUM_COLOR = '#000000'
HIGH_COLOR = '#1F8311'

# Color Palette
COLOR_PALETTE1 = px.colors.qualitative.Plotly[:100]  # 100 unique colors

# Reproducibility
RANDOM_STATE = 42
