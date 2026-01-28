#!pip install fastcluster
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
import pandas as pd
import numpy as np
import random
import plotly.io as pio

from sklearn.metrics import roc_curve, auc
from plotly.subplots import make_subplots

# Color maps
import matplotlib.pyplot as plt
import matplotlib as matplotlib

# Modules
from mods.preprocess import df_random_sampling 

# Local variables
import global_vars

# Plotly's output
pio.renderers.default = 'json'
pio.renderers.default = 'browser'

# Set it as the default
pio.templates.default = 'plotly_dark'

#################### 
# Viz Preparations #
####################


     

# Store Plotly Objects
def store_plotly_object(fig, file_name):
    plot_object = f'{global_vars.DIR_PLOTS}{file_name}'
    fig.write_json(plot_object, pretty = True)

    # Succesfully Stored Plotly Object
    return True

# Scale a pandas series object to a range e.g. from 1 to 20 (for visualization purpose, e.g. size)
def scale_series(series, min_val = 1, max_val = 20):
    normalized = (series - series.min()) / (series.max() - series.min())
    scaled = normalized * (max_val - min_val) + min_val
    return scaled



#####################
# Basic Stats Plots #
#####################

# Create violin plots for comparing features
def viz_density(cloud, df, features, num_features = global_vars.SAMPLE_FEATURES_VIZ):

    # Limit features to 25 randomly selected columns if there are more than 25
    features_random = features

    if len(features) > num_features:
        random.seed(global_vars.RANDOM_STATE)
        features_random = random.sample(features, num_features)

    # Large number of data points to plot
    if len(df) >= global_vars.RANDOM_SAMPLING:
        df = df_random_sampling(df, global_vars.RANDOM_SAMPLING)

    # Melt df
    cat_features = ['index']
    df_melted = pd.melt(df, id_vars = cat_features, value_vars = features_random, var_name = 'Feature', value_name = 'Value') 

    fig = go.Figure()
    for i, feature in enumerate(df_melted['Feature'].unique()):
        values = df_melted.loc[df_melted['Feature'] == feature, 'Value']

        # Compute KDE with numpy
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        y_density = kde(x_range)

        # Convert RGB to rgba
        base_color = global_vars.COLOR_PALETTE1[i % len(global_vars.COLOR_PALETTE1)]  # pick from palette
        r, g, b = tuple(int(base_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        rgba_color = f'rgba({r},{g},{b},0.3)'   # ✅ same hue, transparent fill

        # Add filled density curve
        fig.add_trace(go.Scatter(
            x = x_range,
            y = y_density,
            mode = 'lines',
            name = feature,
            fill = 'tozeroy',            # ✅ fills under the curve
            fillcolor = rgba_color,
            hovertemplate = 'Feature: %{fullData.name}<br>X: %{x}<br>Density: %{y}<extra></extra>'
        ))

    fig.update_layout(
        title = 'Density Plot Overlay',
        xaxis_title = 'Value',
        yaxis_title = 'Density'
    )

    # Create the violin plot
    fig.update_layout(title = 'Density Plots of Features', barmode = 'overlay', bargap = 0.1)
    
    # Disable hover information
    fig.update_traces(hoverinfo = 'none', opacity = 0.6)

    # Store or Plot Viz Object
    #if(cloud):
    #    print('store object')
    #    store_plotly_object(fig, global_vars.VIZ_DENSITY_PATH)
    #else:
    #    store_plotly_object(fig, global_vars.VIZ_DENSITY_PATH)
    fig.show()

    # Plot Succesful
    return True


# Visualizing up to 25 random feature distributions in histograms
def viz_distribution(cloud, df, features):
    
    # Limit features to 25 randomly selected columns if there are more than 25
    features_random = features

    if len(features) > global_vars.SAMPLE_FEATURES_VIZ:
        
        # Random features
        random.seed(global_vars.RANDOM_STATE)
        features_random = random.sample(features, global_vars.SAMPLE_FEATURES_VIZ)

     # Large number of data points to plot
    if len(df) >= global_vars.RANDOM_SAMPLING:
        # Data Preproces Module
        
        df = df_random_sampling(df, global_vars.RANDOM_SAMPLING)

    # Determine the number of features
    num_features = len(features_random)

    # Calculate the number of rows and columns for the subplot grid
    num_cols = global_vars.DEFAULT_FACETS
    num_rows = int(np.ceil(num_features / num_cols))

    # Create a subplot figure with calculated rows and columns
    fig = sp.make_subplots(rows = num_rows, cols = num_cols, subplot_titles = features_random)

    # Add histograms to each subplot
    count = 0
    for i, feature in enumerate(features_random):
        row = i // num_cols + 1
        col = i % num_cols + 1

        fig.add_trace(
            px.histogram(df, x = feature).data[0],
            row = row, col = col
        )
        count += 1
        
    # Update layout for better spacing
    fig.update_layout(title_text = 'Linearity of Features', showlegend = False) # width = global_vars.VIZ_WIDTH,
    
    # Disable hover information
    fig.update_traces(hoverinfo = 'none')
 
    # Store Viz Object
    #if(cloud):
        # Update layout for better spacing
    #    store_plotly_object(fig, global_vars.VIZ_DISTR_PATH)
    #else:
        # Update layout for better spacing
  
    #store_plotly_object(fig, global_vars.VIZ_DISTR_PATH)
    fig.show()

    # Plot Succesful
    return True


# Visualizing up to 25 random feature distributions in histograms
def viz_linearity(cloud, df, features):

    # Limit features to 25 randomly selected columns if there are more than 25
    features_random = features

    if len(features) > global_vars.SAMPLE_FEATURES_VIZ:
        random.seed(global_vars.RANDOM_STATE)
        features_random = random.sample(features, global_vars.SAMPLE_FEATURES_VIZ)

    # Large number of data points to plot
    if len(df) >= global_vars.VIZ_SAMPLING:
         # Data Preproces Module
        df = df_random_sampling(df, global_vars.VIZ_SAMPLING)

    # Add index
    df['Index'] = range(1, len(df) + 1)
    
    # Determine the number of features
    num_features = len(features_random)

    # Calculate the number of rows and columns for the subplot grid
    #num_cols = int(np.ceil(np.sqrt(num_features)))
    num_cols = global_vars.DEFAULT_FACETS
    num_rows = int(np.ceil(num_features / num_cols))

    # Create a subplot figure with calculated rows and columns
    fig = sp.make_subplots(rows = num_rows, cols = num_cols, subplot_titles = features_random)

    # Add histograms to each subplot
    count = 0
    for i, feature in enumerate(features_random):
        row = i // num_cols + 1
        col = i % num_cols + 1

        fig.add_trace(
            # Create the scatter plot
            px.scatter(df, x = 'Index', y = feature, size = scale_series(df[feature]), trendline = 'ols', trendline_color_override = 'red', size_max = 10, render_mode = 'webgl').data[0],
            row = row, col = col  
        )
        fig.update_traces(marker_line = dict(width = 1, color = 'DarkSlateGray'))
        count += 1

    # Update layout for better spacing
    fig.update_layout(title_text = 'Linearity of Features', showlegend = False) # width = global_vars.VIZ_WIDTH,
    
    # Disable hover information
    fig.update_traces(hoverinfo = 'none')

    # Store Viz Object
    #if(cloud):
    #    print('store object')
    #    store_plotly_object(fig, global_vars.VIZ_LINEARITY_PATH)
    #else:
    fig.show()
    #    store_plotly_object(fig, global_vars.VIZ_LINEARITY_PATH)

    # Plot Succesful
    return True


# Missing Data Heatmap
def missing_data_heatmap(cloud, df, features, date_col = 'date_time', ticker_col = 'ticker'):
  
    # Ensure date_time is datetime and sorted
    #df = df.to_pandas()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Count missing values per row for the given features
    df['missing_count'] = df[features].isna().sum(axis = 1)

    # Aggregate per time/ticker
    pivot_df = df.pivot_table(
        index = date_col,
        columns = ticker_col,
        values = 'missing_count',
        aggfunc = 'sum',
        fill_value = 0
    )

    # Plot heatmap with tickers on the x-axis
    fig = px.imshow(
        pivot_df,  
        labels = dict(x = "Ticker", y = "Date/Time", color = "Missing Features"),
        x = pivot_df.columns,
        y = pivot_df.index,
        aspect = "auto",
        color_continuous_scale = [global_vars.MEDIUM_COLOR, global_vars.LOW_COLOR]  # present = light gray, missing = red
    )

    fig.update_layout(
        title = "Missing Data Heatmap",
        xaxis_title = "Ticker",
        yaxis_title = "Date/Time",
        yaxis_autorange = "reversed"  # ✅ time flows top-to-bottom
    )

     # Store Viz Object
    #if(cloud):
    #    store_plotly_object(fig, global_vars.VIZ_MISSING_PATH)
    #else:
    #    store_plotly_object(fig, global_vars.VIZ_MISSING_PATH)
    fig.show()
    
    # Plot Succesful
    return True


# Visualizing a clustered correlation matrix heatmap
def viz_corr(cloud, corr):

    # Create an interactive heatmap
    fig = px.imshow(corr, text_auto = False, color_continuous_scale = [(0, '#800080'), (0.5, '#000000'), (1, '#FFFF00')], aspect = 'auto')

   # Add spacing between cells to simulate borders
    fig.update_traces(xgap = 1, ygap = 1)

    # Set the figure background to act as border color
    fig.update_layout(
        plot_bgcolor = '#000000',  # or "black"
        xaxis = dict(showgrid = False),
        yaxis = dict(showgrid = False)
    )

    # Disable hover information
    #fig.update_traces(hoverinfo = 'none')

    # Disable all interactivity (hovering, zooming, panning)
    fig.update_layout(
        #hovermode = False,
        xaxis = dict(fixedrange = True),
        yaxis = dict(fixedrange = True)
        #autosize = False,
        #width = 0.7 * global_vars.VIZ_WIDTH,
        #height = 0.7 * global_vars.VIZ_HEIGHT
    )

   # Store or Plot Viz Object
    #if(cloud):
    #    print('store object')
    #    store_plotly_object(fig, global_vars.VIZ_COR_HEATMAP_PATH)
    #else:
    #    store_plotly_object(fig, global_vars.VIZ_COR_HEATMAP_PATH)
    fig.show()

    # Plot Succesful
    return True