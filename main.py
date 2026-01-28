# Dependencies
import pandas as pd

# Modules
from mods.viz import viz_density, viz_distribution, viz_linearity, missing_data_heatmap, viz_corr

from mods.preprocess import index_data, handle_missing_data, create_cor_matrix

# Global vars
import global_vars

# Read Main df
df = pd.read_parquet(global_vars.main_df)

# Defined Featuers
features = ['Open', 'Close', 'High', 'Low']

# Create index on df
df = index_data(df)

# Heatmap Visualization on Missing Data per Ticker
missing_data_heatmap(0, df, features)

# Missing Rows On Basic Features
df = handle_missing_data(df, features)

# Distributions of features violin + multi histogram facetted by Open, Close, High, Low and Volume
viz_density(0, df.copy(), features)

# Histograms of features
viz_distribution(0, df.copy(), features)
				
# Linearity of features
viz_linearity(0, df.copy(), features)

# Area plot
#viz_area(settings.CLOUD, df.copy())

# Create clustered correlation matrix
corr = create_cor_matrix(df[features])

# Correlation Matrix, CLustered Heatmap
viz_corr(0, corr)

# Create SPLOM
#splom(settings.CLOUD, df.copy(), features, 'sector', 'Volume')

# Indication Number of Components Through Screeplot
#viz_scree(settings.CLOUD, eigenvalues, factor_count)

# Loadings of components Through Spiderplot
#viz_spider(settings.CLOUD, factor_loadings, factor_count)

# Explorative 3d scatterplot
#viz_3D(settings.CLOUD, df.copy()) # Copy to avoid adjustments

# Class Inbalance Plot
#viz_balance(settings.CLOUD, train_df['target_class'], title = 'Training Set Class Balance')

# Number on Missing Data
#check_missing_columns(df, features)



# Check the size of the Train and Test Set
#check_test_train_data(train_df, test_df)

