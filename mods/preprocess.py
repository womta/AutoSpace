# Dependencies
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list

# Local variables
import global_vars

# Set Index to df
def index_data(df):
	#df = df.with_row_index(name = 'index', offset = 1)
	df['index'] = range(1, len(df) + 1)
	return df


# Randomly sample a DataFrame
def df_random_sampling(df, sample_size = global_vars.RANDOM_SAMPLING):
	if len(df) > sample_size:
		return df.sample(n = sample_size, random_state = global_vars.RANDOM_STATE)  # Set random_state for reproducibility
	else:
		return df
	

# Handle your missing data in the df
def handle_missing_data(df, features):
	
	# Replace inf and -inf with NaN in the specified columns
	df[features] = df[features].replace([np.inf, -np.inf], np.nan)

	# Drop rows with any missing values in the specified columns
	df = df.dropna(subset = features)

	#df = df.drop_nulls(subset = features)

	return df


# Create correlation matrix
def create_cor_matrix(df, order_matrix = True):

	# Calculate the correlation matrix
	corr = df.corr(method = 'spearman')

	if order_matrix:

		from scipy.spatial.distance import squareform
		from scipy.cluster.hierarchy import linkage

		# Convert correlation matrix to distance matrix
		distance_matrix = 1 - np.abs(corr)

		# Convert to condensed form
		condensed_distance = squareform(distance_matrix.values)

		# Perform hierarchical clustering
		#linkage_matrix = linkage(distance_matrix, method = 'average')
		linkage_matrix = linkage(condensed_distance, method = 'average')

		# Get the order of rows and columns
		dendrogram_order = leaves_list(linkage_matrix)

		# Reorder the correlation matrix
		corr = corr.iloc[dendrogram_order, dendrogram_order]

	return corr