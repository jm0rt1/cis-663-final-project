from scipy.spatial import distance
import numpy as np
import pandas as pd


# Loading the provided dataset
iris_df = pd.read_csv(
    'docs/assignments/assignment 4/question 1/data.csv', header=None)
# Adding column names to the dataset
column_names = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'class']
iris_df.columns = column_names


def get_three_closest_pairs(dist_matrix):
    # We set the diagonal to a high value to avoid considering the distance of a point to itself
    np.fill_diagonal(dist_matrix, np.inf)

    # Getting the indices of the sorted distances
    sorted_indices = np.dstack(np.unravel_index(
        np.argsort(dist_matrix.ravel()), dist_matrix.shape))[0]

    # Extracting the three closest pairs
    three_closest = sorted_indices[:3]

    return [(i, j) for i, j in three_closest]


# Computing pairwise distances for each metric using the full dataset
data_points_full = iris_df.iloc[:, :4].values
euclidean_distances_full = distance.squareform(
    distance.pdist(data_points_full, metric='euclidean'))
manhattan_distances_full = distance.squareform(
    distance.pdist(data_points_full, metric='cityblock'))
cosine_distances_full = distance.squareform(
    distance.pdist(data_points_full, metric='cosine'))

# Getting the three closest pairs for each metric using the full dataset
euclidean_pairs_full = get_three_closest_pairs(euclidean_distances_full)
manhattan_pairs_full = get_three_closest_pairs(manhattan_distances_full)
cosine_pairs_full = get_three_closest_pairs(cosine_distances_full)


print(f"closest euclidean pairs = {euclidean_pairs_full}")
print(f"closest manhattan pairs = {manhattan_pairs_full}")
print(f"closest cosine pairs = {cosine_pairs_full}")
