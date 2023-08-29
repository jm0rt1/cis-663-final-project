import numpy as np
import pandas as pd
from scipy.spatial import distance

DATA_PATH = 'docs/assignments/assignment 4/question 1/data.csv'
COLUMN_NAMES = ['sepal_length', 'sepal_width',
                'petal_length', 'petal_width', 'class']
METRICS = ['euclidean', 'cityblock', 'cosine']


def load_data(path, column_names):
    df = pd.read_csv(path, header=None)
    df.columns = column_names
    return df


def compute_pairwise_distances(data, metric):
    return distance.squareform(distance.pdist(data, metric=metric))


def get_three_closest_pairs(dist_matrix):
    np.fill_diagonal(dist_matrix, np.inf)
    sorted_indices = np.dstack(np.unravel_index(
        np.argsort(dist_matrix.ravel()), dist_matrix.shape))[0]
    return [(i, j) for i, j in sorted_indices[:3]]


def main():
    iris_df = load_data(DATA_PATH, COLUMN_NAMES)
    data_points = iris_df.iloc[:, :4].values

    for metric in METRICS:
        distances = compute_pairwise_distances(data_points, metric)
        closest_pairs = get_three_closest_pairs(distances)
        print(f"closest {metric} pairs = {closest_pairs}")


if __name__ == "__main__":
    main()
