import numpy as np

# Binary matrix
matrix = np.array([
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1, 1, 0, 1, 0, 0]
])

# Invert the matrix
matrix = np.logical_not(matrix).astype(int)

# List to store minutiae points
minutiae_points = []

# Traverse through each pixel in the matrix
for i in range(1, matrix.shape[0] - 1):  # Ignore the first and last row
    for j in range(1, matrix.shape[1] - 1):  # Ignore the first and last column
        if matrix[i][j] != 0:  # Only consider '1's
            # 3x3 submatrix centered around the pixel (i, j)
            sub_matrix = matrix[i-1:i+2, j-1:j+2]
            # Flatten the submatrix and calculate the Crossing Number
            sub_array = [sub_matrix[i, j] for i in range(3) for j in range(3)]
            CN = 0.5 * np.abs(np.sum(sub_array) - sub_array[4])
            if CN == 1:  # Ridge ending
                minutiae_points.append(((i, j), 'Ridge Ending'))
            elif CN == 3:  # Ridge bifurcation
                minutiae_points.append(((i, j), 'Ridge Bifurcation'))

for point in minutiae_points:
    print(f"Point {point[0]} is a {point[1]}")
