# Question 5

## Introduction

CN is used to identify ridge bifurcations (CN=3) and ridge endings (CN=1). It is calculated as half the sum of the absolute differences between pairs of consecutive pixels (b0,...,b7), taken in a clockwise direction.



## Solution
Here's the Python code to find the minutiae points for the provided matrix:

```python
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
```
# Output
    Point (1, 5) is a Ridge Bifurcation
    Point (2, 8) is a Ridge Ending
    Point (3, 5) is a Ridge Bifurcation
    Point (4, 1) is a Ridge Bifurcation
    Point (4, 2) is a Ridge Bifurcation
    Point (5, 1) is a Ridge Bifurcation
    Point (7, 4) is a Ridge Ending
    Point (7, 7) is a Ridge Bifurcation
    Point (7, 8) is a Ridge Bifurcation
    Point (8, 2) is a Ridge Bifurcation
    Point (8, 7) is a Ridge Bifurcation


The analysis of the provided binary matrix begins by examining each pixel, while omitting the first and last rows and columns. This exclusion is necessary to obtain a full 3x3 submatrix centered around each examined point. Given this constraint, edge-centered points are ignored.

The primary focus is on the pixels that represent ridges, as determined by the adopted convention of binary representation. These potential minutiae points are further examined by taking a 3x3 submatrix centered around each ridge pixel. This submatrix is flattened, and the Crossing Number (CN) is calculated.

The Crossing Number represents a critical feature of the topography of the matrix. If a point has a Crossing Number of 1, it is a ridge ending. Conversely, if it has a Crossing Number of 3, it is a ridge bifurcation.