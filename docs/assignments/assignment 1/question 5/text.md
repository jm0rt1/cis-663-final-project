Given that we have already transformed the data to a binary matrix, let's proceed to detect the minutiae points using the Crossing Number (CN) concept. As discussed earlier, CN is used to identify ridge bifurcations (CN=3) and ridge endings (CN=1). It is calculated as half the sum of the absolute differences between pairs of consecutive pixels (b0,...,b7), taken in a clockwise direction.

Here's the Python code to find the minutiae points:

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

Please note that we are skipping the first and last rows/columns in the matrix as we can't get a full 3x3 submatrix centered around these points (as per your condition of ignoring edge-centered points). Also, this is a very basic method for detecting minutiae and doesn't handle any post-processing steps such as noise removal or false minutiae elimination.

Again, keep in mind that real-world fingerprint analysis is a lot more complex, and advanced techniques like image enhancement and thinning algorithms, use of machine learning, etc., are usually employed.