# Assignment 1
James Mortensen

# Question 1
## Code
Finding the parameters to answer question 1 were completed in Python below:

```python
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Assuming the data is in this format
# Load the data from the text file
df = pd.read_csv(
    'docs/assignments/assignment 1/question 1/s048r_202307.txt', sep='\t')


# Binarize the labels
df['test.subject'] = df['test.subject'].apply(
    lambda x: 1 if x == 's048' else 0)
df['test.out'] = df['test.out'].apply(lambda x: 1 if x == 's048' else 0)

# Get the labels and predictions
y_true = df['test.subject']
y_pred = df['test.out']

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Print confusion matrix
print(f"Confusion Matrix:\nTP={tp} FN={fn}\nFP={fp} TN={tn}")

# Compute Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy}")

# Compute FMR
fmr = fp / (fp + tn)
print(f"FMR: {fmr}")

# Compute FNMR
fnmr = fn / (fn + tp)
print(f"FNMR: {fnmr}")

# Compute Precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision}")

# Compute Recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall}")


with open('docs/assignments/assignment 1/question 1/results.txt', 'w') as f:
    f.write(f"Confusion Matrix:\nTP={tp} FN={fn}\nFP={fp} TN={tn}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"FMR: {fmr}\n")
    f.write(f"FNMR: {fnmr}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
```
## Output
    
        Confusion Matrix:
        TP=156 FN=48
        FP=6 TN=194
        Accuracy: 0.8663366336633663
        FMR: 0.03
        FNMR: 0.23529411764705882
        Precision: 0.9629629629629629
        Recall: 0.7647058823529411


# Question 2

a. **Singularities in fingerprint recognition:**
In the field of fingerprint recognition, singularities are significant local structures in the ridge pattern of the fingerprint image, typically referring to ridge bifurcations (where one ridge splits into two) and ridge endings (where a ridge ends). These points are also known as minutiae. The spatial arrangement of these minutiae is unique for each individual, making it a key aspect in fingerprint recognition systems. Algorithms use these points to create a unique pattern or template, which can be used to match with stored templates for identification or verification.

b. **Thinning in fingerprint feature extraction:**
The thinning process, also known as skeletonization, is a step in fingerprint feature extraction where the ridges of the fingerprint are transformed into a skeleton form. This is achieved by iteratively eroding the ridges until they are just one pixel wide. This process reduces the complexity of the image, making it easier to extract relevant features like minutiae while maintaining the overall structure of the ridges. The benefit of thinning is that it makes subsequent feature extraction and matching processes less computationally intensive and more accurate, as it eliminates noise and extraneous data.

c. **Local ridge orientation and frequency:**
Local ridge orientation and frequency are crucial properties of a fingerprint image and are typically computed in the early stages of fingerprint image processing. Ridge orientation refers to the direction of the ridges in a localized region, and ridge frequency refers to the number of ridges present per unit length. 

Computing these two features helps in several ways. Firstly, they allow for enhancement of the fingerprint image, making ridges clearer and improving overall image quality. They also provide a kind of 'macro' feature set, giving a broad-stroke description of the fingerprint which can be used for initial matching, improving speed and efficiency of the recognition process. Finally, knowledge of the local ridge orientation and frequency can aid in accurate localization and extraction of minutiae points, which are critical for the final stages of fingerprint matching.

# Question 3

Given this table of values, we have the angle theta which seems to be given in degrees. The singularity of a local area in a fingerprint is typically defined by the changes in the local orientation (theta) of the ridges. In particular, singularities are often categorised into two types:

1. **Whorl/Circular (Loop)**: This is where the local ridge orientation turns through a full 360 degrees. In terms of theta, you would see an increase or decrease of approximately 360 degrees as you traverse around the singularity.

2. **Bifurcation (Delta)**: This is where ridges meet and split, often forming a 'Y' shape. In terms of theta, you would see an increase or decrease of approximately 180 degrees as you traverse around the singularity.

The computations are completed in Python below:

```python
import numpy as np
import pandas as pd
from dataclasses import dataclass
# Defining the data
data = {'K': [0, 1, 2, 3, 4, 5, 6, 7],
        'theta': [80, 90, 260, 50, 110, 270, 130, 180]}

data2 = {
    'K': [0, 1, 2, 3, 4, 5, 6, 7],
    'theta': [350, 0, 45, 100, 170, 200, 250, 270],
}
df = pd.DataFrame(data)

N = len(df)
pi = np.pi

# Compute sigma (theta_change)
# fill the last value with the first value

def sigma(k):
    # Take the modulo N of k+1
    next_k = (k + 1) % N
    # Access the 'theta' values at the current index and the next index
    return df.loc[next_k, 'theta'] - df.loc[k, 'theta']

df['sigma'] = df['K'].apply(sigma)

# Compute delta
df['delta'] = df['sigma'].apply(lambda x: x if np.abs(
    x) < 90 else (x+180 if x <= -90 else x-180))

def sum_delta():
    sum = 0
    for i in range(N):
        sum += df.loc[i, 'delta']
    return sum


@dataclass
class SingularityType():
    name: str
    angle: int

_t = SingularityType  # Type of singularity

class Singularities():
    LOOP = _t("Loop", 180)
    WHORL = _t("Whorl", 360)
    DELTA = _t("Delta", -180)
    NONE = _t("None", 0)


def determine_type() -> SingularityType:
    if sum_delta() == Singularities.LOOP.angle:
        return Singularities.LOOP.name
    elif sum_delta() == Singularities.WHORL.angle:
        return Singularities.WHORL.name
    elif sum_delta() == Singularities.DELTA.angle:
        return Singularities.DELTA.name
    else:
        return Singularities.NONE.name


with open('docs/assignments/assignment 1/question 3/results.txt', 'w') as f:
    f.write(f"Sum of delta: {sum_delta()}, Type: {determine_type()}\n\n")
    f.write(str(df))
print(df)
print(f"Sum of delta: {sum_delta()}, Type: {determine_type()}")

```
## Output

        Sum of delta: 180, Type: Loop

        K  theta  sigma  delta
        0  0     80     10     10
        1  1     90    170    -10
        2  2    260   -210    -30
        3  3     50     60     60
        4  4    110    160    -20
        5  5    270   -140     40
        6  6    130     50     50
        7  7    180   -100     80




# Question 4
Minutiae points, including bifurcation points and termination points, are features in fingerprint analysis. They are used in biometric systems for identifying individuals based on their unique fingerprint patterns. 

Here's how you can describe a 3x3 binary pixel grid for a bifurcation point and a non-minutiae point:

1. Bifurcation Point:

A bifurcation point in a fingerprint is a point where one ridge splits into two (or conversely two ridges merge into one). In a simplified 3x3 binary pixel grid, a bifurcation point might look like this:

```
0 1 0
1 1 1
0 1 0
```

Where '1' indicates the presence of a ridge, and '0' indicates the absence of a ridge. For this bifurcation point, the values of `b0,...,b7` (starting from the pixel in the top left and going clockwise around the center pixel) would be `b0=0, b1=1, b2=0, b3=1, b4=0, b5=1, b6=0, b7=1`.

The crossing number (CN), used to identify minutiae, is defined as half the sum of the absolute differences between pairs of consecutive pixels in the sequence `b0,...,b7`. For the bifurcation point, CN = 3.

2. Non-minutiae Point:

A non-minutiae point in a fingerprint is a point that is not a feature point, like a ridge that doesn't split or end. In a simplified 3x3 binary pixel grid, a non-minutiae point might look like this:

```
0 1 0
0 1 0
0 1 0
```

For this non-minutiae point, the values of `b0,...,b7` (starting from the pixel in the top left and going clockwise around the center pixel) would be `b0=0, b1=1, b2=0, b3=0, b4=0, b5=1, b6=0, b7=0`.

The crossing number for this non-minutiae point would be 0, as there are no changes from 0 to 1 or from 1 to 0 in the sequence `b0,...,b7`.


# Question 5

## Introduction

CN is used to identify ridge bifurcations (CN=3) and ridge endings (CN=1). It is calculated as half the sum of the absolute differences between pairs of consecutive pixels (b0,...,b7), taken in a clockwise direction.



## Code
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

## Output
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