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


t = SingularityType  # Type of singularity


class Singularities():
    LOOP = t("Loop", 180)
    WHORL = t("Whorl", 360)
    DELTA = t("Delta", -180)
    NONE = t("None", 0)


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
