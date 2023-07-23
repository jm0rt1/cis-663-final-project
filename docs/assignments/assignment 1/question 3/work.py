import numpy as np
import pandas as pd

# Defining the data
data = {'K': [0, 1, 2, 3, 4, 5, 6, 7],
        'theta': [80, 90, 260, 50, 110, 270, 130, 180]}
df = pd.DataFrame(data)

N = len(df)
pi = np.pi

# Compute sigma (theta_change)
df['sigma'] = df['theta'].shift(-1).fillna(df.iloc[0]['theta']) - df['theta']

# Compute delta
df['delta'] = df['sigma'].apply(lambda x: x if np.abs(
    x) < pi/2 else (x+pi if x <= -pi/2 else x-pi))

# Identify singularity type based on delta
df['singularity'] = df['delta'].apply(lambda x: 'Loop or Whorl' if np.abs(
    x) >= 2*pi else ('Delta' if np.abs(x) >= pi else 'None'))

print(df)
