# Assignment 4
Jamie Mortensen

## Question 1

I answer the 3 parts to the question via the code below. Luckily scipy provides a function that can do the heavy lifting for us, `scipy.spatial.distance.pdist`. This function computes the pairwise distances between all points in a dataset. The `metric` parameter allows us to specify the distance metric to use. The `scipy.spatial.distance.squareform` function converts the pairwise distances into a square matrix.

```python
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
```

The output of the code above is shown below, each pair is 

cityblock distance = manhattan distance

```
closest euclidean pairs = [(9, 37), (142, 101), (34, 9)]
closest cityblock pairs = [(37, 9), (101, 142), (9, 34)]
closest cosine pairs = [(37, 9), (37, 34), (142, 101)]
```

## Question 2



```python
def naive_bayes(data, input_features):
    # Calculate prior probabilities
    total_fruits = sum(data[fruit]['total'] for fruit in data)
    priors = {fruit: data[fruit]['total'] / total_fruits for fruit in data}

    # Calculate likelihoods
    likelihoods = {}
    for fruit in data:
        likelihood = 1
        for feature, value in input_features.items():
            if value:
                likelihood *= data[fruit][feature] / data[fruit]['total']
            else:
                likelihood *= (1 - (data[fruit]
                               [feature] / data[fruit]['total']))
        likelihoods[fruit] = likelihood

    # Apply Bayes theorem
    unnormalized_posteriors = {
        fruit: priors[fruit] * likelihoods[fruit] for fruit in data}

    # Normalize the probabilities
    normalization_factor = sum(unnormalized_posteriors.values())
    posteriors = {fruit: prob / normalization_factor for fruit,
                  prob in unnormalized_posteriors.items()}

    return posteriors


# Dataset
data = {
    'Strawberry': {'Red': 300, 'Leaves': 250, 'Seeds': 200, 'total': 350},
    'Apple': {'Red': 400, 'Leaves': 100, 'Seeds': 300, 'total': 600},
    'Pear': {'Red': 100, 'Leaves': 50, 'Seeds': 200, 'total': 250},
}

# Test the classifier
inputs = [
    {'Red': False, 'Leaves': False, 'Seeds': True},
    {'Red': True, 'Leaves': False, 'Seeds': False},
    {'Red': True, 'Leaves': True, 'Seeds': True},
]

for idx, input_features in enumerate(inputs, 1):
    probabilities = naive_bayes(data, input_features)
    print(f"Input {idx}:")
    for fruit, probability in probabilities.items():
        print(f"P({fruit}|Input{idx}) = {probability:.4f}")
    print()


```

```
Input 1:
P(Strawberry|Input1) = 0.0435
P(Apple|Input1) = 0.4445
P(Pear|Input1) = 0.5120

Input 2:
P(Strawberry|Input2) = 0.1674
P(Apple|Input2) = 0.7596
P(Pear|Input2) = 0.0729

Input 3:
P(Strawberry|Input3) = 0.7128
P(Apple|Input3) = 0.1940
P(Pear|Input3) = 0.0931
```

## Question 3
**a)**

The ROC (Receiver Operating Characteristic) curve is a graphical representation of the performance of a binary classification model. Specifically, it represents the trade-off between the true positive rate (sensitivity) and the false positive rate (1 - specificity). 

A point on the ROC curve corresponds to a particular decision threshold of the classifier:
- The x-coordinate of the point represents the false positive rate (FPR) for that threshold.
- The y-coordinate of the point represents the true positive rate (TPR) for that threshold.

When you adjust the threshold of the classifier (e.g., the probability threshold in probabilistic classifiers), you move along the ROC curve.

**b)**

When comparing two ROC curves:
- A curve that is more to the top-left corner of the plot is better. This indicates a higher true positive rate for a given false positive rate or vice versa.
- The Area Under the Curve (AUC) can be used as a single metric to summarize the ROC curve. A model with a higher AUC is generally considered better. An AUC of 1.0 indicates a perfect classifier, while an AUC of 0.5 indicates a classifier that is no better than random guessing.

In essence, the better model's ROC curve will have more area under it and will climb faster toward the top-left of the chart.

The curve that is showing a better model here is A, based on these two criteria.



**c)**

For a random guess in a binary classification:
- The ROC curve would be a diagonal line running from the bottom-left corner to the top-right corner of the ROC space. This is often referred to as the "line of no discrimination."
- The reason is that a random guess will produce equal probabilities of a positive or negative outcome for all instances, leading to an equal chance (a coin flip) of being above or below any chosen threshold. As a result, for every TPR value, you'll get an equivalent FPR value, hence the diagonal line.
- The AUC for this curve will be 0.5, indicating no discrimination capability.

What would be varying over the set of random guess algorithms?
- The particular random predictions may vary (since they're random), but on average, over multiple runs or over many instances, the performance would converge to this diagonal line in the ROC space. Different random guess algorithms would essentially shuffle the order of predictions but will not change the overall performance characteristic represented by the diagonal ROC curve.


## Question 4


1. **True Positive (TP)**: A legitimate user is accepted.
   * Benefit: You charge $10.
   
2. **False Positive (FP)**: An imposter is wrongly accepted.
   * Cost: You charge $10, but then you have to refund it, so net is $0.
   * Note: The system could be defrauded multiple times by imposters if the FP rate is high.
   
3. **True Negative (TN)**: An imposter is correctly rejected.
   * Benefit: Fraud is avoided.
   
4. **False Negative (FN)**: A legitimate user is wrongly rejected.
   * Cost: Lost revenue of $100 because they'll use another parking garage.

Given the costs and benefits:

* Each TP is worth $10.
* Each FP costs nothing ($0) because you refund the amount, but it has security implications and could impact trust in your system.
* Each TN has no immediate financial value, but it maintains trust and security in your system.
* Each FN costs $100.

Given that 50% of requests are from imposters, you want to minimize both FP and FN, but FN has a much higher cost.

**Decision**:
You should choose an algorithm that has a very low false negative rate. This means you'd likely be willing to tolerate more false positives (and then deal with the refunds) in order to avoid the high cost of false negatives.

**Setting the threshold**:
Typically, an ROC curve is used to visualize the trade-offs between TP and FP rates for different thresholds. In this case:

* Lowering the threshold will increase both the TP and FP rates. This will make it more likely for legitimate users to gain access, but also more likely for imposters to get in. This may be acceptable since an imposter just results in a refund, but a legitimate user being rejected costs a lot more.

* Raising the threshold will decrease both the TP and FP rates. This makes it harder for both legitimate users and imposters to gain access.

Given the high cost of false negatives, one would likely want to set a threshold that's relatively low. This will allow more users (and unfortunately, more imposters) in, but the financial cost of the occasional imposter is outweighed by the large cost of rejecting a legitimate user.

**Conclusion**:
Choose an algorithm that allows you to control the threshold and has a good balance of TP and FP in the desired range. Set the threshold such that the FN rate is minimized, even if it means a slightly higher FP rate. It might be useful to continuously monitor the rates and adjust as needed, especially if there are changes in user behavior or the imposter rate.

The algorithms that produce this outcome TS2-norm, due to its sharp drop in False Rejects at the cost of a slight increase in False Accepts. It is the curve which also pushes closest to the lower right, however the threshold may be reasonable at a .2 False Accept rate.


## Question 5

Given the fiducial points and their corresponding times, you can extract the features you listed (RQ, RP’, RP, RL’, RS, RS’, RT, RT’) by subtracting the appropriate fiducial point times from each other.

Here's how you can calculate each feature:

1. \( RQ \) = R - Q
2. \( RP' \) = R - P'
3. \( RP \) = R - P
4. \( RL' \) = R - L'
5. \( RS \) = R - S
6. \( RS' \) = R - S'
7. \( RT \) = R - T
8. \( RT' \) = R - T'

Given the fiducial point times provided:

```python
fiducial_points = {
    'L-prime': 8.6,
    'P': 8.9,
    'P-prime': 9.1,
    'Q': 9.3,
    'R': 9.5,
    'S': 9.6,
    'S-prime': 9.9,
    'T': 10.4,
    'T-prime': 10.6
}

features = {
    'RQ': fiducial_points['R'] - fiducial_points['Q'],
    'RP-prime': fiducial_points['R'] - fiducial_points['P-prime'],
    'RP': fiducial_points['R'] - fiducial_points['P'],
    'RL-prime': fiducial_points['R'] - fiducial_points['L-prime'],
    'RS': fiducial_points['R'] - fiducial_points['S'],
    'RS-prime': fiducial_points['R'] - fiducial_points['S-prime'],
    'RT': fiducial_points['R'] - fiducial_points['T'],
    'RT-prime': fiducial_points['R'] - fiducial_points['T-prime']
}

for feature, value in features.items():
    print(f"{feature} = {value:.2f} s")

```

```
RQ = 0.20 s
RP-prime = 0.40 s
RP = 0.60 s
RL-prime = 0.90 s
RS = -0.10 s
RS-prime = -0.40 s
RT = -0.90 s
RT-prime = -1.10 sy
```