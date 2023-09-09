# Assignment 5

All of the Python code displayed in this single Markdown file were run using Python 3.11.2.

## Question 1
### a)
#### Code
```python
def calculate_prob_flagged_given_positive(sensitivity, specificity, p_flagged):
    p_not_flagged = 1 - p_flagged
    p_positive_given_flagged = sensitivity
    p_positive_given_not_flagged = 1 - specificity

    p_positive = p_positive_given_flagged * p_flagged + \
        p_positive_given_not_flagged * p_not_flagged

    return (p_positive_given_flagged * p_flagged) / p_positive


# Given values
sensitivity = 0.99
specificity = 0.99
p_flagged = 0.005

# Calculate P(Flagged|+)
prob = calculate_prob_flagged_given_positive(
    sensitivity, specificity, p_flagged)
print(f"P(Flagged|+) with given values: {prob:.4f}")
```

#### Output:
```
P(Flagged|+) with given values: 0.3322
```



### b)

- Increase Sensitivity (but this is already at 99% and can't be increased much further).
- Increase Specificity. If Specificity is increased (i.e., reducing the false positive rate), then fewer unflagged individuals will test positive, making it more likely that a positive result comes from a flagged individual.

#### Code:
```python
def find_required_specificity(target_prob, sensitivity, p_flagged):
    # Iterating through possible specificity values with a step size of 0.0001
    for specificity in [i/10000 for i in range(10000, 0, -1)]:
        prob = calculate_prob_flagged_given_positive(sensitivity, specificity, p_flagged)
        if prob > target_prob:
            return specificity
    return 0

# Given values
sensitivity = 0.99
specificity = 0.99
p_flagged = 0.005

# Estimate required specificity
target_prob = 0.45
required_specificity = find_required_specificity(target_prob, sensitivity, p_flagged)
print(f"Required specificity to achieve P(Flagged|+) > {target_prob}: {required_specificity:.4f}")

```

#### Output:

```
Required specificity to achieve P(Flagged|+) > 0.45: 1.0000
```

## Question 2
Given a Gaussian distribution, values that are more than two standard deviations above or below the mean can be considered outliers.

A python script below can be used to find and print the outliers in a given dataset.

### Code
```python
def find_outliers(data, mu, sigma):
    upper_limit = mu + 2 * sigma
    lower_limit = mu - 2 * sigma

    outliers = [x for x in data if x < lower_limit or x > upper_limit]

    return outliers


# Given values
mu = 5.1
sigma = 1.9
data = [2.10, 4.20, 6.00, 4.50, 4.20, 4.30, 2.50, 6.90, 8.10, 7.00, 9.10, 5.30, 6.20,
        4.80, 1.80, 5.80, 3.30, 3.30, 4.00, 6.4]

outliers = find_outliers(data, mu, sigma)

print("Outliers:", outliers)
```
### Output

When you run this script, it'll print the outliers from the dataset. Based on the criteria provided, the output will be:
```
Outliers: [9.1]
```