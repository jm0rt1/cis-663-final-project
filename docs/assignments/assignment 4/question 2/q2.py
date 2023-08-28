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
