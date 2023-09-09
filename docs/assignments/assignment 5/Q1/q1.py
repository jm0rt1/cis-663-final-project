def calculate_prob_flagged_given_positive(sensitivity, specificity, p_flagged):
    p_not_flagged = 1 - p_flagged
    p_positive_given_flagged = sensitivity
    p_positive_given_not_flagged = 1 - specificity

    p_positive = p_positive_given_flagged * p_flagged + \
        p_positive_given_not_flagged * p_not_flagged

    return (p_positive_given_flagged * p_flagged) / p_positive


def find_required_specificity(target_prob, sensitivity, p_flagged):
    # Iterating through possible specificity values with a step size of 0.0001
    for specificity in [i/10000 for i in range(10000, 0, -1)]:
        prob = calculate_prob_flagged_given_positive(
            sensitivity, specificity, p_flagged)
        if prob > target_prob:
            return specificity
    return 0


# Given values
sensitivity = 0.99
specificity = 0.99
p_flagged = 0.005

# Calculate P(Flagged|+)
prob = calculate_prob_flagged_given_positive(
    sensitivity, specificity, p_flagged)
print(f"P(Flagged|+) with given values: {prob:.4f}")

# Estimate required specificity
target_prob = 0.45
required_specificity = find_required_specificity(
    target_prob, sensitivity, p_flagged)
print(
    f"Required specificity to achieve P(Flagged|+) > {target_prob}: {required_specificity:.4f}")
