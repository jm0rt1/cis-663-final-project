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
