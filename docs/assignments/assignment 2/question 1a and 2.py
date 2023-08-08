original_image = [
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 3, 3, 3, 3, 3, 2, 1, 1],
    [0, 0, 3, 3, 4, 4, 4, 4, 4, 4],
    [0, 0, 3, 3, 3, 3, 4, 4, 4, 4],
    [0, 0, 0, 1, 1, 3, 4, 4, 4, 4],
    [0, 0, 0, 0, 0, 4, 4, 4, 1, 0],
    [5, 5, 0, 0, 0, 4, 4, 4, 0, 0],
    [5, 5, 0, 0, 0, 4, 4, 4, 0, 0],
    [5, 5, 0, 0, 0, 0, 5, 5, 0, 0],
    [5, 5, 0, 0, 0, 0, 5, 5, 0, 0]
]

haar_filter = [
    [-1, -1,  1,  1, -1, -1],
    [-1, -1,  1,  1, -1, -1],
    [-1, -1,  1,  1, -1, -1],
    [-1, -1,  1,  1, -1, -1],
    [-1, -1,  1,  1, -1, -1],
    [-1, -1,  1,  1, -1, -1],
]

# Convert the original image to an integral image
integral_image = [[0] * 10 for _ in range(10)]
for i in range(10):
    for j in range(10):
        integral_image[i][j] = original_image[i][j] + \
            (integral_image[i-1][j] if i > 0 else 0) + \
            (integral_image[i][j-1] if j > 0 else 0) - \
            (integral_image[i-1][j-1]
             if i > 0 and j > 0 else 0)

# Function to compute the sum of a rectangular area using the integral image


def compute_sum(x, y, width, height):
    A = integral_image[y-1][x-1] if y > 0 and x > 0 else 0
    B = integral_image[y-1][x+width-1] if y > 0 else 0
    C = integral_image[y+height-1][x-1] if x > 0 else 0
    D = integral_image[y+height-1][x+width-1]
    return D - B - C + A


# Apply the Haar filter using the integral image
result = [[0] * 5 for _ in range(5)]

for i in range(5):
    for j in range(5):
        result[i][j] = - compute_sum(j, i, 2, 6) + \
            compute_sum(j+2, i, 2, 6) - \
            compute_sum(j+4, i, 2, 6)
print(integral_image)
print("\n\n")
print("haar filter applied:")
# print the result to see the responses of the Haar filter
for row in result:
    print(row)
