import numpy as np

# Read the text file
with open('data.txt', 'r') as file:
    data = file.readlines()

# Split the rows and convert them into lists of integers
row1 = list(map(int, data[0].split()))
row2 = list(map(int, data[1].split()))

# Determine the number of columns based on the length of the longest row
num_columns = max(len(row1), len(row2))

# Create a sparse matrix using numpy
sparse_matrix = np.zeros((2, num_columns), dtype=int)

# Fill the sparse matrix with the data from the rows
for i, value in enumerate(row1):
    sparse_matrix[0, i] = value

for i, value in enumerate(row2):
    sparse_matrix[1, i] = value

print(sparse_matrix)
