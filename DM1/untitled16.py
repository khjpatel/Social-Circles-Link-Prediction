import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the dataset file
file_path = '/Users/khushipatel/Desktop/glass.data'

# Read the .data file into a DataFrame assuming it's comma-separated
glass_data = pd.read_csv(file_path, header=None)

# Assign column names
columns = ['ID', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
glass_data.columns = columns

# Extract class labels
class_labels = glass_data['Type']

# Drop ID and Type columns as we don't need them for normalization
data = glass_data.drop(columns=['ID', 'Type'])

# Apply z-normalization
def z_normalize(data):
    return (data - data.mean()) / data.std()

normalized_data = z_normalize(data)

# Center the data
centered_data = normalized_data - normalized_data.mean()

# Compute covariance matrix using numpy.cov
cov_matrix_numpy = np.cov(centered_data, rowvar=False, bias=True)

# PCA Algorithm Implementation
def pca(data, variance_threshold=0.95):
    # Compute covariance matrix
    cov_matrix = np.cov(data, rowvar=False, bias=True)
    
    # Perform eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    
    # Find the number of components needed to preserve the desired variance
    num_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
    
    # Return the top principal components
    top_eigenvectors = sorted_eigenvectors[:, :num_components]
    return top_eigenvectors, num_components

# Perform PCA and get the principal components that preserve 95% of variance
principal_components, num_components = pca(centered_data, variance_threshold=0.95)

# Project the data points onto the principal components
projected_data = np.dot(centered_data, principal_components)

# Reconstruct the data from the projection
reconstructed_data = np.dot(projected_data, principal_components.T)

# Compute the Mean Squared Error (MSE)
def compute_mse(original_data, reconstructed_data):
    return np.mean(np.sum((original_data - reconstructed_data) ** 2, axis=1))

# Compute MSE of the projection
mse_projected = compute_mse(centered_data, reconstructed_data)

# Calculate the variance of the projected data
variance_projected = np.mean(np.var(projected_data, axis=0))

# Reconstruct the covariance matrix from eigen-decomposition
Lambda = np.diag(np.linalg.eig(np.cov(centered_data, rowvar=False, bias=True))[0][np.argsort(np.linalg.eig(np.cov(centered_data, rowvar=False, bias=True))[0])[::-1]])  # Diagonal matrix of eigenvalues
U = np.linalg.eig(np.cov(centered_data, rowvar=False, bias=True))[1][:, np.argsort(np.linalg.eig(np.cov(centered_data, rowvar=False, bias=True))[0])[::-1]]  # Matrix of eigenvectors
reconstructed_cov_matrix = np.dot(U, np.dot(Lambda, U.T))

# Sum of the remaining eigenvalues (excluding the first two)
remaining_eigenvalues_sum = np.sum(np.linalg.eig(np.cov(centered_data, rowvar=False, bias=True))[0]) - np.sum(np.linalg.eig(np.cov(centered_data, rowvar=False, bias=True))[0][:num_components])

# Create an output file to store results
output_file = '/Users/khushipatel/Desktop/DM1/output_results.txt'

with open(output_file, 'w') as file:
    file.write("First few rows of the normalized data:\n")
    file.write(normalized_data.head().to_string())
    file.write("\n\nCovariance matrix using numpy.cov:\n")
    file.write(pd.DataFrame(cov_matrix_numpy).to_string())
    file.write("\n\nVariance of the projected data in the subspace spanned by the principal components:\n")
    file.write(str(variance_projected))
    file.write("\n\nReconstructed covariance matrix (UΛU^T):\n")
    file.write(pd.DataFrame(reconstructed_cov_matrix).to_string())
    file.write("\n\nMean Squared Error of the projection onto the principal components:\n")
    file.write(str(mse_projected))
    file.write("\n\nSum of the remaining eigenvalues (except the first few):\n")
    file.write(str(remaining_eigenvalues_sum))

print(f"Results have been written to {output_file}")
