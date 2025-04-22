#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:56:00 2024

@author: khushipatel
"""

import numpy as np

# Define a subroutine for MSE
def compute_mse(original_data, projected_data):
    mse = np.mean(np.sum((original_data - projected_data) ** 2, axis=1))
    return mse

# Sample dataset
data = np.array([[1, 2], [3, 4], [5, 6]])

# Z-normalization
mean = data.mean(axis=0)
std = data.std(axis=0)
z_normalized_data = (data - mean) / std

# Compute covariance matrix
cov_matrix = np.cov(z_normalized_data, rowvar=False)

# Eigen-decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Select the first two dominant eigenvectors and project data
dominant_eigenvectors = eigenvectors[:, :2]
projected_data = np.dot(z_normalized_data, dominant_eigenvectors) @ dominant_eigenvectors.T

# Compute MSE
mse_value = compute_mse(z_normalized_data, projected_data)

# Show that MSE equals the sum of the eigenvalues except the first two
remaining_variance = np.sum(eigenvalues[2:])

print("MSE Value:", mse_value)
print("Sum of remaining eigenvalues (except first two):", remaining_variance)
