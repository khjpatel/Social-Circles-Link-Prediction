#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 20:46:28 2024

@author: khushipatel
"""

import numpy as np

# PCA subroutine to preserve desired variance
def pca(X, variance_threshold=0.95):
    # Compute covariance matrix
    cov_matrix = np.cov(X, rowvar=False)
    
    # Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # Compute cumulative explained variance
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find the number of components required to preserve 95% variance
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Select those principal components
    principal_components = eigenvectors[:, :num_components]
    
    return principal_components, num_components

# Sample dataset
data = np.array([[1, 2], [3, 4], [5, 6]])

# Z-normalization
mean = data.mean(axis=0)
std = data.std(axis=0)
z_normalized_data = (data - mean) / std

# Perform PCA to preserve 95% variance
principal_components, num_components = pca(z_normalized_data, variance_threshold=0.95)

# Print the coordinates of the first 10 data points in the new basis
projected_data = np.dot(z_normalized_data, principal_components)
print("First 10 data points in the new basis:\n", projected_data[:10])
