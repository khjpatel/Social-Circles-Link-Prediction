#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:53:52 2024

@author: khushipatel
"""

import numpy as np

# Sample dataset (can be replaced with the actual data)
data = np.array([[1, 2], [3, 4], [5, 6]])

# Z-normalization: (value - mean) / std
mean = data.mean(axis=0)
std = data.std(axis=0)
z_normalized_data = (data - mean) / std

# Compute the covariance matrix
cov_matrix = np.cov(z_normalized_data, rowvar=False)

# Use linalg.eig to get eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

# Select the first two dominant eigenvectors
dominant_eigenvectors = eigenvectors[:, :2]

# Project data onto the new subspace
projected_data = np.dot(z_normalized_data, dominant_eigenvectors)

# Compute variance of the projected data (sum of variances in the subspace)
projected_variance = np.var(projected_data, axis=0).sum()

print("Variance in the projected subspace:", projected_variance)
