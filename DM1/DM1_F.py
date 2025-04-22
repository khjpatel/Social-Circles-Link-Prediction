#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:55:25 2024

@author: khushipatel
"""

import numpy as np

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

# Reconstruct Σ using UΛU^T (eigen-decomposition form)
reconstructed_cov_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

print("Original Covariance Matrix:\n", cov_matrix)
print("\nReconstructed Covariance Matrix (UΛU^T):\n", reconstructed_cov_matrix)
