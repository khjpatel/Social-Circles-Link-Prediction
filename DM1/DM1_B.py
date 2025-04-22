#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:50:02 2024

@author: khushipatel
"""

import numpy as np

# Z-normalized data (can be from the previous step)
z_normalized_data = np.array([[-1.2247, -1.2247], [0., 0.], [1.2247, 1.2247]])

# Number of data points
n = z_normalized_data.shape[0]

# Mean-centered data (already centered since it is z-normalized)
mean_centered_data = z_normalized_data

# Compute the covariance matrix manually
cov_matrix_manual = np.zeros((z_normalized_data.shape[1], z_normalized_data.shape[1]))

for i in range(n):
    cov_matrix_manual += np.outer(mean_centered_data[i], mean_centered_data[i])

# Normalize by n
cov_matrix_manual /= n

# Verify using numpy's built-in covariance function
cov_matrix_numpy = np.cov(z_normalized_data, rowvar=False, bias=True)

print("Manual Covariance Matrix:")
print(cov_matrix_manual)

print("\nNumpy Covariance Matrix:")
print(cov_matrix_numpy)
