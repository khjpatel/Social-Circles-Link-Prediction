#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:53:52 2024

@author: khushipatel
"""

import numpy as np

# Covariance matrix (from previous step)
cov_matrix = np.array([[1.5, 1.5], [1.5, 1.5]])

# Find the eigenvalues and eigenvectors using numpy
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues in descending order and find the dominant one
idx = np.argsort(eigenvalues)[::-1]
dominant_eigenvalue_numpy = eigenvalues[idx[0]]
dominant_eigenvector_numpy = eigenvectors[:, idx[0]]

print("Dominant Eigenvalue (Numpy):", dominant_eigenvalue_numpy)
print("Dominant Eigenvector (Numpy):", dominant_eigenvector_numpy)
