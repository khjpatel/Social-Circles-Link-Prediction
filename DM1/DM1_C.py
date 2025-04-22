#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:53:52 2024

@author: khushipatel
"""

import numpy as np

# Covariance matrix (from the previous step)
cov_matrix = np.array([[1.5, 1.5], [1.5, 1.5]])

def power_iteration(matrix, num_simulations=1000, tol=1e-6):
    # Random starting vector
    b_k = np.random.rand(matrix.shape[1])

    for _ in range(num_simulations):
        # Matrix-vector multiplication
        b_k1 = np.dot(matrix, b_k)

        # Find the index of the largest absolute element in the vector
        max_index = np.argmax(np.abs(b_k1))

        # Rescale vector by the maximum element
        b_k1 = b_k1 / b_k1[max_index]

        # Check for convergence
        if np.linalg.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    # Compute the eigenvalue as the Rayleigh quotient
    eigenvalue = np.dot(b_k.T, np.dot(matrix, b_k)) / np.dot(b_k.T, b_k)

    # Normalize the final eigenvector to have unit length
    eigenvector = b_k / np.linalg.norm(b_k)

    return eigenvalue, eigenvector

# Apply power iteration to find the dominant eigenvalue and eigenvector
dominant_eigenvalue, dominant_eigenvector = power_iteration(cov_matrix)

print("Dominant Eigenvalue (Power Iteration):", dominant_eigenvalue)
print("Dominant Eigenvector (Power Iteration):", dominant_eigenvector)
