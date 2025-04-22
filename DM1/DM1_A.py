#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:49:54 2024

@author: khushipatel
"""

import numpy as np

# Sample data (2D points as an example)
data = np.array([[1, 2], [3, 4], [5, 6]])

# Z-normalization: (value - mean) / std
mean = data.mean(axis=0)
std = data.std(axis=0)

z_normalized_data = (data - mean) / std

print("Z-Normalized Data:")
print(z_normalized_data)
