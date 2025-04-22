#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 15:55:56 2024

@author: khushipatel
"""

import numpy as np
import pandas as pd

# Load the dataset (assuming you've downloaded and unzipped it)
url = 'https://archive.ics.uci.edu/dataset/42/glass+%20identification'
columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
data = pd.read_csv(url, header=None, names=columns)

# Drop the 'Id' and 'Class' columns as mentioned in the problem
data = data.drop(columns=['Id', 'Class'])

# Z-normalization: (value - mean) / std
z_normalized_data = (data - data.mean()) / data.std()

# Saving the z-normalized data to use in further questions
z_normalized_data.to_csv('z_normalized_glass_data.csv', index=False)
