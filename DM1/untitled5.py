#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:35:10 2024

@author: khushipatel
"""

import pandas as pd
import networkx as nx

# Load the dataset (replace 'file_path' with your dataset path)
edges = pd.read_csv('/Users/khushipatel/Desktop/facebook_combined.txt', sep=' ', header=None, names=['source', 'target'])
print(edges.head())

# Create a graph from the edge list
G = nx.from_pandas_edgelist(edges, source='source', target='target')
print(f"Total Nodes: {G.number_of_nodes()}")
print(f"Total Edges: {G.number_of_edges()}")
