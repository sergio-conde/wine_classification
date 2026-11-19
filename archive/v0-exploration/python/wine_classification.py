# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:28:57 2026

@author: scond
"""

# This script reads a csv file containing the chemical composition of different 
# wines.

#    -- The attributes are (dontated by Riccardo Leardi, 
# 	riclea@anchem.unige.it )
#  	1) Alcohol
#  	2) Malic acid
#  	3) Ash
# 	4) Alcalinity of ash  
#  	5) Magnesium
# 	6) Total phenols
#  	7) Flavanoids
#  	8) Nonflavanoid phenols
#  	9) Proanthocyanins
# 	10)Color intensity
#  	11)Hue
#  	12)OD280/OD315 of diluted wines
#  	13)Proline 

import pandas as pd

wine_file = r"C:\Work\GitHub\wine_classification\example_data\wine.data"
column_names = [
    "cultivar",
    "alcohol",
    "malic_acid",
    "ash",
    "ash_alcalinity",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od_ratio",
    "proline",
]
wines = pd.read_csv(wine_file, header=None, names=column_names)

wines.iloc[:,1:14].sum(axis=1)
# 
# Process not as student like tutorial, think about cool stuff to do with this
# Descriptive statistics per component
# Normalizing data manually and compare that to sklearn
# What other toolboxes are there?
# Check how the distributions changed after normalization

# Principal component analysis
# Plot cumulative explained variance
# kmeans para clusters

# LDA

# transforming wines between cultvars: trajectories?
