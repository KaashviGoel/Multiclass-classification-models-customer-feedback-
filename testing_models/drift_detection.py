### COMP9417 PROJECT ###
# Completed 28.04.25

# Ansh Patel 	                z5478730
# Aravind Nadadur Rangarajan 	z5480898
# Tanvi Multani 	            z5621319
# Kaashvi Goel 	                z5623123
# Ash Lance 	                z5422160

# Tool to detect and visualise data drift

### IMPORTS ###
import pandas as pd

import numpy as np

from sklearn import datasets

from scipy.stats import ks_2samp

data1 = pd.read_csv('y_train.csv')
data2 = pd.read_csv('y_test_2_reduced.csv')

data1.sort_values(by='label')
data2.sort_values(by='label')

# calculate Kolmogorov-Smirnov test to analyse y_test_2 results
result = ks_2samp(data1, data2, alternative='two-sided', axis=0)

print(
    "## y_values ##\n"
    "Statistic:", result[0],
    "p-value:", result[1]
)

# result: Statistic is 0.44759109, p-value is 1.1497e-36
# p-value is extremely small. as the null-hypothesis test was specified to be
# two-sided (i.e., both were drawn from the same set), we reject the hypothesis
# that these these results were drawn from the same distribution.
