import numpy as np


def feature_selection(num_samples):
    data1 = np.random.normal(1, 1, size=num_samples)
    data2 = np.random.normal(1.5, 1, size=num_samples)

    mean1 = np.mean(data1)
    mean2 = np.mean(data2)

    std1 = np.std(data1)
    std2 = np.std(data2)

    q = ((mean1-mean2)-0)/(((std1)**2/num_samples)+((std2)**2/num_samples))**0.5
    if(-1.967<q and q<1.967):
        print(f"number of sample is {num_samples}, q = {q}, so accpet the hypothesis that the mean of two datasets are the same")
    else:
        print(f"number of sample is {num_samples}, q = {q}, so reject the hypothesis that the mean of two datasets are the same")

feature_selection(20)
feature_selection(100)
feature_selection(600)
