#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mirauta
"""

import sys

import numpy as np
import pandas as pd
from copy import deepcopy

from scipy import optimize
from scipy.cluster import hierarchy
#from sklearn.cluster import AffinityPropagation
#import sklearn      
import timeit

import  matplotlib.pyplot as plt 
import seaborn as sb


def plot_solution(abc):
    
    sb.heatmap(abs(abc.covariance.corr()))
    plt.show()
    Z=hierarchy.linkage(1-abs(abc.covariance.corr()),method="ward")
    hierarchy.dendrogram(Z)
    plt.show()
    
    plt.figure(figsize=(8,8))
    plt.subplot(2,2,1)
    plt.plot(beta,abc.optimal_solution.beta,'o')
    plt.subplot(2,2,2)
    plt.plot(np.vstack(list(map(lambda sol:sol.fitness, abc.optimal_solution_tracking))),'o')
    plt.subplot(2,2,3)
    plt.plot(np.vstack(list(map(lambda sol:sol.beta[0], abc.optimal_solution_tracking))),'o')
    plt.plot(np.vstack(list(map(lambda sol:sol.beta[1], abc.optimal_solution_tracking))),'o')
    plt.show()
    print (beta)
    print (abc.optimal_solution.beta)
    print("\n")