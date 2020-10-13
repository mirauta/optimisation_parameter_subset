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


class TargetClass(object):
    """
        Class implementing methods related to the target function 
        and the target function applied on subsets of parameters
    
        Methods
        ---------
        target_function_subset:  creates a -subtarget function applied only 
        on the parameter subset determined by 'index'
        
    """

    def __init__(self, target_function, dim, minf, maxf):
        self.target_function = target_function
        self.dim = dim
        self.minf = minf
        self.maxf = maxf

    def update_target_function(self,target_function):
        self.target_function=target_function

    def sample(self):
        return np.random.uniform(low=self.minf, high=self.maxf, size=self.dim)

    def evaluate(self, beta):
        return self.target_function(beta)
    
    def get_fitness(self,beta):   
        objective_value=self.evaluate(beta)
        return 1 / (1 + objective_value) #if objective_value >= 0 else 1 + np.abs(objective_value)

    def target_function_subset(self,index,beta0):

        def subtarget(beta):
            beta0[index]=beta
            return self.target_function(beta0)
        return subtarget

class SolutionClass(object):
    """
    Class implementing methods related to the Solutions function 
    and the target function applied on subsets of parameters

    Parameters
    ---------
    beta :  float array
        Target parameters

    """

    def __init__(self, obj_target,beta=None):
        self.obj_target = obj_target
        
        self.minf = obj_target.minf
        self.maxf = obj_target.maxf
        
        self.trial = 0
        self.prob = 0
        self.test=0

        if beta is None: self.beta = obj_target.sample()
        else: self.beta=beta 
        self.track_pos=[] 
        self.objective = obj_target.evaluate(self.beta)
        self.fitness =self.obj_target.get_fitness(self.beta)

        self.proposed=None
        self.proposed_fitness=0
        
    def set_id(self,idsol):
        self.id = idsol
        
    def evaluate_boundaries(self, pos):
        if (pos < self.minf).any() or (pos > self.maxf).any():
            pos[pos > self.maxf] = self.maxf
            pos[pos < self.minf] = self.minf
        return pos

    def reset_beta(self, max_trials,beta):

        if self.trial >= max_trials:
            if beta is None: self.beta =self.obj_target.sample()
            else: self.beta=beta 
            self.fitness = self.obj_target.get_fitness(self.beta)
            self.trial = 0
            self.prob = 0

    def propose_combine_friend (self,  othersolution,max_trials=10):

        i=np.random.randint(self.beta.shape[0])
        self.proposed=deepcopy(self.beta)
        self.proposed[i]=othersolution.beta[np.random.randint(self.beta.shape[0])]
        
    def propose_mean_friend (self,  Friendsolution,max_trials=10):
        if self.trial >= max_trials: return
        i=np.random.randint(self.beta.shape[0])
        phi = np.random.uniform(low=-1, high=1, size=len(self.beta))

        self.proposed=deepcopy(self.beta)
        self.proposed[i]+=(self.beta[i] - Friendsolution.beta[i]) * phi[i]

        self.proposed = self.evaluate_boundaries(self.proposed)
 
    def optimize_beta (self,beta):
        
        self.optimized_beta=optimize.minimize(self.obj_target.evaluate,x0=beta,options={"maxiter":10}).x

        self.optimized_beta = self.evaluate_boundaries(self.optimized_beta)
        return [beta,self.proposed]
       
    def update_solution(self):

        self.proposed_fitness =self.obj_target.get_fitness(self.proposed)

        if self.proposed_fitness > self.fitness:
            self.beta = self.proposed
            self.fitness = self.proposed_fitness
            self.trial = 0
           
        else:
            self.trial += 1      
        self.track_pos.append(self.beta)
        
   