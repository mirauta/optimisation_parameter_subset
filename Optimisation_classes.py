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

import  matplotlib.pyplot as plt 
import seaborn as sb
import timeit

from Additional_classes import *
from Target_Solution_classes import *

     
    
class FindSolution(object):

    def __init__(self, obj_target, target_function=None, colony_size=30, n_iter=5000, max_trials=100):

        self.colony_size = colony_size
        self.N_agents =int(self.colony_size)
        self.obj_target = obj_target
        if target_function is not None:
            self.obj_target.update_target_function(target_function)

        self.n_iter = n_iter
        self.max_trials = max_trials

        self.optimal_solution = None
        self.optimality_tracking = []
        self.optimal_solution_tracking = []

    def __initialize_agents(self,beta):
        self.solutions = [SolutionClass(self.obj_target,beta)for itr in range(self.N_agents)]
        self.best_solutions = [SolutionClass(self.obj_target,beta)for itr in range(self.N_agents)]
        self.hibrid_solution = SolutionClass(self.obj_target,beta)
        
    def get_covariance_groups(self,Iter,maxiter=10):
        try:
            self.solutions
        except:
            self.__initialize_agents(beta=None)

        N=self.solutions[0].beta.shape[0]
        values=pd.DataFrame(np.zeros((Iter*N+1,N)),columns=np.arange(N),index=np.arange(Iter*N+1))
        x00=optimize.minimize(self.obj_target.target_function,x0=np.random.randint(self.obj_target.minf,self.obj_target.maxf,size=N)).x.astype(float)
#        print(x00)
        values.iloc[0]=deepcopy(x00)
        for it in np.arange(0,Iter):
            x00=optimize.minimize(self.obj_target.target_function,x0=np.zeros(N)).x.astype(float)
            for i in np.arange(N):
                x0=deepcopy(x00)
                x0[i]=np.random.uniform(self.obj_target.minf,self.obj_target.maxf)
                values.iloc[it*N+i+1]=optimize.minimize(self.obj_target.target_function,x0=x0,options={"maxiter":maxiter}).x.astype(float)
                
        covariance=values
#        covariance=(covariance-covariance.mean(0))/covariance.std(0)
        
        Z=hierarchy.linkage(1-abs(covariance.corr()),method="ward")
        beta_groups=hierarchy.fcluster(Z,t=.9,criterion="distance")
        print ("set a better cut threshold")
        
        return [covariance, beta_groups]

        
    def __explore_combine_phase(self,solutions):
        [solution.propose_combine_friend(self.hibrid_solution) for solution in solutions]
        [solution.optimize_beta(solution.proposed) for solution in solutions]
        [solution.update_solution() for solution in solutions]

    def __explore_mean_phase(self,solutions):
        [solution.propose_mean_friend(Friendsolution=np.random.choice(self.solutions)) for solution in solutions]
        [solution.optimize_beta(solution.proposed) for solution in solutions]
        [solution.update_solution() for solution in solutions]

    def __optimize_phase(self,solutions):

        [solution.optimize_beta(solution.beta) for solution in solutions]
        [solution.update_solution() for solution in solutions]
        
    def __get_solution_probabilities(self):

        sum_fitness = sum([solution.fitness for solution in self.solutions])
        for solution in self.solutions:solution.prob = solution.fitness / sum_fitness

    def __select_hibrid_solutions(self,solutions):
        betas=np.array([solution.beta for solution in solutions]).T
        self.hibrid_solution.beta= [np.random.choice(i) for i in betas]
        
    def __select_best_solutions(self,solutions):
        self.__get_solution_probabilities()

        self.best_solutions =   np.hstack([np.repeat(solution, int(solution.prob*self.N_agents)) for solution in solutions])

    def __reset_phase(self,beta):
        for i,solution in enumerate(self.solutions):
            solution.reset_beta(self.max_trials,beta)

    def __update_optimal_solution(self):

        n_optimal_solution =  max(self.solutions,  key=lambda solution: solution.fitness)
        if not self.optimal_solution:
            self.optimal_solution = deepcopy(n_optimal_solution)
            
        else:
            if n_optimal_solution.fitness > self.optimal_solution.fitness:
                self.optimal_solution = deepcopy(n_optimal_solution)
        self.optimality_tracking.append(self.optimal_solution.fitness)
        self.optimal_solution_tracking.append(self.optimal_solution)

    def run(self,betainit):

        self.__initialize_agents(betainit)
        for i,sol in enumerate(self.solutions):
            sol.set_id(i)
        
        for itr in range(self.n_iter):

            self.__explore_mean_phase(self.solutions)   
            self.__update_optimal_solution()

            
            if itr%5000==0:
#                print ("get best solution")
                self.__select_hibrid_solutions(self.solutions)
                self.__explore_combine_phase(self.solutions)   
                self.__update_optimal_solution()
            
            self.__reset_phase(beta=None)
                
#            self.__explore_phase(self.best_solutions) 
#            self.__update_optimal_solution()
#            self.__reset_phase()

class FindSolution_blocks(object):
    def __init__(self, obj_target, Nparam,Nsolutions,independent_blocks_flag):

        self.independent_blocks_flag=independent_blocks_flag
        self.obj_target=obj_target
        self.Nsolutions=Nsolutions
        _doc="  Step 1: Initialize a global solutions object\
                Step 2: Determine parameter covariance structure "
        self.globalsolution = FindSolution(obj_target=obj_target, colony_size=1, n_iter=0, max_trials=0 )
        self.bestglobalsol = [SolutionClass(self.globalsolution.obj_target)for itr in range(self.Nsolutions)]

        if self.independent_blocks_flag:
            self.beta_optim_sol,self.beta_groups = self.globalsolution.get_covariance_groups(30,maxiter=100)
            
        else:
            temp=optimize.minimize(obj_target.target_function,x0=np.random.randint(-500,500,size=Nparam)).x
            self.beta_optim_sol=pd.DataFrame(np.repeat(temp,self.Nsolutions).reshape(temp.shape[0],-1).T)
            self.beta_groups=np.ones(Nparam)
#            obj_target.target_function_subset=obj_target.target_function

        self.unique_beta_groups = np.unique(self.beta_groups)

        _doc+="\n  Step 3: Set best global to initial solutions"
        for itr in range(self.Nsolutions):
            self.bestglobalsol[itr].beta = np.array(self.beta_optim_sol.iloc[-itr])
            self.bestglobalsol[itr].fitness = self.globalsolution.obj_target.get_fitness(self.bestglobalsol[itr].beta)
            self.bestglobalsol[itr].history = 0
            
            
    def run(self,Niter,max_trials,colony_size ):
        print("optimizing parameters in "+str(self.unique_beta_groups.shape[0])+" blocks")
        for ig,g in enumerate(self.unique_beta_groups):
            index=np.where(self.beta_groups==g)[0]
            print (index)
            for it in np.arange(Niter):  
                index_bestglobalsol=np.random.randint(self.Nsolutions)
                print(it*self.unique_beta_groups.shape[0]+ig)
                beta0=deepcopy(self.bestglobalsol[index_bestglobalsol].beta)
                if self.independent_blocks_flag: self.local_target_function=self.obj_target.target_function_subset(index,beta0=beta0)
                else: self.local_target_function=self.obj_target.target_function
                self.local_obj_target=TargetClass( dim=index.shape[0], minf=self.obj_target.minf, maxf=self.obj_target.maxf,target_function=self.local_target_function)
                solset = FindSolution(obj_target=self.local_obj_target, colony_size=colony_size, n_iter=Niter, max_trials=max_trials )
                solset.run(betainit=None)
#                print (optimize.minimize(self.local_obj_target.target_function,x0 = index*0+100))
                beta0[index]=solset.optimal_solution.beta
                beta0_fitness=self.globalsolution.obj_target.get_fitness(beta0)
                
                if beta0_fitness>self.bestglobalsol[index_bestglobalsol].fitness:
                    self.bestglobalsol[index_bestglobalsol].fitness=beta0_fitness
                    self.bestglobalsol[index_bestglobalsol].beta=beta0
                    self.bestglobalsol[index_bestglobalsol].history=it*self.unique_beta_groups.shape[0]+ig
