#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mirauta
"""
from Optimisation_classes import *
import inspect

from autograd import grad, jacobian
import numpy as np

# import matplotlib.pyplot as plt

# import tensorflow as tf
print (inspect.currentframe().f_code.co_filename)
_doc="Descriptions of the steps followed in this script \n"


">>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
fileNo = inspect.currentframe().f_lineno;
_doc+="Define test functions (line"+str(fileNo)+")\n"



Nparam=5
beta_syntetic=np.random.uniform(size=Nparam); beta_syntetic[:-3]=420
xx=np.random.normal(size=100)+4
xx1=np.random.normal(size=100)+4
y0=xx**1*beta_syntetic[-3]+xx**2*beta_syntetic[-2]+beta_syntetic[-1]*xx1


def sphere(beta):return ((y0-np.dot(beta.reshape(Nparam,1),xx.reshape(1,-1)))**2).mean()

def Schwefel(beta): 
    return  abs(sum(-beta*np.sin(np.sqrt(abs(beta))))+418.9829*beta.shape[0])

def mixedSchwefel(beta): 
    return  abs(sum(-beta[:-3]*np.sin(np.sqrt(abs(beta[:-3]))))+418.9829*beta[:-3].shape[0])+\
            ((xx**1*beta[-3]+xx**2*beta[-2]+beta[-1]*xx1-y0)**2).sum()
def square(beta): 
    return  (beta**2).sum()



target_function=mixedSchwefel


def get_param_correlation(target_function,beta):
    def sens(beta,SENS=100):
        epsilon= np.diag(np.repeat(1.0/SENS,Nparam))
        return np.array([SENS*(target_function(beta+epsilon[j])-target_function(beta))/target_function(beta) for j in np.arange(Nparam)])
    
    sens1=sens(beta,SENS=1000)
    mat=np.linalg.inv(np.dot(sens1[:,None],sens1[None])+np.diag(np.repeat(0.001,Nparam)))
    mat=mat/(np.dot(np.diag(mat).reshape(Nparam,-1),np.diag(mat).reshape(Nparam,-1).T)**.5)
    return mat

Iter=3
mat=np.zeros((Iter,Nparam,Nparam))
for j in np.arange(Iter):
    mat[j]=get_param_correlation(target_function, np.random.normal(beta_syntetic, scale=beta_syntetic/2))
    
for j in np.arange(Iter):
    sb.heatmap(mat[1][::-1])
    plt.show()
    
    
    
    
''' pairwise differentiation'''
def square(x):return np.sum(x**2)

def differentiate(beta,target_function=target_function,SENS=100):
    Nparam=beta.shape[0]
    epsilon= np.diag(np.repeat(1.0/SENS,Nparam))
    return SENS*np.array([target_function(beta+epsilon[j])-target_function(beta) for j in np.arange(Nparam)])

target_function=mixedSchwefel
target=TargetClass(target_function, dim=Nparam, minf=-100, maxf=100)
betainit=np.hstack([optimize.minimize_scalar(target.target_function_subset([i],np.zeros(Nparam))).x for i in np.arange(Nparam)])
print (betainit)
beta_est=optimize.minimize(target_function,x0=betainit).x
print (beta_est)
beta_est=optimize.minimize(target_function,x0=np.zeros(Nparam),method='L-BFGS-B').x
print (beta_est)
print (beta_syntetic)
sys.exit()

epsilon= np.diag(np.repeat(1.0,Nparam))
SENS=1000
sens1=differentiate(beta_est,target_function=mixedSchwefel,SENS=SENS)
print (sens1)

sens1=differentiate(beta_syntetic+2000.0/SENS,target_function=mixedSchwefel,SENS=SENS)
print (sens1)


plt.plot([target_function(beta_syntetic+epsilon[0]*e) for e in np.linspace(-2,2,100)])
    
sys.exit()
grad_fct = grad(sphere)

print ( grad_fct([beta_syntetic[0],beta_syntetic[0],beta_syntetic[0]]))
sphere(beta_syntetic+epsilon)

print (1)
grad_fct = jacobian(mixedSchwefel)
print ( grad_fct(beta_syntetic))
np.gradient([sphere], beta_syntetic)
print (2)
import numdifftools.nd_algopy as nda
import numdifftools as nd
fd = nda.Derivative(sphere)        # 1'st derivative

fd=nda.Jacobian(sphere)
fd(beta_syntetic)


np.allclose(fd(1), 2.7182818284590424)
True

#        
#bnds = list(zip(np.repeat(-500,Nparam),np.repeat(500,Nparam)))
#((0, None), (0, None))
#values=pd.DataFrame(np.zeros((Iter*N+1,N)),columns=np.arange(N),index=np.arange(Iter*N+1))
#
#target=TargetClass(dim=Nparam, minf=-500.0, maxf=500.0,target_function=target_function)
#x01=np.random.randint(-5,5,size=Nparam)
##print (x0[index])
#index=np.setdiff1d(np.arange(Nparam),np.array([]))
##
#
#print (x01[index])
#target2=TargetClass(dim=Nparam, minf=-500.0, maxf=500.0,target_function=target.target_function_subset(index=index,beta0=x01))
#
#optimize.minimize(target2.target_function,x0 = x01[index],options={"maxiter":10})
#
#
#print (optimize.minimize(abc.local_obj_target.target_function,x0 = [1,1,400]))
#
#print (beta_syntetic)
#


#
#optimize.minimize(target_function,x0=x0,bounds=bnds)
#
#x00=np.array(optimize.minimize(target_function,x0=x0,bounds=bnds).x.astype(float))
#print (x00)
#x01=np.array(optimize.minimize(fun,x0 = x0[index],bounds=np.array(bnds)[index]).x.astype(float))
#print (x01)
#x00[index]
#values.iloc[0]=deepcopy(x00)
#for it in np.arange(0,Iter):
#    x00=optimize.minimize(self.obj_target.target_function,x0=np.zeros(N)).x.astype(float)
#    for i in np.arange(N):
#        x0=deepcopy(x00)
#        x0[i]=np.random.uniform(self.obj_target.minf,self.obj_target.maxf)
#        values.iloc[it*N+i+1]=optimize.minimize(self.obj_target.target_function,x0=x0,options={"maxiter":maxiter}).x.astype(float)
#
#covariance=values
##        covariance=(covariance-covariance.mean(0))/covariance.std(0)
#
#Z=hierarchy.linkage(1-abs(covariance.corr()),method="ward")
#beta_groups=hierarchy.fcluster(Z,t=.9,criterion="distance")
#print ("set a better cut threshold")
#
#optimize.approx_fprime(beta_syntetic, target_function, 0.0001)
#
#fun=lambda x: (x**2+x*4).sum()
#derfun=lambda x: (x*2+4).sum()
#a=np.array([3]);eps=0.0001
#optimize.approx_fprime(a,fun , [eps])
#
#(fun(a+eps)-fun(a))/eps
#derfun(a)
#
#[lambda x: x**2 for x in np.arange(10)]

target_function=square

target_function=mixedSchwefel

global_obj_target=TargetClass(dim=Nparam, minf=-500.0, maxf=500.0,target_function=target_function)

beta0=beta_syntetic*0+1

optimfunction=optimize.minimize

x0=np.ones(beta0.shape[0]).astype(float)+450
NNIt=20
x01=np.zeros(NNIt)
t01=np.zeros(NNIt)
x011=np.zeros(NNIt)
t011=np.zeros(NNIt)

for j in np.arange(NNIt):
    x0=np.random.uniform(300,500,Nparam)
    
    start = timeit.default_timer()
    for i in np.arange(100):    
        for index in np.arange(Nparam)*0:
            fun=global_obj_target.target_function_subset(index,beta0)
            x01[j]=np.array(optimfunction(fun,x0 = x0[index]).x)
    t01[j]= timeit.default_timer()-start


    fun=global_obj_target.target_function
    start = timeit.default_timer()
    for i in np.arange(100):
        x011[j]=np.array(optimfunction(fun,x0 = x0).x)[0]
    t011[j]= timeit.default_timer()-start

plt.hist(t01,density=1);sb.kdeplot(t01)
plt.hist(t011,density=1,label="optim");sb.kdeplot(t011)
plt.legend(loc=2)
plt.show()

plt.plot(x01,x011-420,'o')
plt.legend(loc=2)
plt.show()

sys.exit()


">>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
fileNo = inspect.currentframe().f_lineno;
_doc+="Defina and run model (line"+str(fileNo)+")\n"

global_obj_target=TargetClass(dim=Nparam, minf=-500.0, maxf=500.0,target_function=target_function)
abc=FindSolution_blocks(obj_target=global_obj_target , Nparam=Nparam, Nsolutions=4,independent_blocks_flag=1 )
abc.run(Niter=11,colony_size=20, max_trials=20)

global_obj_target.target

">>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
_doc+="Evaluate results (line"+str(inspect.currentframe().f_lineno)+")\n"

abc.bestglobalsol[1].beta
abc.bestglobalsol[1].history
print ("\n\n original:\n")
print (beta_syntetic)
print (target_function(beta_syntetic))
print ("\n\n results:\n")
for itr in np.arange(abc.Nsolutions):
    print (abc.bestglobalsol[itr].history)
    print (target_function(abc.bestglobalsol[itr].beta))
    print (np.around(abc.bestglobalsol[itr].beta,3))

print ("<<< END results optimisation:\n")

_doc+="Evaluate performance for a multiple starting point optimisation algorithm (line"+str(inspect.currentframe().f_lineno)+")\n"
otim=[optimize.minimize(target_function,x0=np.random.randint(-500,500,size=beta_syntetic.shape[0])) for i in np.arange(400)]
otim_solution=otim[np.argmin([aa.fun for aa in otim])]
print (np.around(otim_solution.x,1))
print (beta_syntetic)
#stop = timeit.default_timer()
#print('Time: ', stop - start)  


#targ=target(target_function=target_function,N=N)
#start = timeit.default_timer()
#for g in np.unique(beta_groups):
#    index=np.where(beta_groups==g)[0]
#    print (index)
#
#    otim=[optimize.minimize(targ.target_subset(index,beta0=np.arange(N)),x0=np.random.randint(-500,500,size=index.shape[0]))for i in np.arange(200)]
#otim_solution=otim[np.argmin([aa.fun for aa in otim])]
#print (np.around(otim_solution.x,1))
#stop = timeit.default_timer()
#print('Time: ', stop - start)  


sys.exit()

abc.run(beta_groups)
stop = timeit.default_timer()
print('Time: ', stop - start) 

print (">> optimal solution >>>\n")
print (abc.optimal_solution.beta)
print (otim_solution.x)
sys.exit()
