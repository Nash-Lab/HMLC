#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:00:18 2024

@author: vdoffini
"""

import matplotlib.pyplot as plt
import numpy as np
from os.path import isfile
from scipy.optimize import minimize_scalar
from scipy.special import gamma,betainc
from scipy.special import binom as binom_coef
from scipy.stats import binom as binom_dist
import pandas as pd


#%% Functions

def _real_n_mut_float(n,p,voc_size=20):
    oo=0
    for pp in np.arange(p,-1,-1):
        oo+=gamma(n+1)/(gamma(n-pp+1)*gamma(pp+1))*(voc_size-1)**pp
        pp-=1
    return oo

def p_binom(x,n,v,p_wt):
    p=1-p_wt
    n_var_mut = np.array([binom_coef(n,k)*(v-1)**k for k in range(n+1)]).astype(int)
    out = []
    for k in range(n+1):
        out.append([binom_dist.pmf(k, n, p)/n_var_mut[k]]*n_var_mut[k])
    out = np.concatenate(out)
    return out/np.sum(out)

def p_exp(x,n,v,p_wt):
    p=1-p_wt
    x1 = x[:,1]
    x2 = x[:,2]
    n_var_mut = np.array([binom_coef(n,k)*(v-1)**k for k in range(n+1)]).astype(int)
    p_var_mut = np.array([binom_dist.pmf(k, n, p) for k in range(n+1)])
    out = binom_coef(n,x1)*p**x1*(1-p)**(n-x1)
    for k in range(n_var_mut.size):
        out[x2==k] = out[x2==k]/out[x2==k].sum()
        out[x2==k] = out[x2==k]*p_var_mut[k]
        out[x2==k] = np.sort(out[x2==k])[::-1]
    return out/np.sum(out)
    
def h_disc(p):
    p_temp = p[p>0]
    return np.sum(-p_temp*np.log(p_temp))


#%% Parameters

data_path = '../Data/'
data_name_fitness = 'EvoEF.csv'
data_name_mutations = 'mut.csv'

save_path = '../Results/'
file_out_name = '03_pop_distr'

bool_plot = False

bool_save_data = True


#%% Load Data

df_fitness = pd.read_csv(data_path+data_name_fitness,sep=' ',header=0,index_col=0)
df_temp = df_fitness.sequence.str.split('',expand=True).iloc[:,1:-1]

bool_load_df_mut = isfile(data_path+data_name_mutations)

if bool_load_df_mut:
    df_mut = pd.read_csv(data_path+data_name_mutations,header=[0,1],index_col=0)

#%% Calculated parameters

n = df_temp.shape[1]
voc = np.unique(df_temp.values.flatten())
v = len(voc)
n_variants = v**n

p_wts = np.linspace(0,1,101)
p_wts[0] += 1e-8
p_wts[-1] -= 1e-8

N_norms = np.linspace(0,n,101)


#%% Calculate df_mut (if bool_load_df_mut is False)

if not(bool_load_df_mut):

    f_temp = lambda x,i: np.abs(i-_real_n_mut_float(n,x,voc_size=v))

    t = []
    for i in range(n_variants):
        print(i+1)
        t1 = minimize_scalar(f_temp,bounds=[0,n],args=(i+1,))
        t2 = t1['x']
        t.append([i+1,t2])

    t = np.array(t)
    t[0,1] = 0
    t[0,2] = 0
    t[-1,1] = n
    
    t3 = []
    n_var_mut = np.array([binom_coef(n,k)*(v-1)**k for k in range(n+1)]).astype(int)
    for i,c in enumerate(n_var_mut):
        t3.append(c*[i])
    t3 = np.concatenate(t3)
    
    t = np.column_stack((t,t3))
    
    df_mut = pd.DataFrame(t,columns = ['N','N_norm','n_mutations'])    
    
    df_uni = pd.DataFrame(np.array([1/n_variants]*n_variants),columns = [0])
    df_bin = pd.DataFrame(np.array([p_binom(df_mut.values,n,v,p_wt) for p_wt in p_wts]).T,columns = p_wts)
    df_exp = pd.DataFrame(np.array([p_exp(df_mut.values,n,v,p_wt) for p_wt in p_wts]).T,columns = p_wts)
    
    df_mut = df_mut.convert_dtypes()

    d = {'Info' : df_mut, 'Uni' : df_uni, 'Bin' : df_bin, 'Exp' : df_exp}
    df_mut = pd.concat(d.values(), axis=1, keys=d.keys())
    df_mut.to_csv(data_path+data_name_mutations)


#%% Entropies

P_WTs,N_NORMs = np.meshgrid(p_wts,N_norms)
h_bin = np.concatenate([np.apply_along_axis(h_disc,axis=0,arr=df_mut.Bin.loc[df_mut.Info.query(f'N_norm<={i}').index,:]).reshape(1,-1) for i in N_norms],axis=0)
h_exp = np.concatenate([np.apply_along_axis(h_disc,axis=0,arr=df_mut.Exp.loc[df_mut.Info.query(f'N_norm<={i}').index,:]).reshape(1,-1) for i in N_norms],axis=0)


#%% Optimal p_wt

p_wts_opt_bin = []
p_wts_opt_exp = []
for i in N_norms:
    df_temp = df_mut.Info
    p_wts_opt_bin.append(minimize_scalar(lambda x: -h_disc(p_binom(df_temp.values,n,v,x)[:df_temp.query(f'N_norm<={i}').shape[0]]),bounds=[1e-8,1-1e-8])['x'])
    p_wts_opt_exp.append(minimize_scalar(lambda x: -h_disc(p_exp(df_temp.values,n,v,x)[:df_temp.query(f'N_norm<={i}').shape[0]]),bounds=[1e-8,1-1e-8])['x'])
    print(i)
p_wts_opt_bin = np.array(p_wts_opt_bin)
p_wts_opt_exp = np.array(p_wts_opt_exp)

p_wts_opt_3mut_bin = minimize_scalar(lambda x: -h_disc(p_binom(df_temp.values,n,v,x)[:df_temp.query(f'n_mutations<=3').shape[0]]),bounds=[1e-8,1-1e-8])['x']
p_wts_opt_3mut_exp = minimize_scalar(lambda x: -h_disc(p_exp(df_temp.values,n,v,x)[:df_temp.query(f'n_mutations<=3').shape[0]]),bounds=[1e-8,1-1e-8])['x']

#%% Save

if bool_save_data:
    d_out = {}
    d_out['P_WTs'] = P_WTs
    d_out['N_NORMs'] = N_NORMs
    d_out['h_bin'] = h_bin
    d_out['h_exp'] = h_exp
    d_out['N_norms'] = N_norms
    d_out['p_wts'] = p_wts
    d_out['p_wts_opt_bin'] = p_wts_opt_bin
    d_out['p_wts_opt_exp'] = p_wts_opt_exp
    d_out['p_wts_opt_3mut_bin'] = np.array([p_wts_opt_3mut_bin])
    d_out['p_wts_opt_3mut_exp'] = np.array([p_wts_opt_3mut_exp])
    
    np.savez(save_path+file_out_name, **d_out)


#%% Plot
if bool_plot:
    x1,x2 = np.meshgrid(p_wts,df_mut.Info.N_norm.values)
    t = df_mut.Uni.values[0]/10
    t1 = df_mut.Bin.values
    t1[t1<t] = t
    plt.figure()
    plt.contourf(x1,x2,np.log10(t1),levels=100)
    plt.colorbar()

    t2 = df_mut.Exp.values
    t2[t2<t] = t
    plt.figure()
    plt.contourf(x1,x2,np.log10(t2),levels=100)
    plt.colorbar()
    
    plt.figure()
    plt.contourf(P_WTs,N_NORMs,h_bin,levels=100)
    plt.plot(p_wts_opt_bin,N_norms,'magenta')
    plt.colorbar()

    plt.figure()
    plt.contourf(P_WTs,N_NORMs,h_exp,levels=100)
    plt.plot(p_wts_opt_exp,N_norms,'magenta')
    plt.colorbar()
