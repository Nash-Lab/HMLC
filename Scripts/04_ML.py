#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:17:14 2024

@author: vdoffini
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from scipy.special import gamma
from scipy.special import binom as binom_coef
from scipy.stats import norm,rv_discrete,rv_histogram
from scipy.stats import binom as binom_dist
from time import time
t0 = time()

d_inputs_and_parameters = {}
d_out = {}

#%% Argparser

parser = argparse.ArgumentParser()
# parser.add_argument("--randomisation_valid_test", type=bool, help="mutagenesis or random sampling of validation and test sets, default False",
#                     action="store",default=False)
parser.add_argument("-c","--coordinates",nargs='+', type=int, help="Coordinates to use (flatten array).",
                    action="store",default=[0])
parser.add_argument("--coordinates_step", type=int, help="Coordinates step used. data_coordinates[::coordinates_step][coordinates] or data_coordinates[::coordinates_step,:][:,::coordinates_step].flatten()[coordinates]",
                                        action="store",default=1)
parser.add_argument("--n_mut_train_max", type=int, help="Number of maximum mutations contained in the training set.",
                                        action="store",default=1)
parser.add_argument("--n_lc", type=int, help="number of training points between different mutants used for the learning curves. Minimum: 1.",
                                        action="store",default=10)
parser.add_argument("--n_rep_shuffle", type=int, help="Shuffled used. mode_rep controls if range(n_rep_shuffle) or [n_rep_shuffle] is used.",
                                        action="store",default=0)
parser.add_argument("--n_rep_error_calc", type=int, help="Sampling used. mode_rep controls if range(n_rep_error_calc) or [n_rep_error_calc] is used.",
                                        action="store",default=0)
parser.add_argument("--mode_points", type=str, help="Which mode should be used to select the datapoints (Ns and HSs). Possible values: 2D, n_crit, h_crit, contour_crit_equidistant, eq_equidistant.",
                    action="store",default='2D')
parser.add_argument("-d","--name_dist", type=str, help="Distribution Name. Possible values: 'uni' (uniform), 'bin' (binomial) or 'exp' (experimental).",
                    action="store",default='uni')
parser.add_argument("-s","--save_path", type=str, help="Path where to save.",
                    action="store",default='../Results/')
args = parser.parse_args()


#%% Inputs from argparser
'''
ToDo:
    ~ Argparrser
'''

data_path = '../Data/'
d_inputs_and_parameters['data_path'] = data_path

data_mut_name = 'mut.csv'
d_inputs_and_parameters['data_mut_name'] = data_mut_name

data_fitness_name = 'EvoEF.csv'
d_inputs_and_parameters['data_fitness_name'] = data_fitness_name

data_coordinates_path_name = '../Results/02_bias_vs_variance_error_sqrt2_pi_sigma.npy'
d_inputs_and_parameters['data_coordinates_path_name'] = data_coordinates_path_name

save_path = args.save_path
d_inputs_and_parameters['save_path'] = save_path

# N = 1#10**4.575008019678023 # number of cells "sorted" per each variant
# hs = np.sqrt(6)/20
name_dist = args.name_dist
d_inputs_and_parameters['name_dist'] = name_dist

p_wt = 0.690
d_inputs_and_parameters['p_wt'] = p_wt


n_mut_train_max = args.n_mut_train_max
d_inputs_and_parameters['n_mut_train_max'] = n_mut_train_max

n_mut_valid = 4
d_inputs_and_parameters['n_mut_valid'] = n_mut_valid

n_mut_test = 5
d_inputs_and_parameters['n_mut_test'] = n_mut_test


n_rep_shuffle = args.n_rep_shuffle
d_inputs_and_parameters['n_rep_shuffle'] = n_rep_shuffle

n_rep_error_calc = args.n_rep_error_calc
d_inputs_and_parameters['n_rep_error_calc'] = n_rep_error_calc


coordinates = np.sort(args.coordinates).tolist()
d_inputs_and_parameters['coordinates'] = coordinates

coordinates_step = args.coordinates_step
d_inputs_and_parameters['coordinates_step'] = coordinates_step


mode = 'single'# single, multiple
d_inputs_and_parameters['mode'] = mode

mode_rep = 'single'# single, multiple
d_inputs_and_parameters['mode_rep'] = mode_rep

mode_points = args.mode_points# 2D, n_crit, h_crit, contour_crit_equidistant, eq_equidistant
d_inputs_and_parameters['mode_points'] = mode_points


sigma_y_sima_pop_ratio = 0.5
d_inputs_and_parameters['sigma_y_sima_pop_ratio'] = sigma_y_sima_pop_ratio


# for n_lc == 10 use 4 cpus (400s*n_rep_shuffle*n_rep_error_calc per run). (1 cpu -> ~650s; 2 cpus -> ~500s; 4 cpus -> ~400s; 25 cpus -> ~450s)
# for n_lc == 1 use 2 cpus (175s*n_rep_shuffle*n_rep_error_calc per run). (1 cpu -> ~230s; 2 cpus -> ~175s; 4 cpus -> ~150s; 25 cpus -> ~175s)
n_lc = args.n_lc
d_inputs_and_parameters['n_lc'] = n_lc


bool_plot = False
d_inputs_and_parameters['bool_plot'] = bool_plot

bool_save_data = True
d_inputs_and_parameters['bool_save_data'] = bool_save_data

bool_store_weights = False
d_inputs_and_parameters['bool_store_weights'] = bool_store_weights

#%% Parameters

sigma_s = np.logspace(-5,5,25)
d_inputs_and_parameters['sigma_s'] = sigma_s

lambda_s = np.logspace(-8,0,25)
d_inputs_and_parameters['lambda_s'] = lambda_s


#%% Calculated parameters


data_coordinates = np.load(data_coordinates_path_name,allow_pickle=True).flatten()[0]

if mode_points.upper() == '2D':
    t1 = data_coordinates['Ns'][::coordinates_step,:][:,::coordinates_step].reshape(-1,1)
    t2 = data_coordinates['HSs'][::coordinates_step,:][:,::coordinates_step].reshape(-1,1)
    data_coordinates = np.concatenate((t1,t2),axis=1)
else:
    try:
        data_coordinates = data_coordinates[f'var_{mode_points}'][:,:2][::coordinates_step,:]
    except:
        raise KeyError(f'var_{mode_points} does not exist in {data_coordinates_path_name}, check the file or the spelling of mode_points (current: {mode_points}; possible values: 2D, n_crit, h_crit, contour_crit_equidistant, eq_equidistant).')

len_data_coordinates = len(data_coordinates)

if  mode.lower() == 'single':
    data_coordinates = data_coordinates[coordinates]

if mode_rep.lower() == 'single':
    range_rep_shuffle = np.array([n_rep_shuffle])
    range_rep_error_calc = np.array([n_rep_error_calc])
else:
    range_rep_shuffle = np.arange(n_rep_shuffle)
    range_rep_error_calc = np.arange(n_rep_error_calc)

d_out['Ns'] = data_coordinates[:,:1].reshape(1,1,1,1,-1)
d_out['HSs'] = data_coordinates[:,1:2].reshape(1,1,1,1,-1)

#%% Functions

def calc_dist(x1,x2):
    return cdist(x1,x2,'cityblock')

def arr_from_df(df):
    return np.array(list(df['x'].values))

def error_fun(y_true,y_pred,axis=None):
    return np.mean(np.abs(y_pred-y_true),axis=axis)

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

#%% Import Data

df_mut = pd.read_csv(data_path+data_mut_name,header=[0,1],index_col=0)

df = pd.read_csv(data_path+data_fitness_name,sep=' ',header=0,index_col=0)
df['idx'] = np.arange(df.shape[0])
    
df_tr = df.query(f'n_mutations<={n_mut_train_max}')
df_va_te = df.query(f'n_mutations>{n_mut_train_max}')

#%% Calculate x (OHE flatten)

temp = np.array([list(i) for i in df.sequence.values])
n = temp.shape[1]
voc = np.unique(temp)
voc_len = len(voc)
x_ohe = np.zeros((*temp.shape,voc.size))
for i,s in enumerate(voc):
    x_ohe[temp==s,i]=1
x_ohe_flatten = x_ohe.reshape(x_ohe.shape[0],-1)

df['x'] = list(x_ohe_flatten)

#%% Calculate Ns_training (normalized and not normalized), N_validation and N_test

Ns_tr_norm = np.linspace(0,n_mut_train_max,int(np.round(n_lc*n_mut_train_max+1)))
d_out['Ns_tr_norm'] = Ns_tr_norm

Ns_tr = np.round([_real_n_mut_float(n,p,voc_len) for p in Ns_tr_norm]).astype(int)
d_out['Ns_tr'] = Ns_tr

N_va = df.query(f'n_mutations=={n_mut_valid}').shape[0]
d_out['N_va'] = N_va

N_te = df.query(f'n_mutations=={n_mut_test}').shape[0]
d_out['N_te'] = N_te


#%% Calculate distribution (only for training data)

if name_dist.lower() == 'uni':
    dist_population = rv_discrete(values=(np.arange(df_tr.shape[0]),[1/df_tr.shape[0]]*df_tr.shape[0]))
elif name_dist.lower() == 'bin':
    temp = p_binom(df_mut.Info.values,n,voc_len,p_wt)[:df_tr.shape[0]]
    temp = temp/temp.sum()
    dist_population = rv_discrete(values=(np.arange(df_tr.shape[0]),temp))
elif name_dist.lower() == 'exp':
    temp = p_exp(df_mut.Info.values,n,voc_len,p_wt)[:df_tr.shape[0]]
    temp = temp/temp.sum()
    dist_population = rv_discrete(values=(np.arange(df_tr.shape[0]),temp))
else:
    raise ValueError(f'name_dist shoud be "uni", "bin" or "exp". Current value: {name_dist}')


#%% Define variables to export

# re-sorted ys [[mu_y,mu_z,y_bar,z_bar],number of coordinates, number of shuffling replicates, number of sampling (based on Central limit theorem)]
# --> re-sorted means that the order is the same as df_tr
ys_resorted = np.nan*np.ones((4,data_coordinates.shape[0],range_rep_shuffle.size,range_rep_error_calc.size,df_tr.shape[0]))

# re-sorted sigma_zs
sigma_zs_resorted = ys_resorted[:1,...].copy()


# as learning curves (see below) but with validation error
val_errs_opt = np.inf*np.ones((*ys_resorted.shape[:-1],len(Ns_tr)))

# learning curves
lcs = np.nan*val_errs_opt.copy()


# optimal kernel lengths
ls_opt = lcs.copy()

# optimal regularisation parameters (lambdas)
ss_opt = lcs.copy()


# training indeces
idxs_tr = np.ones((data_coordinates.shape[0],range_rep_shuffle.size,df_tr.shape[0]),int)
# idxs_tr = np.ones((1,data_coordinates.shape[0],range_rep_shuffle.size,range_rep_error_calc.size,df_tr.shape[0]),int)

# validation indeces
idxs_va = np.ones((data_coordinates.shape[0],range_rep_shuffle.size,N_va),int)
# idxs_va = np.ones((1,data_coordinates.shape[0],range_rep_shuffle.size,range_rep_error_calc.size,N_va),int)

# test indeces
idxs_te = np.ones((data_coordinates.shape[0],range_rep_shuffle.size,N_te),int)
# idxs_te = np.ones((1,data_coordinates.shape[0],range_rep_shuffle.size,range_rep_error_calc.size,N_te),int)


if bool_store_weights:
    #optimal weights. This is the larges object.
    ws_opt = lcs.copy().astype(object)#np.nan*np.ones((*lcs.shape,df_tr.shape[0],1)).astype(object)

ys_resorted = np.nan*ys_resorted


#%% start of the loop

if bool_plot:
    fig,axs = plt.subplots(2,ys_resorted.shape[0])
    
for i_coord,(N,hs) in enumerate(zip(*data_coordinates.T)):
    print('\n')
    print(N,hs)

    
#%% shuffle training and validation+test
    
    idxs_va_te = []
    for i,rep_shuffle in enumerate(range_rep_shuffle):
        np.random.seed(rep_shuffle)
        df_tr_temp = df_tr.copy().sample(frac=1).sort_values('n_mutations')
        idxs_tr[i_coord,i,:] = df_tr_temp.loc[:,'idx'].values
        df_va_te_temp = df_va_te.copy().sample(frac=1)
        idxs_va_te.append(df_va_te_temp.loc[:,'idx'].values)
         

    idxs_va_te = np.array(idxs_va_te)
    
    
    #%% Calculate Ns_sample (only for training data)
    
    N_tot = N*df_tr.shape[0]
    Ns_sample = dist_population.pk*N_tot
    
    
    #%% Calculate sigma_y and h
    
    sigma_y = np.std(df_tr.energy.values)*sigma_y_sima_pop_ratio
    h = hs*sigma_y
    
    
    #%% Calculate bins
    
    y_tr_min = df_tr.loc[:,'energy'].values.min()
    y_tr_max = df_tr.loc[:,'energy'].values.max()
    bins_edges = [y_tr_min-sigma_y*8]
    while bins_edges[-1]<y_tr_max+sigma_y*8:
        bins_edges.append(bins_edges[-1]+h)
    
    bins_edges = np.array(bins_edges)
    bins_median = bins_edges[:-1]+np.diff(bins_edges)/2
    
    
    #%% Calculate error
    
    ys = ys_resorted.copy()[:,i_coord,...]
    sigma_zs = sigma_zs_resorted.copy()[0,i_coord,...]
    for i,rep_shuffle in enumerate(range_rep_shuffle):
        df_tr_temp = df.iloc[idxs_tr[i_coord,i,:]]
        for ii,rep_error_calc in enumerate(range_rep_error_calc):
            np.random.seed(rep_error_calc)
            for iii,mu_y in enumerate(df_tr_temp.loc[:,'energy'].values):
                n_sample=Ns_sample[iii]
                
                pop_cont = norm(mu_y,sigma_y)
                p_temp = np.diff(pop_cont.cdf(bins_edges))
                
                pop_disc = rv_discrete(values=(bins_median,p_temp))
                pop_hist = rv_histogram(histogram=(p_temp,bins_edges))
                
                mu_z = pop_disc.mean()
                sigma_z = pop_disc.std()
                pop_cont_y_bar = norm(mu_y,sigma_y/np.sqrt(n_sample))
                pop_cont_z_bar = norm(mu_z,sigma_z/np.sqrt(n_sample))
                y_bar = pop_cont_y_bar.rvs(1)[0]
                z_bar = pop_cont_z_bar.rvs(1)[0]
                
                
                ys[0,i,ii,iii] = mu_y
                ys[1,i,ii,iii] = mu_z
                ys[2,i,ii,iii] = y_bar
                ys[3,i,ii,iii] = z_bar
                sigma_zs[i,ii,iii] = sigma_z
    
            print(i_coord,i,ii,iii,time()-t0)
    
    for i,rep_shuffle in enumerate(range_rep_shuffle):
        for ii,rep_error_calc in enumerate(range_rep_error_calc):
            ys_resorted[:,i_coord,i,ii] = ys[:,i,ii][:,np.argsort(idxs_tr[i_coord,i,:])]
            sigma_zs_resorted[:,i_coord,i,ii] = sigma_zs[i,ii][np.argsort(idxs_tr[i_coord,i,:])]
        
    
    #%% ML Training Loop    
    for i,rep_shuffle in enumerate(range_rep_shuffle):
        df_tr_loop = df.iloc[idxs_tr[i_coord,i,:]]
        df_va_te_loop = df.iloc[idxs_va_te[i,:]]
        df_va_loop = df_va_te_loop.iloc[:N_va,:]
        df_te_loop = df_va_te_loop.iloc[N_va:N_te+N_va,:]
        
        idxs_va[i_coord,i,:] = idxs_va_te[i,:][:N_va]
        idxs_te[i_coord,i,:] = idxs_va_te[i,:][N_va:N_te+N_va]
        
        d_tr_tr = calc_dist(arr_from_df(df_tr_loop),arr_from_df(df_tr_loop))
        d_va_tr = calc_dist(arr_from_df(df_va_loop),arr_from_df(df_tr_loop))
        d_te_tr = calc_dist(arr_from_df(df_te_loop),arr_from_df(df_tr_loop))
        
        y_va = df_va_loop.loc[:,'energy'].values.reshape(-1,1)
        y_te = df_te_loop.loc[:,'energy'].values.reshape(-1,1)
        for ii,rep_error_calc in enumerate(range_rep_error_calc):
            y_tr = ys_resorted[:,i_coord,i,ii,:][:,idxs_tr[i_coord,i,:]].T
    
            for iii,N_tr in enumerate(Ns_tr):
                y_tr_lc = y_tr[:N_tr,:]
    
                d_tr_tr_lc = d_tr_tr[:N_tr,:][:,:N_tr]
                d_va_tr_lc = d_va_tr[:,:N_tr]
                d_te_tr_lc = d_te_tr[:,:N_tr]
                
                w_temp = np.nan*np.ones(y_tr_lc.shape)
                for iv,l in enumerate(sigma_s):
                    K_tr_tr_lc = np.exp(-d_tr_tr_lc/l)
                    K_va_tr_lc = np.exp(-d_va_tr_lc/l)
                    for v,lam in enumerate(lambda_s):
                        c, low = cho_factor(K_tr_tr_lc+lam*np.eye(N_tr))
                        w = cho_solve((c, low), y_tr_lc)
                        val_err = error_fun(y_va,K_va_tr_lc@w,axis=0)

                        bool_opt_check = (val_errs_opt[:,i_coord,i,ii,iii]>val_err)
                        if bool_opt_check.any():
                            val_errs_opt[bool_opt_check,i_coord,i,ii,iii] = val_err[bool_opt_check]
                            ls_opt[bool_opt_check,i_coord,i,ii,iii] = l
                            ss_opt[bool_opt_check,i_coord,i,ii,iii] = lam
                            w_temp[:,bool_opt_check]=w[:,bool_opt_check]
    
                
                for vi in range(y_tr.shape[1]):
                    K_te_tr_lc = np.exp(-d_te_tr_lc/ls_opt[vi,i_coord,i,ii,iii])
                    lcs[vi,i_coord,i,ii,iii] = error_fun(y_te,K_te_tr_lc@w_temp[:,vi:vi+1],axis=None)
                    
                    if bool_store_weights:
                        ws_opt[vi,i_coord,i,ii,iii] = w_temp[:,vi:vi+1]
                    # ws_opt[vi,i,ii,iii,np.argsort(idxs_tr[i,:])[:N_tr]] = w_temp[:,vi:vi+1]
                print(i,ii,iii,time()-t0)
    
    
    #%% Plot
    if bool_plot:
        ratio_lcs = lcs[1:]/lcs[:1]
        med_ratio_lcs = np.median(ratio_lcs,axis=[2,3])
        med_lcs = np.median(lcs,axis=[2,3])
        
        for vi in range(y_tr.shape[1]):
            axs[0,vi].set_ylim(10**(np.log10(np.nanmin(lcs))*0.9),
                               10**(np.log10(np.nanmax(lcs))*1.1))
            axs[1,vi].set_ylim(10**(np.log10(np.nanmin(ratio_lcs))*0.9),
                               10**(np.log10(np.nanmax(ratio_lcs))*1.1))
        
        for i,rep_shuffle in enumerate(range_rep_shuffle):
            for ii,rep_error_calc in enumerate(range_rep_error_calc):
                for vi in range(y_tr.shape[1]):
                    axs[0,vi].semilogy(Ns_tr_norm,lcs[vi,i_coord,i,ii,:],'.k',alpha=0.2)
                    if vi > 0:
                        axs[1,vi].semilogy(Ns_tr_norm,ratio_lcs[vi-1,i_coord,i,ii,:],'.k',alpha=0.2)
        for vi in range(y_tr.shape[1]):
            axs[0,vi].semilogy(Ns_tr_norm,med_lcs[vi,i_coord],'r',zorder=np.inf)
            if vi > 0:
                axs[1,vi].semilogy(Ns_tr_norm,med_ratio_lcs[vi-1,i_coord],'r',zorder=np.inf)
            else:
                try:
                    axs[1,vi].remove()
                except:
                    1
        
        plt.pause(0.01)
        
        
#%% Save Data
print('initialize saving. ',time()-t0)
if bool_save_data:
    d_out['inputs_and_parameters'] = d_inputs_and_parameters
    d_out['ys_resorted'] = ys_resorted
    d_out['sigma_zs_resorted'] = sigma_zs_resorted
    d_out['val_errs_opt'] = val_errs_opt
    d_out['lcs'] = lcs
    d_out['ls_opt'] = ls_opt
    d_out['ss_opt'] = ss_opt
    d_out['idxs_tr'] = idxs_tr.reshape(1,data_coordinates.shape[0],range_rep_shuffle.size,1,-1)
    d_out['idxs_va'] = idxs_va.reshape(1,data_coordinates.shape[0],range_rep_shuffle.size,1,-1)
    d_out['idxs_te'] = idxs_te.reshape(1,data_coordinates.shape[0],range_rep_shuffle.size,1,-1)
    if bool_store_weights:
        d_out['ws_opt'] = ws_opt
    
    file_out_name = f'{name_dist}_{mode_points}_{coordinates[0]:0{len(str(len_data_coordinates))}d}_{n_rep_shuffle}_{n_rep_error_calc}'
    try:
        np.savez(save_path+file_out_name,**d_out)
    except:
        np.savez('./'+file_out_name,**d_out)
        
print('saved. ',time()-t0)
