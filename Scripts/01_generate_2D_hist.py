#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:43:39 2024

@author: vdoffini
"""

import numpy as np
from scipy.stats import norm,rv_histogram,rv_discrete
from scipy.special import erf
from scipy.integrate import quad
import math 
import warnings
#%%
def shannon_h(p):
    return -np.nansum(p*np.log(p))

def integrand(x,dist_norm,dist_hist):
    hist_x = dist_hist.pdf(x)
    norm_x = dist_norm.pdf(x)
    if (norm_x == 0)|(hist_x == 0):
        return 0
    else:
        return norm_x * math.log(norm_x / hist_x)
#%% Parameters

bool_plot = True # if plot or not

bool_save = True # if save data (not plot) or not
save_path = '../Results/'
save_name = '01_2D_hist.npy'

n_grid = 501 # nuber of grid points (n_grid * n_grid)

h_crit = np.sqrt(6)

zh_s_span = (-0.5,0.5)# min and distance between (true) mean and closest gate edge / gates width

multiplier4maxBin = 20.


bool_test = False # if run test. If True it prints the maximum absolute difference between all measures using different mus and sigmas

#%% Derived parameters
hs_s_span = (h_crit/20,h_crit*20) # min and max gates width / sigma of normal distributed population (true distribution)

hs_s = np.logspace(np.log10(hs_s_span[0]),np.log10(hs_s_span[1]),n_grid)[:]
zh_s = np.linspace(zh_s_span[0],zh_s_span[-1],n_grid)[:]

HSs,ZHs = np.meshgrid(hs_s,zh_s)

max_bin = HSs.max()*multiplier4maxBin

if bool_test:
    t1,t2 = np.meshgrid([0,1,2],[1,2,4])
    t1,t2 = t1.flatten(),t2.flatten()
    mus_sigmas = np.vstack((t1,t2))
    bool_plot = False
    bool_save = False
else:
    mus_sigmas = np.array([[0,2]]).T

if bool_plot:
    import matplotlib.pyplot as plt
    plt.close('all')
    
    #%% Generate Data

for mu,sigma in zip(*mus_sigmas):
    
    pop_cont = norm(mu,sigma)
    
    med = pop_cont.median()# for norm mu==median


    out_temp_kld = []
    out_temp_h_hist = []
    out_temp_h_disc = []
    out_temp_m = []
    out_temp_med = []
    out_temp_s_hist = []
    out_temp_s_disc = []
    out_temp_q_68_min = [] #0.68268949
    out_temp_q_68_max = [] #0.68268949
    out_temp_q_95_min = [] #0.95449974
    out_temp_q_95_max = [] #0.95449974
    out_temp_q_99_min = [] #0.9973002
    out_temp_q_99_max = [] #0.9973002
    for i,(hs,zh0) in enumerate(zip(HSs.flatten(),
                                    ZHs.flatten())):
        b = hs*sigma*np.ceil(max_bin/hs)
        bins_tot = zh0*hs*sigma + mu + np.concatenate((-np.linspace(0,b,np.ceil(max_bin/hs).astype(int)*2+1)[1:][::-1],
                                                   np.linspace(0,b,np.ceil(max_bin/hs).astype(int)*2+1)))
        bins_edges = bins_tot[::2]
        bins_median = bins_tot[1::2]
        p_temp = np.diff(pop_cont.cdf(bins_edges))
        
        pop_hist = rv_histogram(histogram=(p_temp,bins_edges))
        pop_disc = rv_discrete(values=(bins_median,p_temp))
        
        kld = 0
        err = 0
        # for ii in np.arange(p_temp.size)[p_temp>0]:
        #     kld_temp, err_temp = quad(integrand, bins_edges[ii], bins_edges[ii+1], args=(pop_cont,pop_hist))
        #     kld += kld_temp
        #     err += err_temp

        h_hist = pop_hist.entropy()
        h_disc = pop_disc.entropy()
        mean = pop_hist.mean() - mu
        median = pop_hist.median() - med
        s_hist = pop_hist.std()
        s_disc = pop_disc.std()
        
        q_temp = (np.array(pop_hist.interval([0.68268949,0.95449974,0.9973002])).T - med)
        q_68_min = np.min(q_temp[0])
        q_68_max = np.max(q_temp[0])
        q_95_min = np.min(q_temp[1])
        q_95_max = np.max(q_temp[1])
        q_99_min = np.min(q_temp[2])
        q_99_max = np.max(q_temp[2])
    
        out_temp_kld.append(kld)
        out_temp_h_hist.append(h_hist)
        out_temp_h_disc.append(h_disc)
        out_temp_m.append(mean)
        out_temp_med.append(median)
        out_temp_s_hist.append(s_hist)
        out_temp_s_disc.append(s_disc)
        out_temp_q_68_min.append(q_68_min)
        out_temp_q_68_max.append(q_68_max)
        out_temp_q_95_min.append(q_95_min)
        out_temp_q_95_max.append(q_95_max)
        out_temp_q_99_min.append(q_99_min)
        out_temp_q_99_max.append(q_99_max)
        
        print(i,(hs,zh0))

    
    # out_kld = np.array(out_temp_kld).reshape(HSs.shape)
    out_h_hist = np.array(out_temp_h_hist).reshape(HSs.shape)
    out_h_disc = np.array(out_temp_h_disc).reshape(HSs.shape)
    out_kld = out_h_hist-norm(mu,sigma).entropy()
    out_mean = np.array(out_temp_m).reshape(HSs.shape)/sigma
    out_medi = np.array(out_temp_med).reshape(HSs.shape)/sigma
    out_std_hist = np.array(out_temp_s_hist).reshape((HSs.shape))/sigma
    out_std_disc = np.array(out_temp_s_disc).reshape((HSs.shape))/sigma
    out_q_68_min = np.array(out_temp_q_68_min).reshape((HSs.shape))/sigma
    out_q_68_max = np.array(out_temp_q_68_max).reshape((HSs.shape))/sigma
    out_q_95_min = np.array(out_temp_q_95_min).reshape((HSs.shape))/sigma
    out_q_95_max = np.array(out_temp_q_95_max).reshape((HSs.shape))/sigma
    out_q_99_min = np.array(out_temp_q_99_min).reshape((HSs.shape))/sigma
    out_q_99_max = np.array(out_temp_q_99_max).reshape((HSs.shape))/sigma
    
    ss_out = [
              'out_kld',
              'out_h_hist',
              'out_h_disc',
              'out_mean',
              'out_medi',
              'out_std_hist',
              'out_std_disc',
              'out_q_68_min',
              'out_q_68_max',
              'out_q_95_min',
              'out_q_95_max',
              'out_q_99_min',
              'out_q_99_max'
              ]
    
    for s in ss_out:
        locals()[s+f'_m{int(mu)}s{int(sigma)}']=locals()[s].copy()
    
#%%
    if bool_plot:        
            for s in ss_out:
                fig,ax = plt.subplots(1,1)
                ct = ax.contourf(ZHs,HSs,locals()[s],levels=100)
                plt.semilogy()
                cb = fig.colorbar(ct)
                ax.set_xlabel('$z_0/h$')
                ax.set_ylabel(r'$h/\sigma_{y}$')
                cb.ax.set_ylabel(s)
                plt.pause(0.01)

#%%
if bool_test:
    temp = []
    for s in ss_out:
        for mu1,sigma1 in zip(*mus_sigmas[:,:-1]):
            for mu2,sigma2 in zip(*mus_sigmas[:,1:]):
                temp.append(np.max(np.abs((locals()[s+f'_m{int(mu1)}s{int(sigma1)}']-locals()[s+f'_m{int(mu2)}s{int(sigma2)}']).flatten())))
        print(f's={s};')
        print(f'max(abs(s-s*))={np.max(temp)}')
        print('')

#%%

if bool_save:
    d_out = {}
    d_out['HSs'] = HSs
    d_out['ZHs'] = ZHs
    d_out['multiplier4maxBin'] = np.array([multiplier4maxBin])
    for s in ss_out: d_out[s] = locals()[s]
    
    try:
        np.save(save_path+save_name,d_out)
    except:
        np.save('./'+save_name,d_out)
        warnings.warn('Data saved on "./"')


    