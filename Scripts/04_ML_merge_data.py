#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 07:43:40 2024

@author: vdoffini
"""

import argparse
import glob
import numpy as np
from os.path import isfile
#%% ArgParser

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to merge",
                    action="store",default='../Results/')
parser.add_argument("--pattern", type=str, help="Pattern to merge",
                    action="store",default='uni_2D_*_*_*.npz')
args = parser.parse_args()

path = args.path
pattern = args.pattern

# pattern = 'bin_n_crit_*_*_*.npz'
# path = '/Users/vdoffini/Data/sorting_error_2D/bin_n_crit/'

#%% Functions
def f_reshape(k):
    if 'Ns' in k:
        1

#%% Parameters

list_files = glob.glob(path+pattern)

temp = []
for s in list_files:
    s0 = s.split('/')[-1].split('.')[0].split('_')
    temp.append([s0[-3],s0[-2],s0[-1]])

temp = np.array(temp)#.astype(int)
print(temp)
# Ns (1, 1)
# HSs (1, 1)
# Ns_tr_norm (4,)
# Ns_tr (4,)
# N_va ()
# N_te ()
# inputs_and_parameters ()
# ys_resorted (4, 1, 1, 1, 2605)
# sigma_zs_resorted (1, 1, 1, 1, 2605)
# val_errs_opt (4, 1, 1, 1, 4)
# lcs (4, 1, 1, 1, 4)
# ls_opt (4, 1, 1, 1, 4)
# ss_opt (4, 1, 1, 1, 4)
# idxs_tr (1, 1, 2605)
# idxs_va (1, 1, 8960)
# idxs_te (1, 1, 21504)

k_merge_not_merged = ['Ns_tr_norm','Ns_tr','N_va','N_te','inputs_and_parameters']
k_merge_i = ['Ns','HSs']
k_merge_ii = ['idxs_tr','idxs_va','idxs_te']
k_merge_i_ii_iii = ['ys_resorted','sigma_zs_resorted','val_errs_opt','lcs','ls_opt','ss_opt']


#%%
s_0 = 'conda run --no-capture-output -n Process4DataAnalysis python3 -u ./04_ML.py'

# with open('./SLURM_list_comands_repeat.txt', "w") as text_file:
#     text_file.write('')

# s0 = pattern.split('*')
# for i,s in enumerate(np.sort(np.unique(temp[:,0]))):
#     # print(s)
#     for ii,ss in enumerate(np.sort(np.unique(temp[:,1]))):
#         for iii,sss in enumerate(np.sort(np.unique(temp[:,2]))):
#             s_temp = path+s0[0]+s+s0[1]+ss+s0[2]+sss+s0[3]
#             if isfile(s_temp):
#                 1
#             else:
#                 print(i,ii,iii)
#                 with open('./SLURM_list_comands_repeat.txt', "a") as text_file:
#                     text_file.write(s_0 + f' -d {pattern.split("*")[0].split("_")[0]} --mode_points {"_".join(pattern.split("*")[0].split("_")[1:-1])} --coordinates_step 10 --n_lc 1  --n_mut_train_max 3 -c {i} --n_rep_shuffle {ii} --n_rep_error_calc {iii}\n')


# s0 = pattern.split('*')
# for i,s in enumerate(np.sort(np.unique(temp[:,0]))):
#     print(s)
#     for ii,ss in enumerate(np.sort(np.unique(temp[:,1]))):
#         for iii,sss in enumerate(np.sort(np.unique(temp[:,2]))):
#             s_temp = path+s0[0]+s+s0[1]+ss+s0[2]+sss+s0[3]
#             with np.load(s_temp,allow_pickle=True) as file:
#                 lcs = file['lcs']
#                 if lcs.shape[-1] != 4:
#                     print(s_temp)
#                     with open('./SLURM_list_comands_repeat.txt', "a") as text_file:
#                         text_file.write(s_0 + f' -d {pattern.split("*")[0].split("_")[0]} --mode_points {"_".join(pattern.split("*")[0].split("_")[1:-1])} --coordinates_step 10 --n_lc 1  --n_mut_train_max 3 -c {i} --n_rep_shuffle {ii} --n_rep_error_calc {iii}\n')


#%% Merge i_ii_iii
excluded = []
s0 = pattern.split('*')
for i,s in enumerate(np.sort(np.unique(temp[:,0]))):
    print(s)
    for ii,ss in enumerate(np.sort(np.unique(temp[:,1]))):
        for iii,sss in enumerate(np.sort(np.unique(temp[:,2]))):
            s_temp = path+s0[0]+s+s0[1]+ss+s0[2]+sss+s0[3]
            if isfile(s_temp):
                if iii == 0:
                    with np.load(s_temp,allow_pickle=True) as file:
                        for k in np.concatenate((k_merge_i,k_merge_ii,k_merge_i_ii_iii)):
                            if k in k_merge_i_ii_iii:
                                locals()[k+f'_{i}_{ii}'] = file[k]
                            elif k in k_merge_i:
                                locals()[k+f'_{i}_{ii}'] = file[k].reshape(1,1,1,1,1)
                            elif k in k_merge_ii:
                                locals()[k+f'_{i}_{ii}'] = file[k].reshape(1,1,1,1,-1)
                else:
                    with np.load(s_temp,allow_pickle=True) as file:
                        for k in np.concatenate((k_merge_i,k_merge_ii,k_merge_i_ii_iii)):
                            if k in k_merge_i_ii_iii:
                                locals()[k] = file[k]
                            elif k in k_merge_i:
                                locals()[k] = file[k].reshape(1,1,1,1,1)
                            elif k in k_merge_ii:
                                locals()[k] = file[k].reshape(1,1,1,1,-1)
                            locals()[k+f'_{i}_{ii}'] = np.concatenate((locals()[k+f'_{i}_{ii}'],locals()[k]),axis=3)

                            
            else:
                print(i,ii,iii)
            
        for k in np.concatenate((k_merge_i,k_merge_ii,k_merge_i_ii_iii)):
            if ii == 0:
                locals()[k+f'_{i}'] = locals()[k+f'_{i}_{ii}']
                del(locals()[k+f'_{i}_{ii}'])
            else:
                locals()[k+f'_{i}'] = np.concatenate((locals()[k+f'_{i}'],locals()[k+f'_{i}_{ii}']),axis=2)
                del(locals()[k+f'_{i}_{ii}'])
    
    for k in np.concatenate((k_merge_i,k_merge_ii,k_merge_i_ii_iii)):
        if i == 0:
            locals()[k+'_out'] = locals()[k+f'_{i}']
            del(locals()[k+f'_{i}'])
        else:
            locals()[k+'_out'] = np.concatenate((locals()[k+'_out'],locals()[k+f'_{i}']),axis=1)
            del(locals()[k+f'_{i}'])
            
for k in np.concatenate((k_merge_i,k_merge_ii,k_merge_i_ii_iii)):
    locals()[k] = locals()[k+'_out']
    del(locals()[k+'_out'])


#%% Not Merged

with np.load(s_temp,allow_pickle=True) as file:
    for k in k_merge_not_merged:
        locals()[k] = file[k]


#%%
d_out = {}
for k in np.concatenate((k_merge_not_merged,k_merge_i,k_merge_ii,k_merge_i_ii_iii)):
    d_out[k] = locals()[k]

temp = pattern.split('*')
try:
    np.savez(path+f'{temp[0]}out.npz',**d_out)
except:
    np.savez('./'+f'{temp[0]}out.npz',**d_out)
