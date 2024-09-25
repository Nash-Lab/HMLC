#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:35:21 2024

@author: vdoffini
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm,lognorm
from matplotlib.patches import PathPatch,Path


#%% Function

def rot_matrix(angle):
    return np.array([[np.cos(angle),-np.sin(angle)],
                     [np.sin(angle),np.cos(angle)]])


#%% Parameters

n_gates = 8
n_samples = 200
off_set = 1

mus_x = np.array([-0.5, 0.0, 0.5])
sigmax = np.array([0.2])

save_path = '../Results/'
file_out_name = '00_summary_data'

bool_plot = True

bool_save_data = True


#%% Calculations

pop_dist_1 = norm(mus_x[0], sigmax[0])
pop_dist_2 = norm(mus_x[1], sigmax[0])
pop_dist_3 = norm(mus_x[2], sigmax[0])

pop_dist_y = lognorm(0.35,0.1)

np.random.seed(17)
x_log_norm_rot = np.concatenate((pop_dist_1.rvs(size=(n_samples,1)),
                                 pop_dist_2.rvs(size=(n_samples,1)),
                                 pop_dist_3.rvs(size=(n_samples,1))),axis=0)
x_log_norm_rot = np.concatenate((x_log_norm_rot,
                                 pop_dist_y.rvs(size=(n_samples*3,1))),axis=1)

y = np.repeat([i for i in range(3)],n_samples)

idx = np.arange(y.size)
np.random.shuffle(idx)
x_log_norm_rot = x_log_norm_rot[idx]
y = y[idx]

x_log_norm = x_log_norm_rot@rot_matrix(np.pi/4)

x_log = x_log_norm+off_set

x = 10**x_log

bins_edges_log_norm_rot = np.histogram(x_log_norm_rot[:,0],bins=np.linspace([x_log_norm_rot[:,0].min()*1.1 if np.sign(x_log_norm_rot[:,0].min())<0 else x_log_norm_rot[:,0].min()*0.9][0],
                                                                            x_log_norm_rot[:,0].max()*1.1,
                                                                            n_gates+1))[1]

gates_log_norm_rot=[]
for i in range(len(bins_edges_log_norm_rot)-1):
    gates_log_norm_rot.append([
                               [bins_edges_log_norm_rot[i],[x_log_norm_rot[:,1].min()*1.1 if np.sign(x_log_norm_rot[:,1].min())<0 else x_log_norm_rot[:,1].min()*0.9][0]],
                               [bins_edges_log_norm_rot[i],x_log_norm_rot[:,1].max()*1.1],
                               [bins_edges_log_norm_rot[i+1],x_log_norm_rot[:,1].max()*1.1],
                               [bins_edges_log_norm_rot[i+1],[x_log_norm_rot[:,1].min()*1.1 if np.sign(x_log_norm_rot[:,1].min())<0 else x_log_norm_rot[:,1].min()*0.9][0]],
                               [bins_edges_log_norm_rot[i],[x_log_norm_rot[:,1].min()*1.1 if np.sign(x_log_norm_rot[:,1].min())<0 else x_log_norm_rot[:,1].min()*0.9][0]],
                               ])
gates_log_norm_rot = np.array(gates_log_norm_rot)
gates = 10**(gates_log_norm_rot@rot_matrix(np.pi/4)+off_set)

arrow_coordinates = 10**(np.array([
                                   [bins_edges_log_norm_rot[0],.075],
                                   [bins_edges_log_norm_rot[-1],.075],
                                   ])@rot_matrix(np.pi/4)+off_set)

text_coordinates = 10**(np.array([
                                  [np.mean([bins_edges_log_norm_rot[-1],bins_edges_log_norm_rot[0]]),-.2],
                                  ])@rot_matrix(np.pi/4)+off_set)


#%% Save

if bool_save_data:
    d_out = {}
    d_out['x_log_norm_rot'] = x_log_norm_rot
    d_out['x_log_norm'] = x_log_norm
    d_out['x_log'] = x_log
    d_out['x'] = x
    d_out['y'] = y
    
    d_out['bins_edges_log_norm_rot'] = bins_edges_log_norm_rot
    d_out['gates_log_norm_rot'] = gates_log_norm_rot
    d_out['gates'] = gates
    d_out['arrow_coordinates'] = arrow_coordinates
    d_out['text_coordinates'] = text_coordinates

    d_out['n_gates'] = np.array([n_gates])
    d_out['n_samples'] = np.array([n_samples])
    d_out['off_set'] = np.array([off_set])

    d_out['sigmax'] = sigmax
    d_out['mus_x'] = mus_x
    
    np.savez(save_path+file_out_name, **d_out)


#%% Plot

if bool_plot:
    plt.close('all')
    fig,ax = plt.subplots(1,1);
    ax.scatter(x[:,0],x[:,1],s=5,c=y);
    ax.loglog()
    ax.set_aspect('equal')
    ax.set_xlim(1, ax.get_xlim()[1])
    ax.set_ylim(1, ax.get_ylim()[1])
    
    ax.annotate('', xy=arrow_coordinates[0,:], xytext=arrow_coordinates[1,:],arrowprops=dict(arrowstyle="<-"), horizontalalignment='center',
                verticalalignment='center',transform=ax.transData, annotation_clip=False, clip_on=False)
    ax.text(text_coordinates[0,0], text_coordinates[0,1], '$y_i$'+' or '+'$z_i$' , horizontalalignment='center',
            verticalalignment='center', rotation=-45)
    
    for i in range(len(gates)):
        ax.add_patch(PathPatch(Path(gates[i]), edgecolor = 'grey', alpha = 1, facecolor='none',zorder=np.inf))
    
    
