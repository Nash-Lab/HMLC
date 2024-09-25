#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:25:53 2024

@author: vdoffini
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import plasma,winter,bwr,cividis,twilight_shifted,ScalarMappable,get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize,LinearSegmentedColormap,ListedColormap,LogNorm,SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch,Rectangle
from matplotlib.path import Path
import matplotlib.transforms as transforms
cm = ListedColormap(twilight_shifted(np.linspace(0,31/64, 1024)))
plt.rcParams["font.family"] = "Times New Roman"#https://stackoverflow.com/questions/40734672/how-to-set-the-label-fonts-as-time-new-roman-by-drawparallels-in-python/40734893
fontsize = 13
fontsize_sup = 15
fontsize_letter = 30
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.size': fontsize})
plt.rcParams.update({'legend.fontsize': fontsize})
plt.rcParams.update({'axes.labelsize': fontsize})

#%% Parameters
data_path = '/Users/vdoffini/Data/sorting_error_2D/'
data_name = 'bin_2D_lcs.npz'

# cmap = plt.cm.plasma
# cmap = plt.cm.RdYlGn_r
cmap = plt.cm.summer
cmap = LinearSegmentedColormap.from_list("", [
                                              # 'peru','goldenrod','olive','olivedrab',
                                              'green',
                                              'limegreen','yellowgreen','palegoldenrod','orange'])
# cmap = plt.cm.viridis

gs_subfig_ABC_ratio = np.array([0.7,2])

color_h_crit = 'tab:pink'
cmap_h_crit = LinearSegmentedColormap.from_list("", ['black',color_h_crit,'white'])
cmap_h_crit = LinearSegmentedColormap.from_list("", [cmap_h_crit(0.25),color_h_crit,cmap_h_crit(0.75)])
color_n_crit = 'tab:cyan'
cmap_n_crit = LinearSegmentedColormap.from_list("", ['black',color_n_crit,'white'])
cmap_n_crit = LinearSegmentedColormap.from_list("", [cmap_n_crit(0.25),color_n_crit,cmap_n_crit(0.75)])
color_eq = 'red'
cmap_eq = LinearSegmentedColormap.from_list("", ['black',color_eq,'white'])
cmap_eq = LinearSegmentedColormap.from_list("", [cmap_eq(0.25),color_eq,cmap_eq(0.75)])
color_contour_crit = 'blue'
cmap_contour_crit = LinearSegmentedColormap.from_list("", ['black',color_contour_crit,'white'])
cmap_contour_crit = LinearSegmentedColormap.from_list("", [cmap_contour_crit(0.25),color_contour_crit,cmap_contour_crit(0.75)])

d_style = {'bias':{'linestyle':'--','dashes':[3, 3]},
           'var':{'linestyle':'--','dashes':[1, 1, 1, 1]},
           'bias_plus_var':{'linestyle':'-'}}
d_style_contour = {'linewidth':2,'alpha':1}

crit_ticks_length = 5

alpha_main = 0.5
alpha_sec = 0.15
alpha_contour = 0.5

step_detials_sec = 5

bounds_insets = [0.01,0.01,0.375,0.375]

level_of_accuracy = 3 # 0 => \mu_y; 1 => \mu_z; 2 => \bar{y}; 3 => \bar{z}

#%% Load Data

# with np.load(data_path+data_name,allow_pickle=True) as data:
#     #for k in data.keys():
#         1

datalim_lcs = [np.inf,-np.inf]
datalim_lcs_ratio = [np.inf,-np.inf]
for s in ['eq_equidistant','contour_crit_equidistant','n_crit','h_crit']:
    with np.load(data_path+f'{data_name.split("_")[0]}_{s}_out.npz',allow_pickle=True) as data:
        #for k in data.keys():
            temp = np.median(data['lcs'],axis=[2,3])[...,:]
            locals()[f'lcs_{s}'] = temp
            temp = temp/temp[:1,...]
            locals()[f'lcs_ratio_{s}'] = temp
            Ns_tr_norm = data['Ns_tr_norm']
            Ns_tr = data['Ns_tr']
            
            locals()[f'Ns_{s}'] = data['Ns'][0,:,0,0,0]
            locals()[f'HSs_{s}'] = data['HSs'][0,:,0,0,0]
            
            datalim_lcs = [np.min([datalim_lcs[0],np.min(locals()[f'lcs_{s}'])]),
                           np.max([datalim_lcs[1],np.max(locals()[f'lcs_{s}'])])]
            datalim_lcs_ratio = [np.min([datalim_lcs_ratio[0],np.min(locals()[f'lcs_ratio_{s}'])]),
                                 np.max([datalim_lcs_ratio[1],np.max(locals()[f'lcs_ratio_{s}'])])]

d_norm_not_norm = {i:j for (i,j) in zip(Ns_tr_norm,Ns_tr)}
x_ticks_N_norm = [0,1,2,3]

#%%
plt.close('all')

fig = plt.figure(layout='constrained', figsize=(8.9, 9.5))#latex --> (5.5,9)

gs = fig.add_gridspec(2,2, hspace=0.12, wspace=0.12)
gs.set_height_ratios([1,1])
gs.set_width_ratios([1,1])

subfig_A = fig.add_subfigure(gs[0, 0])
subfig_B = fig.add_subfigure(gs[0, -1])
subfig_C = fig.add_subfigure(gs[-1, 0])
subfig_D = fig.add_subfigure(gs[-1, -1])
# subfig_CM_y = fig.add_subfigure(gs[:, 1])
# subfig_CM_x_AC = fig.add_subfigure(gs[1, 0])
# subfig_CM_x_BD = fig.add_subfigure(gs[1, -1])

# subfig_A.set_facecolor('1.')
# subfig_B.set_facecolor('0.2')
# subfig_C.set_facecolor('0.4')
# subfig_D.set_facecolor('0.6')
# subfig_CM.set_facecolor('0.8')


gs_A = subfig_A.add_gridspec(1,1)
gs_B = subfig_B.add_gridspec(1,1)
gs_C = subfig_C.add_gridspec(1,1)
gs_D = subfig_D.add_gridspec(1,1)

# gs_CM_y = subfig_CM_y.add_gridspec(1,2)

# gs_CM_x_AC = subfig_CM_x_AC.add_gridspec(2,1)
# gs_CM_x_BD = subfig_CM_x_BD.add_gridspec(2,1)


ax_A = subfig_A.add_subplot(gs_A[0,0])
# ax_A_inset = ax_A.inset_axes(bounds_insets)

ax_B = subfig_B.add_subplot(gs_B[0,0])
# ax_B_inset = ax_B.inset_axes(bounds_insets)

ax_C = subfig_C.add_subplot(gs_C[0,0])
# ax_C_inset = ax_C.inset_axes(bounds_insets)

ax_D = subfig_D.add_subplot(gs_D[0,0])
# ax_D_inset = ax_D.inset_axes(bounds_insets)

# ax_CM_x_A = subfig_CM_x_AC.add_subplot(gs_CM_x_AC[0, 0])
# ax_CM_x_C = subfig_CM_x_AC.add_subplot(gs_CM_x_AC[-1, 0])
# ax_CM_x_B = subfig_CM_x_BD.add_subplot(gs_CM_x_BD[0, 0])
# ax_CM_x_D = subfig_CM_x_BD.add_subplot(gs_CM_x_BD[-1, 0])

# ax_CM_y = subfig_CM_y.add_subplot(gs_CM_y[:, :])
# ax_CM_y_inset = subfig_CM_y.add_subplot(gs_CM_y[:, 1])
# ax_CM_y_inset = ax_CM_y.twinx()


#%% Colorbar


#%% subfig A

s = 'eq_equidistant'
cmap_A = cmap_eq

temp_shape = locals()[f'lcs_ratio_{s}'].shape[1]
for i in range(temp_shape):
    ax_A.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][level_of_accuracy,i,:],color = cmap_A(i/(1e-8+temp_shape-1)))
ax_A.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][0,0,:],color = 'k', linestyle = ':')
ax_A.set_xlim(ax_A.get_xlim())
ax_A.xaxis.grid()
ax_A.set_yscale('log')
# ax_A.set_yticklabels([])
ax_A.set_xlabel(r'$N_{norm}^{train}$')
# ax_A.xaxis.set_label_position('top')
# ax_A.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                  left=True, labelleft=True, right=False, labelright=False)
ax_A.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                  left=True, labelleft=True, right=False, labelright=False)
ax_A.set_ylabel(r'$MAE_{\mu_{y}\leftarrow\bar{z}}^{test}$'+' [kcal/mol]')
ax_A.set_xticks(x_ticks_N_norm)

# temp_shape = locals()[f'lcs_ratio_{s}'].shape[1]
# for i in range(temp_shape):
#     ax_A_inset.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][level_of_accuracy,i,:],color = cmap_A(i/(1e-8+temp_shape-1)))
# ax_A_inset.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][0,0,:],color = 'k', linestyle = ':')
# ax_A_inset.set_yscale('log')
# ax_A_inset.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                        left=False, labelleft=False, right=True, labelright=True)
# ax_A_inset.set_xticks(ax_A.get_xticks())
# ax_A_inset.set_xlim(ax_A.get_xlim())
# ax_A_inset.xaxis.grid()
# ax_A_inset.set_xticklabels([])
# ax_A_inset.set_yticklabels([])

ax_A_sec = ax_A.secondary_xaxis('top')
ax_A_sec.set_xticks(x_ticks_N_norm)
ax_A_sec.set_xticklabels([d_norm_not_norm[i] for i in x_ticks_N_norm])
ax_A_sec.set_xlabel(r'$N_{}^{train}$')

cb_A = plt.colorbar(mappable=ScalarMappable(cmap=cmap_A), ax=ax_A, orientation='horizontal', location = 'top')
cb_A.set_ticks(cb_A.ax.get_xlim(),minor=False)
cb_A.set_ticks(np.linspace(*cb_A.ax.get_xlim(),temp_shape),minor=True)
cb_A.set_ticks(np.linspace(*cb_A.ax.get_xlim(),temp_shape),minor=True)
cb_A.set_ticklabels([])
cb_A.set_label("".join([r'$R_{D}=1$','\n',r'$\ high\ B.+V.\longrightarrow low\ B.+V.$']))


#%% subfig B

s = 'contour_crit_equidistant'
cmap_B = cmap_contour_crit

temp_shape = locals()[f'lcs_ratio_{s}'].shape[1]
for i in range(temp_shape):
    ax_B.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][level_of_accuracy,i,:],color = cmap_B(i/(1e-8+temp_shape-1)))
ax_B.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][0,0,:],color = 'k', linestyle = ':')
ax_B.set_xlim(ax_B.get_xlim())
ax_B.xaxis.grid()
ax_B.set_yscale('log')
# ax_B.set_yticklabels([])
ax_B.set_xlabel(r'$N_{norm}^{train}$')
# ax_B.xaxis.set_label_position('top')
# ax_B.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                       left=False, labelleft=False, right=True, labelright=True)
ax_B.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                       left=True, labelleft=True, right=False, labelright=False)
ax_B.set_ylabel(r'$MAE_{\mu_{y}\leftarrow\bar{z}}^{test}$'+' [kcal/mol]')
ax_B.set_xticks(x_ticks_N_norm)

# temp_shape = locals()[f'lcs_ratio_{s}'].shape[1]
# for i in range(temp_shape):
#     ax_B_inset.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][level_of_accuracy,i,:],color = cmap_B(i/(1e-8+temp_shape-1)))
# ax_B_inset.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][0,0,:],color = 'k', linestyle = ':')
# ax_B_inset.set_yscale('log')
# ax_B_inset.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                        left=False, labelleft=False, right=True, labelright=True)
# ax_B_inset.set_xticks(ax_B.get_xticks())
# ax_B_inset.set_xlim(ax_B.get_xlim())
# ax_B_inset.xaxis.grid()
# ax_B_inset.set_xticklabels([])
# ax_B_inset.set_yticklabels([])

ax_B_sec = ax_B.secondary_xaxis('top')
ax_B_sec.set_xticks(x_ticks_N_norm)
ax_B_sec.set_xticklabels([d_norm_not_norm[i] for i in x_ticks_N_norm])
ax_B_sec.set_xlabel(r'$N_{}^{train}$')


cb_B = plt.colorbar(mappable=ScalarMappable(cmap=cmap_B), ax=ax_B, orientation='horizontal', location = 'top')
cb_B.set_ticks(cb_B.ax.get_xlim(),minor=False)
cb_B.set_ticks(np.linspace(*cb_B.ax.get_xlim(),temp_shape),minor=True)
cb_B.set_ticks(np.linspace(*cb_B.ax.get_xlim(),temp_shape),minor=True)
cb_B.set_ticklabels([])
cb_B.set_label("".join([r'$Bias+Var.$','\n',r'$\ Var.\longrightarrow Bias.$']))


#%% subfig C

s = 'h_crit'
cmap_C = cmap_h_crit

temp_shape = locals()[f'lcs_ratio_{s}'].shape[1]
for i in range(temp_shape):
    ax_C.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][level_of_accuracy,i,:],color = cmap_C(i/(1e-8+temp_shape-1)))
ax_C.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][0,0,:],color = 'k', linestyle = ':')
ax_C.set_xlim(ax_C.get_xlim())
ax_C.xaxis.grid()
ax_C.set_yscale('log')
# ax_C.set_yticklabels([])
ax_C.set_xlabel(r'$N_{norm}^{train}$')
# ax_C.xaxis.set_label_position('top')
ax_C.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                       left=True, labelleft=True, right=False, labelright=False)
ax_C.set_ylabel(r'$MAE_{\mu_{y}\leftarrow\bar{z}}^{test}$'+' [kcal/mol]')
ax_C.set_xticks(x_ticks_N_norm)

# temp_shape = locals()[f'lcs_ratio_{s}'].shape[1]
# for i in range(temp_shape):
#     ax_C_inset.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][level_of_accuracy,i,:],color = cmap_C(i/(1e-8+temp_shape-1)))
# ax_C_inset.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][0,0,:],color = 'k', linestyle = ':')
# ax_C_inset.set_yscale('log')
# ax_C_inset.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                        left=False, labelleft=False, right=True, labelright=True)
# ax_C_inset.set_xticks(ax_C.get_xticks())
# ax_C_inset.set_xlim(ax_C.get_xlim())
# ax_C_inset.xaxis.grid()
# ax_C_inset.set_xticklabels([])
# ax_C_inset.set_yticklabels([])

ax_C_sec = ax_C.secondary_xaxis('top')
ax_C_sec.set_xticks(x_ticks_N_norm)
ax_C_sec.set_xticklabels([d_norm_not_norm[i] for i in x_ticks_N_norm])
ax_C_sec.set_xlabel(r'$N_{}^{train}$')


cb_C = plt.colorbar(mappable=ScalarMappable(cmap=cmap_C, norm = LogNorm(locals()[f'Ns_{s}'].min(),locals()[f'Ns_{s}'].max())), ax=ax_C, orientation='horizontal', location="top")
cb_C.set_ticks(10**np.unique(np.floor(np.log10(locals()[f'Ns_{s}'])))[1::2],minor=False)
cb_C.set_label("".join([r'$\bar{N}_{s}$']))


#%% subfig D

s = 'n_crit'
cmap_D = cmap_n_crit

temp_shape = locals()[f'lcs_ratio_{s}'].shape[1]
for i in range(temp_shape)[::-1]:
    ax_D.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][level_of_accuracy,i,:],color = cmap_D((temp_shape-i)/(1e-8+temp_shape-1)))
ax_D.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][0,0,:],color = 'k', linestyle = ':')
ax_D.set_xlim(ax_D.get_xlim())
ax_D.xaxis.grid()
ax_D.set_yscale('log')
# ax_D.set_yticklabels([])
ax_D.set_xlabel(r'$N_{norm}^{train}$')
# ax_D.xaxis.set_label_position('top')
ax_D.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                       left=True, labelleft=True, right=False, labelright=False)
ax_D.set_ylabel(r'$MAE_{\mu_{y}\leftarrow\bar{z}}^{test}$'+' [kcal/mol]')
ax_D.set_xticks(x_ticks_N_norm)

# temp_shape = locals()[f'lcs_ratio_{s}'].shape[1]
# for i in range(temp_shape)[::-1]:
#     ax_D_inset.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][level_of_accuracy,i,:],color = cmap_D((temp_shape-i)/(1e-8+temp_shape-1)))
# ax_D_inset.plot(Ns_tr_norm,locals()[f'lcs_ratio_{s}'][0,0,:],color = 'k', linestyle = ':')
# ax_D_inset.set_yscale('log')
# ax_D_inset.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                        left=False, labelleft=False, right=True, labelright=True)
# ax_D_inset.set_xticks(ax_D.get_xticks())
# ax_D_inset.set_xlim(ax_D.get_xlim())
# ax_D_inset.xaxis.grid()
# ax_D_inset.set_xticklabels([])
# ax_D_inset.set_yticklabels([])

ax_D_sec = ax_D.secondary_xaxis('top')
ax_D_sec.set_xticks(x_ticks_N_norm)
ax_D_sec.set_xticklabels([d_norm_not_norm[i] for i in x_ticks_N_norm])
ax_D_sec.set_xlabel(r'$N_{}^{train}$')


cb_D = plt.colorbar(mappable=ScalarMappable(cmap=cmap_D.reversed(), norm = LogNorm(locals()[f'HSs_{s}'].min(),locals()[f'HSs_{s}'].max())), ax=ax_D, orientation='horizontal', location="top")
cb_D.set_label("".join([r'$h/\sigma_{y}$']))


#%% Set limits

ylim_lcs = [np.inf,-np.inf]
for ax in [ax_A,ax_B,ax_C,ax_D]:
        ylim_lcs = [np.min([ylim_lcs[0],np.min(ax.get_ylim())]),
                    np.max([ylim_lcs[1],np.max(ax.get_ylim())])]
for ax in [ax_A,ax_B,ax_C,ax_D]:
    ax.set_ylim(ylim_lcs)


# ylim_lcs_ratio = [np.inf,-np.inf]
# for ax in [ax_A_inset,ax_B_inset,ax_C_inset,ax_D_inset]:
#         ylim_lcs_ratio = [np.min([ylim_lcs_ratio[0],np.min(ax.get_ylim())]),
#                           np.max([ylim_lcs_ratio[1],np.max(ax.get_ylim())])]
# for ax in [ax_A_inset,ax_B_inset,ax_C_inset,ax_D_inset]:
#     ax.set_ylim(ylim_lcs_ratio)


#%% Colorbars

# colornorm = LogNorm(vmin=datalim_lcs[0],vmax=datalim_lcs[1])

# t1,t2 = np.meshgrid(np.linspace(*ax_A.get_xlim(),100),np.logspace(*np.log10(ax_A.get_ylim()),100))
# for ax in [ax_A,ax_B,ax_C,ax_D]:
#     ax.pcolormesh(t1,t2,t2, norm=colornorm,cmap = cmap, alpha = alpha_main,zorder=0)

# CB_y = plt.colorbar(mappable=ScalarMappable(cmap=cmap, norm = colornorm), alpha = alpha_main, cax=ax_CM_y, extend = 'both')
# CB_y.set_label(r'$MAE_{\mu_{y}\leftarrow\bar{z}}^{test}$'+' [kcal/mol] (main figures)')
# CB_y.ax.yaxis.set_label_position('left')
# CB_y.ax.tick_params(which='both',top=False, labeltop=False, bottom=False, labelbottom=False,
#                     left=True, labelleft=True, right=False, labelright=False)


# colornorm_inset = LogNorm(vmin=1,vmax=datalim_lcs_ratio[1])

# t1,t2 = np.meshgrid(np.linspace(*ax_A_inset.get_xlim(),100),np.logspace(*np.log10(ax_A_inset.get_ylim()),100))
# for ax in [ax_A_inset,ax_B_inset,ax_C_inset,ax_D_inset]:
#     ax.pcolormesh(t1,t2,t2, norm=colornorm_inset,cmap = cmap, alpha = alpha_main,zorder=0)

# CB_y_inset = plt.colorbar(mappable=ScalarMappable(cmap=cmap, norm = colornorm_inset), alpha = 0, cax=ax_CM_y_inset, extend = 'both')
# CB_y_inset.set_label(r'$MAE_{\mu_{y}\leftarrow\bar{z}}^{test}\ /\ MAE_{\mu_{y}\leftarrow\mu_{y}}^{test}$'+' [-] (insets)')

# sep_CM_line = Line2D([0.5,0.5],[-0.05,1.05],color='k',linewidth=0.8,
#                      transform=transforms.blended_transform_factory(ax_CM_y.transAxes, ax_CM_y.transAxes),zorder=np.inf,clip_on=False)
# ax_CM_y.add_artist(sep_CM_line)
# ax_CM_y_inset.set_yscale('log')
# ax_CM_y_inset.set_ybound(colornorm_inset.vmin,colornorm_inset.vmax)

# ax_CM_y.right_ax.set_ylim(colornorm_inset.vmin,colornorm_inset.vmax)


#%% Text

# fig.canvas.draw()
# fig.set_constrained_layout(False)


temp = 0.015

label_A = plt.text(temp, 1-temp,
                    'A', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=subfig_A.transSubfigure, wrap=False)
label_A.set_in_layout(False)

label_B = plt.text(temp, 1-temp,
                    'B', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=subfig_B.transSubfigure, wrap=False)
label_B.set_in_layout(False)

label_C = plt.text(temp, 1-temp,
                    'C', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=subfig_C.transSubfigure, wrap=False)
label_C.set_in_layout(False)

label_D = plt.text(temp, 1-temp,
                    'D', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=subfig_D.transSubfigure, wrap=False)
label_D.set_in_layout(False)



#%% Save
plt.savefig('SF7'+".svg",dpi = 300, format="svg",transparent=True)
plt.savefig('SF7'+'.png',dpi = 400, format='png')