#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:01:37 2024

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
data_name = 'uni_2D_lcs.npz'

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


#%% Load Data
with np.load(data_path+data_name,allow_pickle=True) as data:
    #for k in data.keys():
        lcs = data['lcs']
        lcs = lcs.reshape(lcs.shape[0],51,51,lcs.shape[2],lcs.shape[3],lcs.shape[4])
        HSs = data['HSs'][0,:,0,0,0].reshape(51,51)
        Ns = data['Ns'][0,:,0,0,0].reshape(51,51)
        lcs_ratio = lcs/lcs[:1,...]

tt = 3

bias = np.median(lcs_ratio,axis=[3,4])[1,...,tt]
var = np.median(lcs_ratio,axis=[3,4])[2,...,tt]
bias_plus_var = np.median(lcs_ratio,axis=[3,4])[3,...,tt]

tt = tt*10 #+ 1

for s in ['eq_equidistant','contour_crit_equidistant','n_crit','h_crit']:
    with np.load(data_path+f'{data_name.split("_")[0]}_{s}_out.npz',allow_pickle=True) as data:
        #for k in data.keys():
            temp = np.median(data['lcs'],axis=[2,3])[...,tt]
            temp = temp/temp[:1,...]
            locals()[f'bias_{s}'] = np.column_stack((data['Ns'][0,:,0,0,0],data['HSs'][0,:,0,0,0],temp[1,:]))
            locals()[f'var_{s}'] = np.column_stack((data['Ns'][0,:,0,0,0],data['HSs'][0,:,0,0,0],temp[2,:]))
            locals()[f'bias_plus_var_{s}'] = np.column_stack((data['Ns'][0,:,0,0,0],data['HSs'][0,:,0,0,0],temp[3,:]))


#%%
plt.close('all')

fig = plt.figure(layout='constrained', figsize=(8.9, 8))#latex --> (5.5,9)

gs = fig.add_gridspec(2,3, hspace=0.12, wspace=0.02)
gs.set_height_ratios([1,1])
gs.set_width_ratios([1,0.325,1])

subfig_A = fig.add_subfigure(gs[0, 0])
subfig_B = fig.add_subfigure(gs[0, 2])
subfig_C = fig.add_subfigure(gs[1, 0])
subfig_D = fig.add_subfigure(gs[1, 2])
subfig_CM = fig.add_subfigure(gs[:, 1])

# subfig_A.set_facecolor('1.')
# subfig_B.set_facecolor('0.2')
# subfig_C.set_facecolor('0.4')
# subfig_D.set_facecolor('0.6')
# subfig_CM.set_facecolor('0.8')


gs_A = subfig_A.add_gridspec(2,2)
gs_A.set_height_ratios([gs_subfig_ABC_ratio.min(),gs_subfig_ABC_ratio.max(),])
gs_A.set_width_ratios([gs_subfig_ABC_ratio.min(),gs_subfig_ABC_ratio.max(),])

gs_B = subfig_B.add_gridspec(2,2)
gs_B.set_height_ratios([gs_subfig_ABC_ratio.min(),gs_subfig_ABC_ratio.max(),])
gs_B.set_width_ratios([gs_subfig_ABC_ratio.max(),gs_subfig_ABC_ratio.min(),])

gs_C = subfig_C.add_gridspec(2,2)
gs_C.set_height_ratios([gs_subfig_ABC_ratio.max(),gs_subfig_ABC_ratio.min(),])
gs_C.set_width_ratios([gs_subfig_ABC_ratio.min(),gs_subfig_ABC_ratio.max(),])

gs_D = subfig_D.add_gridspec(3,2)
gs_D.set_height_ratios([1,0.05,1])

gs_CM = subfig_CM.add_gridspec(1,1)


ax_A = subfig_A.add_subplot(gs_A[1,1])
ax_A_x = subfig_A.add_subplot(gs_A[0, 1])
ax_A_y = subfig_A.add_subplot(gs_A[1,0])

ax_B = subfig_B.add_subplot(gs_B[1,0])
ax_B_x = subfig_B.add_subplot(gs_B[0, 0])
ax_B_y = subfig_B.add_subplot(gs_B[1,1])

ax_C = subfig_C.add_subplot(gs_C[0,1])
ax_C_x = subfig_C.add_subplot(gs_C[1, 1])
ax_C_y = subfig_C.add_subplot(gs_C[0,0])

ax_D_00 = subfig_D.add_subplot(gs_D[0, 0])
ax_D_01 = subfig_D.add_subplot(gs_D[0, 1])
ax_D_10 = subfig_D.add_subplot(gs_D[-1, 0])
ax_D_11 = subfig_D.add_subplot(gs_D[-1, 1])
ax_D_legend = subfig_D.add_subplot(gs_D[1, :])

ax_CM = subfig_CM.add_subplot(gs_CM[:, :])

#%% Colorbar
cmap_min = np.nanmin(var) # exlude bias
cmap_max = np.nanmax(bias_plus_var)

temp1 = []
for i in np.arange(np.floor(np.log10(cmap_min)),np.ceil(np.log10(cmap_max))):
    for ii in np.linspace(0,10,11)[1:]/1:
        temp1.append(ii*10**i)
temp1 = np.array(temp1)[(temp1<cmap_max)&(temp1>cmap_min)]
temp2 = np.concatenate((np.array([cmap_min,cmap_max]),temp1))
tot_levels = np.unique(np.sort(temp2))
tot_levels_f = np.logspace(*np.log10([cmap_min,cmap_max]),400)

colornorm = LogNorm(vmin=cmap_min,vmax=cmap_max)
colormap = ScalarMappable(cmap=cmap, norm=colornorm)

cbar = subfig_CM.colorbar(
            colormap,
            cax=ax_CM, orientation='vertical',
            extend='both',# extendfrac='auto',
)

# remove cbar, could be improved
temp = cbar.ax.get_children()
# print(temp)
temp[3].remove()
temp[2].remove()
# temp[1].remove()
# temp[0].remove()

ax_CM.set_ylim(cmap_min,cmap_max)#cmap_max)
ax_CM.set_xlim([0,1])

# 
pmeshcolor_span = np.logspace(*np.log10([cmap_min,cmap_max]),100)
pmeshcolor_span = np.unique(np.sort(pmeshcolor_span))


t1,t2 = np.meshgrid(np.linspace(0.,0.5),pmeshcolor_span)
ax_CM.pcolormesh(t1,t2,t2, norm=colornorm,cmap = cmap, alpha = alpha_main,zorder=np.inf)
ax_CM.contour(t1,t2,t2,levels=tot_levels, alpha = alpha_contour, linewidths=0.5,zorder=np.inf,cmap = cmap, norm=colornorm)
t1,t2 = np.meshgrid(np.linspace(0.5,1),pmeshcolor_span)
ax_CM.pcolormesh(t1,t2,t2, norm=colornorm,cmap = cmap, alpha = alpha_sec,transform=transforms.blended_transform_factory(ax_CM.transData, ax_CM.transData),zorder=np.inf)


patches = []
patches.append(PathPatch(Path([[0.5,1],[1,1],[0.5,1.05]]), transform=ax_CM.transAxes, edgecolor = 'none', alpha = alpha_sec, facecolor=cmap(1.),zorder=0))
patches.append(PathPatch(Path([[0.5,0],[1,0],[0.5,-0.05]]), transform=ax_CM.transAxes, edgecolor = 'none', alpha = alpha_sec, facecolor=cmap(0.),zorder=0))
patches.append(PathPatch(Path([[0.,1],[0.5,1],[0.5,1.05]]), transform=ax_CM.transAxes, edgecolor = 'none', alpha = alpha_main, facecolor=cmap(1.),zorder=0))
patches.append(PathPatch(Path([[0.,0],[0.5,0],[0.5,-0.05]]), transform=ax_CM.transAxes, edgecolor = 'none', alpha = alpha_main, facecolor=cmap(0.),zorder=0))
for p in patches: subfig_CM.patches.append(p)

sep_CM_line = Line2D([0.5,0.5],[-0.05,1.05],color='k',linewidth=0.8,
                      transform=transforms.blended_transform_factory(ax_CM.transAxes, ax_CM.transAxes),zorder=np.inf)
subfig_CM.lines.append(sep_CM_line)

if tot_levels[0] == np.round(tot_levels[0]):
    ax_CM.set_yticks(tot_levels,minor=True)
else:
    ax_CM.set_yticks(tot_levels[1:-1],minor=True)


#%% subfig A
s = 'bias'

temp = locals()[s].copy()
temp[np.abs(temp)<cmap_min]=cmap_min
ax_A.contourf(Ns,HSs,temp,levels=tot_levels_f,alpha = 1,zorder=0,cmap = cmap, norm=colornorm)
rect_A = Rectangle((0, 0), width=1, height=1, transform=ax_A.transAxes, alpha = 1-alpha_main, zorder = 1, facecolor='white', edgecolor='black', linewidth = 0.8)
ax_A.add_patch(rect_A)
# ax_A.contour(Ns,HSs,temp,levels=tot_levels, alpha = alpha_contour, linewidths=alpha_main, zorder=2, cmap = cmap, norm=colornorm)
ax_A.loglog(locals()[f'{s}_contour_crit_equidistant'][:,0],locals()[f'{s}_contour_crit_equidistant'][:,1],color = color_contour_crit, **d_style[s], **d_style_contour)
ax_A.loglog(locals()[f'{s}_eq_equidistant'][:,0],locals()[f'{s}_eq_equidistant'][:,1],color = color_eq, **d_style[s], **d_style_contour)
# ax_A.loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,1],color = color_h_crit, **d_style[s], **d_style_contour)
# ax_A.loglog(locals()[f'{s}_n_crit'][:,0],locals()[f'{s}_n_crit'][:,1],color = color_n_crit, **d_style[s], **d_style_contour)

ax_A.set_xscale('log')
ax_A.set_yscale('log')
ax_A.set_xticklabels([])
ax_A.set_yticklabels([])
ax_A.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False)


t1,t2 = np.meshgrid(np.unique(Ns),tot_levels_f)
ax_A_x.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,Ns.shape[0],step_detials_sec):
    ax_A_x.plot(Ns[:,i],locals()[s][:,i],'k',linewidth=0.25)
ax_A_x.loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,2],color = color_h_crit, **d_style[s], **d_style_contour)
ax_A_x.set_xscale('log')
ax_A_x.set_yscale('log')
ax_A_x.set_ylim(cmap_min,cmap_max)
ax_A_x.set_yticklabels([])
ax_A_x.set_xlabel(r'$\bar{N}_{s}$')
ax_A_x.xaxis.set_label_position("top")
ax_A_x.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False)


t1,t2 = np.meshgrid(np.unique(HSs),tot_levels_f)
ax_A_y.pcolormesh(t2.T, t1.T, t2.T, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,Ns.shape[0],step_detials_sec):
    ax_A_y.plot(locals()[s][i,:],HSs[i,:],'k',linewidth=0.25)
ax_A_y.loglog(locals()[f'{s}_n_crit'][:,2],locals()[f'{s}_n_crit'][:,1],color = color_n_crit, **d_style[s], **d_style_contour)
ax_A_y.set_xscale('log')
ax_A_y.set_yscale('log')
ax_A_y.set_xlim(cmap_min,cmap_max)
ax_A_y.set_xticklabels([])
ax_A_y.set_ylabel('$h/\sigma_y$')
ax_A_y.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False)
ax_A_y.invert_xaxis()


#add h_crit/n_crit ticks
ax_A_secy = ax_A.secondary_yaxis(location='left')
ax_A_secy.set_yticks([locals()[f'{s}_h_crit'][:,1][0]])
ax_A_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_A_secy.set_alpha(d_style_contour['alpha'])
ax_A_secy.yaxis.set_ticklabels([])

ax_A_secx = ax_A.secondary_xaxis(location='top')
ax_A_secx.set_xticks([locals()[f'{s}_n_crit'][:,0][0]])
ax_A_secx.tick_params(axis='x',direction='out',color=color_n_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_A_secx.set_alpha(d_style_contour['alpha'])
ax_A_secx.xaxis.set_ticklabels([])

ax_A_y_secy = ax_A_y.secondary_yaxis(location='left')
ax_A_y_secy.set_yticks([locals()[f'{s}_h_crit'][:,1][0]])
ax_A_y_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_A_y_secy.set_alpha(d_style_contour['alpha'])
ax_A_y_secy.yaxis.set_ticklabels([])

ax_A_x_secx = ax_A_x.secondary_xaxis(location='top')
ax_A_x_secx.set_xticks([locals()[f'{s}_n_crit'][:,0][0]])
ax_A_x_secx.tick_params(axis='x',direction='out',color=color_n_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_A_x_secx.set_alpha(d_style_contour['alpha'])
ax_A_x_secx.xaxis.set_ticklabels([])


#%% subfig B
s = 'var'

ax_B.contourf(Ns,HSs,locals()[s],levels=tot_levels_f,alpha = 1,zorder=0,cmap = cmap, norm = colornorm)
rect_B = Rectangle((0, 0), width=1, height=1, transform=ax_B.transAxes, alpha = 1-alpha_main, zorder = 1, facecolor='white', edgecolor='black', linewidth = 0.8)
ax_B.add_patch(rect_B)
ax_B.contour(Ns,HSs,locals()[s],levels=tot_levels, alpha = alpha_contour, linewidths=alpha_main, zorder=2, cmap = cmap, norm=colornorm)
ax_B.loglog(locals()[f'{s}_contour_crit_equidistant'][:,0],locals()[f'{s}_contour_crit_equidistant'][:,1],color = color_contour_crit, **d_style[s], **d_style_contour)
ax_B.loglog(locals()[f'{s}_eq_equidistant'][:,0],locals()[f'{s}_eq_equidistant'][:,1],color = color_eq, **d_style[s], **d_style_contour)
# ax_B.loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,1],color = color_h_crit, **d_style[s], **d_style_contour)
# ax_B.loglog(locals()[f'{s}_n_crit'][:,0],locals()[f'{s}_n_crit'][:,1],color = color_n_crit, **d_style[s], **d_style_contour)

ax_B.set_xscale('log')
ax_B.set_yscale('log')
ax_B.set_xticklabels([])
ax_B.set_yticklabels([])
ax_B.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                              left=False, labelleft=False, right=True, labelright=True,rotation = 90)


t1,t2 = np.meshgrid(np.unique(Ns),tot_levels_f)
ax_B_x.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,Ns.shape[0],step_detials_sec):
    ax_B_x.plot(Ns[:,i],locals()[s][:,i],'k',linewidth=0.25)
ax_B_x.loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,2],color = color_h_crit, **d_style[s], **d_style_contour)
ax_B_x.set_xscale('log')
ax_B_x.set_yscale('log')
ax_B_x.set_ylim(cmap_min,cmap_max)
ax_B_x.set_yticklabels([])
ax_B_x.set_xlabel(r'$\bar{N}_{s}$')
ax_B_x.xaxis.set_label_position("top")
ax_B_x.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                                left=False, labelleft=False, right=True, labelright=True)


t1,t2 = np.meshgrid(np.unique(HSs),tot_levels_f)
ax_B_y.pcolormesh(t2.T, t1.T, t2.T, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,Ns.shape[0],step_detials_sec):
    ax_B_y.plot(locals()[s][i,:],HSs[i,:],'k',linewidth=0.25)
ax_B_y.loglog(locals()[f'{s}_n_crit'][:,2],locals()[f'{s}_n_crit'][:,1],color = color_n_crit, **d_style[s], **d_style_contour)
ax_B_y.set_xscale('log')
ax_B_y.set_yscale('log')
ax_B_y.set_xlim(cmap_min,cmap_max)
ax_B_y.set_xticklabels([])
ax_B_y.set_ylabel('$h/\sigma_y$')
ax_B_y.yaxis.set_label_position("right")
ax_B_y.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                                left=False, labelleft=False, right=True, labelright=True)
ax_B_y.invert_xaxis()


#add h_crit/n_crit ticks
ax_B_secy = ax_B.secondary_yaxis(location='right')
ax_B_secy.set_yticks([locals()[f'{s}_h_crit'][:,1][0]])
ax_B_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_B_secy.set_alpha(d_style_contour['alpha'])
ax_B_secy.yaxis.set_ticklabels([])

ax_B_secx = ax_B.secondary_xaxis(location='top')
ax_B_secx.set_xticks([locals()[f'{s}_n_crit'][:,0][0]])
ax_B_secx.tick_params(axis='x',direction='out',color=color_n_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_B_secx.set_alpha(d_style_contour['alpha'])
ax_B_secx.xaxis.set_ticklabels([])

ax_B_y_secy = ax_B_y.secondary_yaxis(location='right')
ax_B_y_secy.set_yticks([locals()[f'{s}_h_crit'][:,1][0]])
ax_B_y_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_B_y_secy.set_alpha(d_style_contour['alpha'])
ax_B_y_secy.yaxis.set_ticklabels([])

ax_B_x_secx = ax_B_x.secondary_xaxis(location='top')
ax_B_x_secx.set_xticks([locals()[f'{s}_n_crit'][:,0][0]])
ax_B_x_secx.tick_params(axis='x',direction='out',color=color_n_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_B_x_secx.set_alpha(d_style_contour['alpha'])
ax_B_x_secx.xaxis.set_ticklabels([])


#%% subfig C
s = 'bias_plus_var'

ax_C.contourf(Ns,HSs,locals()[s],levels=tot_levels_f,alpha = 1,zorder=0,cmap = cmap, norm = colornorm)
rect_C = Rectangle((0, 0), width=1, height=1, transform=ax_C.transAxes, alpha = 1-alpha_main, zorder = 1, facecolor='white', edgecolor='black', linewidth = 0.8)
ax_C.add_patch(rect_C)
ax_C.contour(Ns,HSs,locals()[s],levels=tot_levels, alpha = alpha_contour, linewidths=alpha_main, zorder=2, cmap = cmap, norm=colornorm)

points = locals()[f'{s}_contour_crit_equidistant'][:,:2].reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
dydx = np.arange(points.shape[0])
lc = LineCollection(segments, cmap=cmap_contour_crit, norm=plt.Normalize(dydx.min(),dydx.max()), **d_style_contour)
lc.set_array(dydx)
line = ax_C.add_collection(lc)

points = locals()[f'{s}_eq_equidistant'][:,:2].reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
dydx = np.arange(points.shape[0])
lc = LineCollection(segments, cmap=cmap_eq, norm=plt.Normalize(dydx.min(),dydx.max()), **d_style_contour)
lc.set_array(dydx)
line = ax_C.add_collection(lc)

# ax_C.loglog(locals()[f'{s}_contour_crit_equidistant'][:,0],locals()[f'{s}_contour_crit_equidistant'][:,1],color = color_contour_crit, **d_style[s], **d_style_contour)
# ax_C.loglog(locals()[f'{s}_eq_equidistant'][:,0],locals()[f'{s}_eq_equidistant'][:,1],color = color_eq, **d_style[s], **d_style_contour)
# ax_C.loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,1],color = color_h_crit, **d_style[s], **d_style_contour)
# ax_C.loglog(locals()[f'{s}_n_crit'][:,0],locals()[f'{s}_n_crit'][:,1],color = color_n_crit, **d_style[s], **d_style_contour)

ax_C.set_xscale('log')
ax_C.set_yscale('log')
ax_C.set_xticklabels([])
ax_C.set_yticklabels([])
ax_C.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                              left=True, labelleft=True, right=False, labelright=False,rotation = 90)

t1,t2 = np.meshgrid(np.unique(Ns),tot_levels_f)
ax_C_x.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,Ns.shape[0],step_detials_sec):
    ax_C_x.plot(Ns[:,i],locals()[s][:,i],'k',linewidth=0.25)
ax_C_x.set_xscale('log')
ax_C_x.set_yscale('log')
ax_C_x.set_ylim(cmap_min,cmap_max)
ax_C_x.set_yticklabels([])
ax_C_x.set_xlabel(r'$\bar{N}_{s}$')
ax_C_x.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                                left=True, labelleft=True, right=False, labelright=False)

points = locals()[f'{s}_h_crit'][:,::2].reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
dydx = np.arange(points.shape[0])
lc = LineCollection(segments, cmap=cmap_h_crit, norm=plt.Normalize(dydx.min(),dydx.max()),zorder=np.inf, **d_style_contour)
lc.set_array(dydx)
line = ax_C_x.add_artist(lc)



t1,t2 = np.meshgrid(np.unique(HSs),tot_levels_f)
ax_C_y.pcolormesh(t2.T, t1.T, t2.T, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,Ns.shape[0],step_detials_sec):
    ax_C_y.plot(locals()[s][i,:],HSs[i,:],'k',linewidth=0.25)
ax_C_y.set_xscale('log')
ax_C_y.set_yscale('log')
ax_C_y.set_xlim(cmap_min,cmap_max)
ax_C_y.set_xticklabels([])
ax_C_y.set_ylabel('$h/\sigma_y$')
ax_C_y.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                                left=True, labelleft=True, right=False, labelright=False)
ax_C_y.invert_xaxis()

points = np.column_stack([locals()[f'{s}_n_crit'][:,2],
                          locals()[f'{s}_n_crit'][:,1]]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
dydx = np.arange(points.shape[0])
lc = LineCollection(segments, cmap=cmap_n_crit, norm=plt.Normalize(dydx.min(),dydx.max()),zorder=np.inf, **d_style_contour)
lc.set_array(dydx[::-1])
line = ax_C_y.add_collection(lc)


#add h_crit/n_crit ticks
ax_C_secy = ax_C.secondary_yaxis(location='left')
ax_C_secy.set_yticks([locals()[f'{s}_h_crit'][:,1][0]])
ax_C_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_C_secy.set_alpha(d_style_contour['alpha'])
ax_C_secy.yaxis.set_ticklabels([])

ax_C_secx = ax_C.secondary_xaxis(location='bottom')
ax_C_secx.set_xticks([locals()[f'{s}_n_crit'][:,0][0]])
ax_C_secx.tick_params(axis='x',direction='out',color=color_n_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_C_secx.set_alpha(d_style_contour['alpha'])
ax_C_secx.xaxis.set_ticklabels([])

ax_C_y_secy = ax_C_y.secondary_yaxis(location='left')
ax_C_y_secy.set_yticks([locals()[f'{s}_h_crit'][:,1][0]])
ax_C_y_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_C_y_secy.set_alpha(d_style_contour['alpha'])
ax_C_y_secy.yaxis.set_ticklabels([])

ax_C_x_secx = ax_C_x.secondary_xaxis(location='bottom')
ax_C_x_secx.set_xticks([locals()[f'{s}_n_crit'][:,0][0]])
ax_C_x_secx.tick_params(axis='x',direction='out',color=color_n_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_C_x_secx.set_alpha(d_style_contour['alpha'])
ax_C_x_secx.xaxis.set_ticklabels([])


#%% subfig D

# D00
s = 'bias_plus_var'
points = np.column_stack([np.arange(locals()[f'{s}_eq_equidistant'].shape[0]),
                          locals()[f'{s}_eq_equidistant'][:,2]]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
dydx = np.arange(points.shape[0])
lc = LineCollection(segments, cmap=cmap_eq, norm=plt.Normalize(dydx.min(),dydx.max()), **d_style_contour)
lc.set_array(dydx)
line = ax_D_00.add_collection(lc)

for s in ['bias','var']:#,'bias_plus_var']:
    ax_D_00.semilogy(np.arange(locals()[f'{s}_eq_equidistant'][:,0].shape[0]),locals()[f'{s}_eq_equidistant'][:,2],color = color_eq, **d_style[s], **d_style_contour)

ax_D_00.set_xlim(ax_D_00.get_xlim())
t1,t2 = np.meshgrid(np.linspace(*ax_D_00.get_xlim(),Ns.shape[0]),tot_levels_f)
ax_D_00.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
# ax_D_00.set_xscale('log')
ax_D_00.set_yscale('log')
ax_D_00.set_ylim(cmap_min,cmap_max)
ax_D_00.set_xticklabels([])
ax_D_00.set_xticks(np.arange(locals()[f'{s}_eq_equidistant'][:,0].shape[0])[::10],minor=True)
# ax_D_00.set_xlabel('$Equidistant\ points$',alpha=0)
ax_D_00.xaxis.set_label_position("top")
ax_D_00.set_yticklabels([])
ax_D_00.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                                  left=True, labelleft=True, right=False, labelright=False)

# D01
s = 'bias_plus_var'
points = np.column_stack([np.arange(locals()[f'{s}_contour_crit_equidistant'].shape[0]),
                          locals()[f'{s}_contour_crit_equidistant'][:,2]]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
dydx = np.arange(points.shape[0])
lc = LineCollection(segments, cmap=cmap_contour_crit, norm=plt.Normalize(dydx.min(),dydx.max()), **d_style_contour)
lc.set_array(dydx)
line = ax_D_01.add_collection(lc)

for s in ['bias','var']:#,'bias_plus_var']:
    ax_D_01.semilogy(np.arange(locals()[f'{s}_contour_crit_equidistant'][:,0].shape[0]),locals()[f'{s}_contour_crit_equidistant'][:,2],color = color_contour_crit, **d_style[s], **d_style_contour)

ax_D_01.set_xlim(ax_D_01.get_xlim())
t1,t2 = np.meshgrid(np.linspace(*ax_D_01.get_xlim(),Ns.shape[0]),tot_levels_f)
ax_D_01.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
# ax_D_01.set_xscale('log')
ax_D_01.set_yscale('log')
ax_D_01.set_ylim(cmap_min,cmap_max)
ax_D_01.set_xticklabels([])
ax_D_01.set_xticks(np.arange(locals()[f'{s}_contour_crit_equidistant'][:,0].shape[0])[::10],minor=True)
ax_D_01.set_ylabel('$h/\sigma_y$',alpha=0)
ax_D_01.yaxis.set_label_position("right")
ax_D_01.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                                  left=False, labelleft=False, right=True, labelright=True)
plt.setp(ax_D_01.get_yticklabels(which=u'both'),alpha=0)


# D10
s = 'bias_plus_var'
points = np.column_stack([locals()[f'{s}_h_crit'][:,0],
                          locals()[f'{s}_h_crit'][:,2]]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
dydx = np.arange(points.shape[0])
lc = LineCollection(segments, cmap=cmap_h_crit, norm=plt.Normalize(dydx.min(),dydx.max()), **d_style_contour)
lc.set_array(dydx)
line = ax_D_10.add_collection(lc)

for s in ['bias','var']:#,'bias_plus_var']:
    ax_D_10.loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,2],color = color_h_crit, **d_style[s], **d_style_contour)

ax_D_10.set_xlim(ax_D_10.get_xlim())
t1,t2 = np.meshgrid(np.linspace(*ax_D_10.get_xlim(),Ns.shape[0]),tot_levels_f)
ax_D_10.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
ax_D_10.set_xscale('log')
ax_D_10.set_yscale('log')
ax_D_10.set_ylim(cmap_min,cmap_max)
# ax_D_10.set_xticklabels([])
ax_D_10.set_xlabel(r'$\bar{N}_{s}$')
ax_D_10.set_yticklabels([])
ax_D_10.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                                  left=True, labelleft=True, right=False, labelright=False)

# D11
s = 'bias_plus_var'
points = np.column_stack([locals()[f'{s}_n_crit'][:,1],
                          locals()[f'{s}_n_crit'][:,2]]).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
dydx = np.arange(points.shape[0])
lc = LineCollection(segments, cmap=cmap_n_crit, norm=plt.Normalize(dydx.min(),dydx.max()), **d_style_contour)
lc.set_array(dydx[::-1])
line = ax_D_11.add_collection(lc)

for s in ['bias','var']:#,'bias_plus_var']:
    ax_D_11.loglog(locals()[f'{s}_n_crit'][:,1],locals()[f'{s}_n_crit'][:,2],color = color_n_crit, **d_style[s], **d_style_contour)

ax_D_11.set_xlim(ax_D_11.get_xlim())
t1,t2 = np.meshgrid(np.linspace(*ax_D_11.get_xlim(),Ns.shape[0]),tot_levels_f)
ax_D_11.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
ax_D_11.set_xscale('log')
ax_D_11.set_yscale('log')
ax_D_11.set_ylim(cmap_min,cmap_max)
# ax_D_11.set_xticklabels([])
# ax_D_11.set_yticklabels([])
ax_D_11.yaxis.set_label_position("right")
ax_D_11.set_xlabel('$h/\sigma_y$')
ax_D_11.set_yticklabels([])
ax_D_11.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                                  left=False, labelleft=False, right=True, labelright=True)


# D_legend

ax_D_legend.tick_params(which='both',top=True, labeltop=True, bottom=True, labelbottom=True,
                        left=True, labelleft=True, right=True, labelright=True)
ax_D_legend.set_xticklabels([])
ax_D_legend.set_yticklabels([])
ax_D_legend.set_axis_off()


legend_elements = [
                    Line2D([0], [0], color='k', label='B.', **d_style['bias']),
                    Line2D([0], [0], color='k', label='V.', **d_style['var']),
                    Line2D([0], [0], color='k', label='B.+V.', **d_style['bias_plus_var']),
                    ]

ax_D_legend.legend(handles=legend_elements,ncol=3,loc='center',handlelength=1)


temp = 0.05

caption_D_00 = plt.text(1-temp, 1-temp, r'$@R_{D}=1$', horizontalalignment='right', fontsize=fontsize,
                     verticalalignment='top', transform=ax_D_00.transAxes)
caption_D_00.set_in_layout(False)

caption_D_01 = plt.text(temp, 1-temp, r'$@B.+V.$', horizontalalignment='left', fontsize=fontsize,
                     verticalalignment='top', transform=ax_D_01.transAxes)
caption_D_01.set_in_layout(False)

caption_D_10 = plt.text(1-temp, 1-temp, r'$@h_{norm}^{crit}$', horizontalalignment='right', fontsize=fontsize,
                     verticalalignment='top', transform=ax_D_10.transAxes)
caption_D_10.set_in_layout(False)

caption_D_11 = plt.text(temp, 1-temp, r'$@\bar{N}_{s}^{crit}$', horizontalalignment='left', fontsize=fontsize,
                     verticalalignment='top', transform=ax_D_11.transAxes)
caption_D_11.set_in_layout(False)


# #%% subfig D (old)

#         # axs[0,0].loglog(locals()[f'{s}_eq_equidistant'][:,0],locals()[f'{s}_eq_equidistant'][:,2],color = color_eq, **d_style[s])
#         # axs[0,1].loglog(locals()[f'{s}_contour_crit_equidistant'][:,1],locals()[f'{s}_contour_crit_equidistant'][:,2],color = color_contour_crit, **d_style[s])
#         # # axs[0,1].semilogy(locals()[f'{s}_contour_crit_equidistant'][:,2],color = color_contour_crit, **d_style[s])    
#         # axs[1,0].loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,2],color = color_h_crit, **d_style[s])
#         # axs[1,1].loglog(locals()[f'{s}_n_crit'][:,1],locals()[f'{s}_n_crit'][:,2],color = color_n_crit, **d_style[s])

# # D00
# for s in ['bias','var','bias_plus_var']:
#     ax_D_00.loglog(locals()[f'{s}_eq_equidistant'][:,0],locals()[f'{s}_eq_equidistant'][:,2],color = color_eq, **d_style[s])
# t1,_ = np.meshgrid(np.logspace(*np.log10(ax_D_00.get_xlim()),Ns.shape[0]),np.ones(Ns.shape[0]))
# ax_D_00.pcolormesh(t1.T, locals()['bias'], locals()['bias'], norm=colornorm,cmap=cmap,alpha=alpha_sec)
# ax_D_00.set_xscale('log')
# ax_D_00.set_yscale('log')
# ax_D_00.set_ylim(cmap_min,cmap_max)
# ax_D_00.set_xticklabels([])
# # ax_D_00.set_xlabel(r'$\bar{N}_{s}$')
# # ax_D_00.xaxis.set_label_position("top")
# ax_D_00.set_yticklabels([])
# ax_D_00.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                                   left=True, labelleft=True, right=False, labelright=False)



# # D01
# for s in ['bias','var','bias_plus_var']:
#     ax_D_01.loglog(locals()[f'{s}_n_crit'][:,1],locals()[f'{s}_n_crit'][:,2],color = color_n_crit, **d_style[s])
# t1,_ = np.meshgrid(np.logspace(*np.log10(ax_D_01.get_xlim()),Ns.shape[0]),np.ones(Ns.shape[0]))
# ax_D_01.pcolormesh(t1.T, locals()['bias'], locals()['bias'], norm=colornorm,cmap=cmap,alpha=alpha_sec)
# ax_D_01.set_xscale('log')
# ax_D_01.set_yscale('log')
# ax_D_01.set_ylim(cmap_min,cmap_max)
# ax_D_01.set_xticklabels([])
# # ax_D_01.set_xlabel('$h/\sigma_y$')
# # ax_D_01.xaxis.set_label_position("top")
# # ax_D_01.set_yticklabels([])
# ax_D_01.yaxis.set_label_position("right")
# ax_D_01.set_ylabel('$h/\sigma_y$',alpha=0)
# ax_D_01.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                                   left=False, labelleft=False, right=True, labelright=True)
# plt.setp(ax_D_01.get_yticklabels(which=u'both'),alpha=0)

# # D10
# for s in ['bias','var','bias_plus_var']:
#     ax_D_10.loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,2],color = color_h_crit, **d_style[s])

# t1,_ = np.meshgrid(np.logspace(*np.log10(ax_D_10.get_xlim()),Ns.shape[0]),np.ones(Ns.shape[0]))
# ax_D_10.pcolormesh(t1.T, locals()['bias'], locals()['bias'], norm=colornorm,cmap=cmap,alpha=alpha_sec)
# ax_D_10.set_xscale('log')
# ax_D_10.set_yscale('log')
# ax_D_10.set_ylim(cmap_min,cmap_max)
# # ax_D_10.set_xticklabels([])
# ax_D_10.set_xlabel(r'$\bar{N}_{s}$')
# ax_D_10.set_yticklabels([])
# ax_D_10.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
#                                   left=True, labelleft=True, right=False, labelright=False)

# # D11
# for s in ['bias','var','bias_plus_var']:
#     ax_D_11.loglog(locals()[f'{s}_contour_crit_equidistant'][:,1],locals()[f'{s}_contour_crit_equidistant'][:,2],color = color_contour_crit, **d_style[s])
# ax_D_11.set_xlim(ax_D_01.get_xlim()[0],ax_D_11.get_xlim()[1])
# t1,_ = np.meshgrid(np.logspace(*np.log10(ax_D_11.get_xlim()),Ns.shape[0]),np.ones(Ns.shape[0]))
# ax_D_11.pcolormesh(t1.T, locals()['bias'], locals()['bias'], norm=colornorm,cmap=cmap,alpha=alpha_sec)
# ax_D_11.set_xscale('log')
# ax_D_11.set_yscale('log')
# ax_D_11.set_ylim(cmap_min,cmap_max)
# # ax_D_11.set_xticklabels([])
# ax_D_11.set_xlabel('$h/\sigma_y$')
# ax_D_11.set_yticklabels([])
# ax_D_11.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
#                                   left=False, labelleft=False, right=True, labelright=True)

# ax_CM.tick_params(which='both', left=True, labelleft=True, right=True, labelright=True) # need to be located here
# fig.canvas.draw()
# fig.set_constrained_layout(False)

# trans = ax_D_01.transData.transform([ax_D_11.get_xlim()[1],1])
# trans = subfig_D.transSubfigure.inverted().transform(trans)
# # line1 = Line2D([trans[0],trans[0]], [0,1], color='r', lw = 5, transform=subfig_D.transSubfigure)
# # fig.lines.append(line1)
# # plt.show()

# temp1 = ax_D_11.get_position().bounds
# temp2 = np.array(temp1).reshape(2,2)
# temp2[1,0]=trans[0]-temp2[0,0]
# temp3 = transforms.Bbox(temp2)
# ax_D_11.remove()
# ax_D_11 = subfig_D.add_axes(temp2.flatten())

# for s in ['bias','var','bias_plus_var']:
#     ax_D_11.loglog(locals()[f'{s}_contour_crit_equidistant'][:,1],locals()[f'{s}_contour_crit_equidistant'][:,2],color = color_contour_crit, **d_style[s])
# ax_D_11.set_xlim(ax_D_01.get_xlim()[0],ax_D_11.get_xlim()[1])
# t1,_ = np.meshgrid(np.logspace(*np.log10(ax_D_11.get_xlim()),Ns.shape[0]),np.ones(Ns.shape[0]))
# ax_D_11.pcolormesh(t1.T, locals()['bias'], locals()['bias'], norm=colornorm,cmap=cmap,alpha=alpha_sec)
# ax_D_11.set_xscale('log')
# ax_D_11.set_yscale('log')
# ax_D_11.set_ylim(cmap_min,cmap_max)
# # ax_D_11.set_xticklabels([])
# ax_D_11.set_xlabel('$h/\sigma_y$')
# ax_D_11.set_yticklabels([])
# ax_D_11.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
#                                   left=False, labelleft=False, right=True, labelright=True)


# #%% Legend

# legend_elements = [
#                     Line2D([0], [0], color='k', label='Bias (B.)', **d_style['bias']),
#                     Line2D([0], [0], color='k', label='Var. (V.)', **d_style['var']),
#                     Line2D([0], [0], color=color_eq, label='@R_{D}=1'),
#                     Line2D([0], [0], color=color_n_crit, label='@$N_{crit}$'),
#                     Line2D([0], [0], color=color_h_crit, label='@$h_{crit}$'),
#                     Line2D([0], [0], color=color_contour_crit, label='@B.+V.'),
#                     # Line2D([0], [0], color='k', label='B.+V.', **d_style['bias_plus_var']),
#                     ]

# plt.legend(handles=legend_elements,bbox_to_anchor=(1.05, 0.5), loc='center left',handlelength=1)


#%% Fix ax_CM

ax_CM.tick_params(which='both', left=True, labelleft=True, right=True, labelright=True) # need to be located here


#%% Text

fig.canvas.draw()
fig.set_constrained_layout(False)

# caption_A = plt.text(0.5, 0.5, r'$\displaystyle\frac{|\mu_{y}-\mu_{z}|}{\sigma_y}$', horizontalalignment='center', fontsize=fontsize_sup,
caption_A = plt.text(1, 0.5, r'$\displaystyle\frac{MAE^{test}_{\mu_{y} \leftarrow \mu_{z}}}{MAE^{test}_{\mu_{y} \leftarrow \mu_{y}}}$', horizontalalignment='right', fontsize=fontsize_sup,
                     verticalalignment='center', transform=transforms.blended_transform_factory(ax_A_y.transAxes, ax_A_x.transAxes))
caption_A.set_in_layout(False)
# caption_B = plt.text(0.5, 0.5, r'$\displaystyle\frac{|\mu_{z}-\bar{z}|}{\sigma_y}$', horizontalalignment='center', fontsize=fontsize_sup,
caption_B = plt.text(0, 0.5, r'$\displaystyle\frac{MAE^{test}_{\mu_{y} \leftarrow \bar{y}}}{MAE^{test}_{\mu_{y} \leftarrow \mu_{y}}}$', horizontalalignment='left', fontsize=fontsize_sup,
                     verticalalignment='center', transform=transforms.blended_transform_factory(ax_B_y.transAxes, ax_B_x.transAxes))
caption_B.set_in_layout(False)
# caption_C = plt.text(0.4, 0.5, r'$|\mu_{y}-\mu_{z}|/\sigma_y$'+'\n'+'$+$'+'\n'+r'$|\mu_{z}-\bar{z}|/\sigma_y$', horizontalalignment='center', fontsize=fontsize_sup,
caption_C = plt.text(1, 0.5, r'$\displaystyle\frac{MAE^{test}_{\mu_{y} \leftarrow \bar{z}}}{MAE^{test}_{\mu_{y} \leftarrow \mu_{y}}}$', horizontalalignment='right', fontsize=fontsize_sup,
                     verticalalignment='center', transform=transforms.blended_transform_factory(ax_C_y.transAxes, ax_C_x.transAxes))
caption_C.set_in_layout(False)


temp = 0.005

label_A = plt.text(temp, 1-temp, 'A', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=fig.transFigure, wrap=False)
label_A.set_in_layout(False)

label_B = plt.text(1-temp, 1-temp,
                    'B', horizontalalignment='right', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=fig.transFigure, wrap=False)
label_B.set_in_layout(False)

label_C = plt.text(temp, 0,
                    'C', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='bottom', transform=fig.transFigure, wrap=False)
label_C.set_in_layout(False)

label_D = plt.text(1-temp, 0,
                    'D', horizontalalignment='right', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='bottom', transform=fig.transFigure, wrap=False)
label_D.set_in_layout(False)


#%% Final rearrangement
temp1 = ax_A_x.get_xticks()
temp2 = ax_A.get_xlim()
ax_A.set_xticks(temp1[(temp1>=temp2[0])&(temp1<=temp2[1])])

temp1 = ax_B_x.get_xticks()
temp2 = ax_B.get_xlim()
ax_B.set_xticks(temp1[(temp1>=temp2[0])&(temp1<=temp2[1])])

temp1 = ax_C_x.get_xticks()
temp2 = ax_C.get_xlim()
ax_C.set_xticks(temp1[(temp1>=temp2[0])&(temp1<=temp2[1])])


temp1 = ax_CM.get_yticks(minor=False)
temp2 = ax_CM.get_yticks(minor=True)

temp3 = ax_A_x.get_ylim()
ax_A_x.set_yticks(temp1,minor=False)
ax_A_x.set_yticks(temp2,minor=True)
ax_A_x.set_ylim(temp3)

temp3 = ax_A_y.get_xlim()
ax_A_y.set_xticks(temp1,minor=False)
ax_A_y.set_xticks(temp2,minor=True)
ax_A_y.set_xlim(temp3)

temp3 = ax_B_x.get_ylim()
ax_B_x.set_yticks(temp1,minor=False)
ax_B_x.set_yticks(temp2,minor=True)
ax_B_x.set_ylim(temp3)

temp3 = ax_B_y.get_xlim()
ax_B_y.set_xticks(temp1,minor=False)
ax_B_y.set_xticks(temp2,minor=True)
ax_B_y.set_xlim(temp3)

temp3 = ax_C_x.get_ylim()
ax_C_x.set_yticks(temp1,minor=False)
ax_C_x.set_yticks(temp2,minor=True)
ax_C_x.set_ylim(temp3)

temp3 = ax_C_y.get_xlim()
ax_C_y.set_xticks(temp1,minor=False)
ax_C_y.set_xticks(temp2,minor=True)
ax_C_y.set_xlim(temp3)

temp3 = ax_D_00.get_ylim()
ax_D_00.set_yticks(temp1,minor=False)
ax_D_00.set_yticks(temp2,minor=True)
ax_D_00.set_ylim(temp3)

temp3 = ax_D_01.get_ylim()
ax_D_01.set_yticks(temp1,minor=False)
ax_D_01.set_yticks(temp2,minor=True)
ax_D_01.set_ylim(temp3)

temp3 = ax_D_10.get_ylim()
ax_D_10.set_yticks(temp1,minor=False)
ax_D_10.set_yticks(temp2,minor=True)
ax_D_10.set_ylim(temp3)

temp3 = ax_D_11.get_ylim()
ax_D_11.set_yticks(temp1,minor=False)
ax_D_11.set_yticks(temp2,minor=True)
ax_D_11.set_ylim(temp3)

#%% Save
plt.savefig('SF4'+".svg",dpi = 300, format="svg",transparent=True)
plt.savefig('SF4'+'.png',dpi = 400, format='png')