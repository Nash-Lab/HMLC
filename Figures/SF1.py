#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:46:35 2024

@author: vdoffini
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import plasma,winter,bwr,cividis,twilight_shifted,ScalarMappable,get_cmap
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
data_path = '../Results/'
data_name = '01_2D_hist.npy'

# cmap = plt.cm.plasma
# cmap = plt.cm.RdYlGn_r
# cmap = plt.cm.summer
# cmap = plt.cm.viridis
cmap_A = LinearSegmentedColormap.from_list("", [
                                                #'peru','goldenrod','olive','olivedrab',
                                                'green',
                                                'limegreen','yellowgreen','palegoldenrod','orange'])

cmap_B = LinearSegmentedColormap.from_list("", [
                                                #'peru','goldenrod','olive','olivedrab',
                                                'green',
                                                'limegreen','yellowgreen','palegoldenrod','orange'][::-1])

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

color_H_0_5 = 'mediumpurple'

d_style = {'bias':{'linestyle':'--','dashes':[3, 3]},
           'var':{'linestyle':'--','dashes':[1, 1, 1, 1]},
           'bias_plus_var':{'linestyle':'-'}}
d_style_contour = {'linewidth':2,'alpha':1}

crit_ticks_length = 5

alpha_main = 0.5
alpha_sec = 0.15
alpha_contour = 0.5

step_detials_sec = 25

#%% Load Data
data_2D = np.load(data_path+data_name,allow_pickle=True).flatten()[0]
for k in data_2D.keys():
    locals()[k] = data_2D[k]
out_kld = data_2D['out_kld']# correct \mu_y-\mu_z vs \mu_z-\mu_y
out_h_disc = data_2D['out_h_disc']
ZHs = data_2D['ZHs']
HSs = data_2D['HSs']

#%%
plt.close('all')

fig = plt.figure(layout='constrained', figsize=(8.9, 4*(1+0.325/2)))#latex --> (5.5,9)

gs = fig.add_gridspec(1,2, hspace=0., wspace=0.02)
gs.set_height_ratios([1])
gs.set_width_ratios([1,1])

subfig_A = fig.add_subfigure(gs[0, 0])
subfig_B = fig.add_subfigure(gs[0, -1])

gs_A = subfig_A.add_gridspec(3,2)
gs_A.set_height_ratios([gs_subfig_ABC_ratio.min(),gs_subfig_ABC_ratio.max(),gs_subfig_ABC_ratio.sum()*0.325/2])
gs_A.set_width_ratios([gs_subfig_ABC_ratio.min(),gs_subfig_ABC_ratio.max()])

gs_B = subfig_B.add_gridspec(3,2)
gs_B.set_height_ratios([gs_subfig_ABC_ratio.min(),gs_subfig_ABC_ratio.max(),gs_subfig_ABC_ratio.sum()*0.325/2])
gs_B.set_width_ratios([gs_subfig_ABC_ratio.max(),gs_subfig_ABC_ratio.min()])


ax_A = subfig_A.add_subplot(gs_A[1,1])
ax_A_x = subfig_A.add_subplot(gs_A[0, 1])
ax_A_y = subfig_A.add_subplot(gs_A[1,0])

ax_B = subfig_B.add_subplot(gs_B[1,0])
ax_B_x = subfig_B.add_subplot(gs_B[0, 0])
ax_B_y = subfig_B.add_subplot(gs_B[1,1])

ax_CM_A = subfig_A.add_subplot(gs_A[-1, 1])
ax_CM_B = subfig_B.add_subplot(gs_B[-1, 0])


#%% Colorbar (A)
cmap_A_min = 0 # exlude bias
cmap_A_max = out_kld.max()

linthresh=1

temp1 = []
for i in np.arange(np.ceil(np.log10(cmap_A_max))):
    for ii in np.linspace(0,10,11)[1:]/1:
        temp1.append(ii*10**i)
temp1 = np.array(temp1)[temp1<cmap_A_max]
# temp1 = np.logspace(-2,np.log10(cmap_A_max),15)
tot_levels_A = np.unique(np.sort(np.concatenate((np.array([1e-2]),np.arange(np.ceil(cmap_A_max*10+1e-8))[1:]/10))))
temp = np.concatenate((np.linspace(0,linthresh,100),np.logspace(np.log10(linthresh),np.log10(cmap_A_max),100)))
tot_levels_f = np.unique(np.concatenate((-temp,temp)))

colornorm = plt.Normalize(#linthresh=linthresh, linscale=1,
                       vmin=cmap_A_min,
                       vmax=cmap_A_max,
                       )
colormap = ScalarMappable(cmap=cmap_A, norm=colornorm)

cbar = subfig_A.colorbar(
            colormap,
            cax=ax_CM_A, orientation='horizontal',
            extend='both',# extendfrac='auto',
)

# remove cbar, could be improved
temp = cbar.ax.get_children()
print(temp)
temp[3].remove()
temp[2].remove()
temp[1].remove()

# ax_CM_A.set_yscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)

ax_CM_A.set_ylim([0,1])
ax_CM_A.set_xlim(ax_CM_A.get_xlim())

# # 
pmeshcolor_span = np.linspace(0,cmap_A_max)
# pmeshcolor_span = np.unique(np.sort(np.concatenate((-pmeshcolor_span,pmeshcolor_span))))


t1,t2 = np.meshgrid(pmeshcolor_span,np.linspace(0.,0.5))
ax_CM_A.contourf(t1,t2,t1,levels=tot_levels_f, alpha = alpha_main,zorder=np.inf,cmap = cmap_A, norm=colornorm)
ax_CM_A.contour(t1,t2,t1,levels=tot_levels_A, alpha = alpha_contour, linewidths=0.5,zorder=np.inf,cmap = cmap_A, norm=colornorm)
t1,t2 = np.meshgrid(pmeshcolor_span,np.linspace(0.5,1))
ax_CM_A.pcolormesh(t1,t2,t1, norm=colornorm,cmap=cmap_A,alpha=alpha_sec,transform=transforms.blended_transform_factory(ax_CM_A.transData, ax_CM_A.transData),zorder=np.inf)

patches = []
patches.append(PathPatch(Path([[1,1],[1,0.5],[1.05, 0.5]]), transform=ax_CM_A.transAxes, edgecolor = 'none', alpha = alpha_sec, facecolor=cmap_A(1.),zorder=0))
patches.append(PathPatch(Path([[0.,1],[0,0.5],[-0.05,0.5]]), transform=ax_CM_A.transAxes, edgecolor = 'none', alpha = alpha_sec, facecolor=cmap_A(0.),zorder=0))
patches.append(PathPatch(Path([[1,0],[1,0.5],[1.05, 0.5]]), transform=ax_CM_A.transAxes, edgecolor = 'none', alpha = alpha_main, facecolor=cmap_A(1.),zorder=0))
patches.append(PathPatch(Path([[0.,0],[0,0.5],[-0.05,0.5]]), transform=ax_CM_A.transAxes, edgecolor = 'none', alpha = alpha_main, facecolor=cmap_A(0.),zorder=0))
for p in patches: subfig_A.patches.append(p)

sep_CM_line = Line2D([-0.05,1.05],[0.5,0.5],color='k',linewidth=0.8,
                      transform=transforms.blended_transform_factory(ax_CM_A.transAxes, ax_CM_A.transAxes),zorder=np.inf)
subfig_A.lines.append(sep_CM_line)

# if tot_levels_A[0] == np.round(tot_levels_A[0]):
#     ax_CM_A.set_yticks(tot_levels_A,minor=True)
# else:
#     ax_CM_A.set_yticks(tot_levels_A[1:-1],minor=True)


#%% Colorbar (B)
cmap_B_min = 0 # exlude bias
cmap_B_max = out_h_disc.max()

linthresh=1

temp1 = []
for i in np.arange(np.ceil(np.log10(cmap_B_max))):
    for ii in np.linspace(0,10,11)[1:]/1:
        temp1.append(ii*10**i)
temp1 = np.array(temp1)[temp1<cmap_B_max]
# temp1 = np.logspace(-2,np.log10(cmap_B_max),15)
tot_levels_B = np.unique(np.sort(np.concatenate((np.array([1e-8]),np.arange(np.ceil(cmap_B_max*10+1e-8))[1:]/10))))
temp = np.concatenate((np.linspace(0,linthresh,100),np.logspace(np.log10(linthresh),np.log10(cmap_B_max),100)))
tot_levels_f = np.unique(np.concatenate((-temp,temp)))

colornorm = plt.Normalize(#linthresh=linthresh, linscale=1,
                       vmin=cmap_B_min,
                       vmax=cmap_B_max,
                       )
colormap = ScalarMappable(cmap=cmap_B, norm=colornorm)

cbar = subfig_B.colorbar(
            colormap,
            cax=ax_CM_B, orientation='horizontal',
            extend='both',# extendfrac='auto',
)

# remove cbar, could be improved
temp = cbar.ax.get_children()
print(temp)
temp[3].remove()
temp[2].remove()
temp[1].remove()

# ax_CM_B.set_yscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)

ax_CM_B.set_ylim([0,1])
ax_CM_B.set_xlim(ax_CM_B.get_xlim())

# # 
pmeshcolor_span = np.linspace(0,cmap_B_max)
# pmeshcolor_span = np.unique(np.sort(np.concatenate((-pmeshcolor_span,pmeshcolor_span))))


t1,t2 = np.meshgrid(pmeshcolor_span,np.linspace(0.,0.5))
ax_CM_B.contourf(t1,t2,t1,levels=tot_levels_f, alpha = alpha_main,zorder=np.inf,cmap = cmap_B, norm=colornorm)
ax_CM_B.contour(t1,t2,t1,levels=tot_levels_B, alpha = alpha_contour, linewidths=0.5,zorder=np.inf,cmap = cmap_B, norm=colornorm)
t1,t2 = np.meshgrid(pmeshcolor_span,np.linspace(0.5,1))
ax_CM_B.pcolormesh(t1,t2,t1, norm=colornorm,cmap=cmap_B,alpha=alpha_sec,transform=transforms.blended_transform_factory(ax_CM_B.transData, ax_CM_B.transData),zorder=np.inf)

patches = []
patches.append(PathPatch(Path([[1,1],[1,0.5],[1.05, 0.5]]), transform=ax_CM_B.transAxes, edgecolor = 'none', alpha = alpha_sec, facecolor=cmap_B(1.),zorder=0))
patches.append(PathPatch(Path([[0.,1],[0,0.5],[-0.05,0.5]]), transform=ax_CM_B.transAxes, edgecolor = 'none', alpha = alpha_sec, facecolor=cmap_B(0.),zorder=0))
patches.append(PathPatch(Path([[1,0],[1,0.5],[1.05, 0.5]]), transform=ax_CM_B.transAxes, edgecolor = 'none', alpha = alpha_main, facecolor=cmap_B(1.),zorder=0))
patches.append(PathPatch(Path([[0.,0],[0,0.5],[-0.05,0.5]]), transform=ax_CM_B.transAxes, edgecolor = 'none', alpha = alpha_main, facecolor=cmap_B(0.),zorder=0))
for p in patches: subfig_B.patches.append(p)

H_0_5_CM_line = Line2D([-np.log(0.5),-np.log(0.5)],[0.,0.5],color=color_H_0_5,linewidth=1,
                       transform=transforms.blended_transform_factory(ax_CM_B.transData, ax_CM_B.transAxes),zorder=np.inf)
subfig_B.lines.append(H_0_5_CM_line)

sep_CM_line = Line2D([-0.05,1.05],[0.5,0.5],color='k',linewidth=0.8,
                      transform=transforms.blended_transform_factory(ax_CM_B.transAxes, ax_CM_B.transAxes),zorder=np.inf)
subfig_B.lines.append(sep_CM_line)

# if tot_levels_B[0] == np.round(tot_levels_B[0]):
#     ax_CM_B.set_yticks(tot_levels_B,minor=True)
# else:
#     ax_CM_B.set_yticks(tot_levels_B[1:-1],minor=True)

ax_CM_B_secx1 = ax_CM_B.secondary_xaxis(location='top')
ax_CM_B_secx1.set_xticks([-np.log(0.5)])
ax_CM_B_secx1.tick_params(axis='x',direction='out',color=color_H_0_5, width=1, length = crit_ticks_length)
ax_CM_B_secx1.xaxis.set_ticklabels([])

ax_CM_B_secx2 = ax_CM_B.secondary_xaxis(location='bottom')
ax_CM_B_secx2.set_xticks([-np.log(0.5)])
ax_CM_B_secx2.tick_params(axis='x',direction='out',color=color_H_0_5, width=1, length = crit_ticks_length)
ax_CM_B_secx2.xaxis.set_ticklabels([])

ax_CM_B_secx3 = ax_CM_B.secondary_xaxis(location='top')
ax_CM_B_secx3.set_xticks([0,1,2,3])
plt.setp(ax_CM_B_secx3.get_xticklabels(), alpha=0.)


#%% subfig A
s = 'out_kld'

temp = locals()[s].copy()
# temp[np.abs(temp)<1e-4]=0
# ax_A.contourf(ZHs,HSs,temp,levels=tot_levels_A, alpha = alpha_main,zorder=np.inf,cmap = cmap, norm=colornorm)
ax_A.contourf(ZHs,HSs,temp,levels=tot_levels_f, alpha = 1,zorder=0,cmap = cmap_A, norm=colornorm)
rect_A = Rectangle((0, 0), width=1, height=1, transform=ax_A.transAxes, alpha = 1-alpha_main, zorder = 1, facecolor='white', edgecolor='black', linewidth = 0.8)
ax_A.add_patch(rect_A)
ax_A.contour(ZHs,HSs,temp,levels=tot_levels_A, alpha = alpha_contour, linewidths=alpha_main, zorder=2, cmap = cmap_A, norm=colornorm)

ax_A.set_yscale('log')
ax_A.set_xticklabels([])
ax_A.set_yticklabels([])
ax_A.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False, rotation = 90)


ax_A_secy = ax_A.secondary_yaxis(location='left')
ax_A_secy.set_yticks([np.sqrt(6)])
ax_A_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_A_secy.set_alpha(d_style_contour['alpha'])
ax_A_secy.yaxis.set_ticklabels([])


t1,t2 = np.meshgrid(np.unique(ZHs),pmeshcolor_span)
ax_A_x.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap_A,alpha=alpha_sec)
for i in range(0,ZHs.shape[0],step_detials_sec):
    ax_A_x.plot(ZHs[:,i],locals()[s][:,i],'k',linewidth=0.25)
# ax_A_x.set_yscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
ax_A_x.set_ylim(cmap_A_min,cmap_A_max)
ax_A_x.set_yticklabels([])
ax_A_x.set_xlabel('$z_{0}/h$')
ax_A_x.xaxis.set_label_position("top")
ax_A_x.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False)


t1,t2 = np.meshgrid(pmeshcolor_span,np.unique(HSs))
ax_A_y.pcolormesh(t1,t2, t1, norm=colornorm,cmap=cmap_A,alpha=alpha_sec)
for i in range(0,ZHs.shape[0],step_detials_sec):
    ax_A_y.plot(locals()[s][i,:],HSs[i,:],'k',linewidth=0.25)
# ax_A_y.set_xscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
ax_A_y.set_yscale('log')
ax_A_y.set_xlim(cmap_A_min,cmap_A_max)
ax_A_y.set_xticklabels([])
ax_A_y.set_ylabel('$h/\sigma_{y}$')
ax_A_y.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False)
ax_A_y.invert_xaxis()

ax_A_y_secy = ax_A_y.secondary_yaxis(location='left')
ax_A_y_secy.set_yticks([np.sqrt(6)])
ax_A_y_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_A_y_secy.set_alpha(d_style_contour['alpha'])
ax_A_y_secy.yaxis.set_ticklabels([])

#%% subfig B
s = 'out_h_disc'

temp = locals()[s].copy()
# temp[np.abs(temp)<1e-4]=0
# ax_B.contourf(ZHs,HSs,temp,levels=tot_levels_B, alpha = alpha_main,zorder=np.inf,cmap = cmap, norm=colornorm)
ax_B.contourf(ZHs,HSs,temp,levels=tot_levels_f, alpha = 1,zorder=0,cmap = cmap_B, norm=colornorm)
rect_B = Rectangle((0, 0), width=1, height=1, transform=ax_B.transAxes, alpha = 1-alpha_main, zorder = 1, facecolor='white', edgecolor='black', linewidth = 0.8)
ax_B.add_patch(rect_B)
ax_B.contour(ZHs,HSs,temp,levels=tot_levels_B, alpha = alpha_contour, linewidths=alpha_main, zorder=2, cmap = cmap_B, norm=colornorm)
ax_B.contour(ZHs,HSs,temp,levels=-np.log(0.5)+np.array([-1e-12,1e-12]), alpha = 1, linewidths=1,linestyles = '-',zorder=np.inf,cmap=LinearSegmentedColormap.from_list("", [color_H_0_5,color_H_0_5]))#, norm=Normalize(0,-2*np.log(0.5)))

ax_B.set_yscale('log')
ax_B.set_xticklabels([])
ax_B.set_yticklabels([])
ax_B.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                              left=False, labelleft=False, right=True, labelright=True,rotation = 90)

ax_B_secy = ax_B.secondary_yaxis(location='right')
ax_B_secy.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                      left=False, labelleft=False, right=True, labelright=True,rotation = 90)
ax_B_secy.set_yticks([np.sqrt(6)])
ax_B_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_B_secy.set_alpha(d_style_contour['alpha'])
ax_B_secy.yaxis.set_ticklabels([])


t1,t2 = np.meshgrid(np.unique(ZHs),pmeshcolor_span)
ax_B_x.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap_B,alpha=alpha_sec)
for i in range(0,ZHs.shape[0],step_detials_sec):
    ax_B_x.plot(ZHs[:,i],locals()[s][:,i],'k',linewidth=0.25)
# ax_B_x.set_xscale('log')
# ax_B_x.set_yscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
ax_B_x.set_ylim(0,cmap_B_max)
ax_B_x.set_yticklabels([])
ax_B_x.set_xlabel('$z_{0}/h$')
ax_B_x.xaxis.set_label_position("top")
ax_B_x.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                                left=False, labelleft=False, right=True, labelright=True)

# ax_B_x_secy1 = ax_B_x.secondary_yaxis(location='left')
# ax_B_x_secy1.set_yticks([-np.log(0.5)])
# ax_B_x_secy1.tick_params(axis='y',direction='out',color=color_H_0_5, width=1, length = crit_ticks_length)
# ax_B_x_secy1.yaxis.set_ticklabels([])

ax_B_x_secy2 = ax_B_x.secondary_yaxis(location='right')
ax_B_x_secy2.set_yticks([-np.log(0.5)])
ax_B_x_secy2.tick_params(axis='y',direction='out',color=color_H_0_5, width=1, length = crit_ticks_length)
ax_B_x_secy2.yaxis.set_ticklabels([])


t1,t2 = np.meshgrid(pmeshcolor_span,np.unique(HSs))
ax_B_y.pcolormesh(t1,t2, t1, norm=colornorm,cmap=cmap_B,alpha=alpha_sec)
for i in range(0,HSs.shape[0],step_detials_sec):
    ax_B_y.plot(locals()[s][i,:],HSs[i,:],'k',linewidth=0.25)
# ax_B_y.set_xscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
ax_B_y.set_yscale('log')
ax_B_y.set_xlim(0,cmap_B_max)
ax_B_y.set_xticklabels([])
ax_B_y.set_ylabel('$h/\sigma_{y}$')
ax_B_y.yaxis.set_label_position("right")
ax_B_y.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                                left=False, labelleft=False, right=True, labelright=True)
ax_B_y.invert_xaxis()

ax_B_y_secy = ax_B_y.secondary_yaxis(location='right')
ax_B_y_secy.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                        left=False, labelleft=False, right=True, labelright=True,rotation = 90)
ax_B_y_secy.set_yticks([np.sqrt(6)])
ax_B_y_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_B_y_secy.set_alpha(d_style_contour['alpha'])
ax_B_y_secy.yaxis.set_ticklabels([])

ax_B_y_secx1 = ax_B_y.secondary_xaxis(location='top')
ax_B_y_secx1.set_xticks([-np.log(0.5)])
ax_B_y_secx1.tick_params(axis='x',direction='out',color=color_H_0_5, width=1, length = crit_ticks_length)
ax_B_y_secx1.xaxis.set_ticklabels([])

# ax_B_y_secx2 = ax_B_y.secondary_xaxis(location='bottom')
# ax_B_y_secx2.set_xticks([-np.log(0.5)])
# ax_B_y_secx2.tick_params(axis='x',direction='out',color=color_H_0_5, width=1, length = crit_ticks_length)
# ax_B_y_secx2.xaxis.set_ticklabels([])


ax_CM_A.tick_params(which='both',top=True, labeltop=True, bottom=True, labelbottom=True,
                    left=False, labelleft=False, right=False, labelright=False)
ax_CM_B.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                    left=False, labelleft=False, right=False, labelright=False)


#%% Final rearrangement
temp1 = ax_CM_A.get_xticks(minor=False)
temp2 = ax_CM_A.get_xticks(minor=True)

temp3 = ax_A_x.get_ylim()
ax_A_x.set_yticks(temp1,minor=False)
ax_A_x.set_yticks(temp2,minor=True)
ax_A_x.set_ylim(temp3)

temp3 = ax_A_y.get_xlim()
ax_A_y.set_xticks(temp1,minor=False)
ax_A_y.set_xticks(temp2,minor=True)
ax_A_y.set_xlim(temp3)


temp1 = ax_CM_B.get_xticks(minor=False)
temp2 = ax_CM_B.get_xticks(minor=True)

temp3 = ax_B_x.get_ylim()
ax_B_x.set_yticks(temp1,minor=False)
ax_B_x.set_yticks(temp2,minor=True)
ax_B_x.set_ylim(temp3)

temp3 = ax_B_y.get_xlim()
ax_B_y.set_xticks(temp1,minor=False)
ax_B_y.set_xticks(temp2,minor=True)
ax_B_y.set_xlim(temp3)


#%% Text

fig.canvas.draw()
fig.set_constrained_layout(False)

caption_A = plt.text(0.5, 0.5, r'$D_{KL}(P_{y}||P_{z})$', horizontalalignment='center', fontsize=fontsize_sup,
                     verticalalignment='center', transform=transforms.blended_transform_factory(ax_A_y.transAxes, ax_A_x.transAxes))
caption_A.set_in_layout(False)
caption_B = plt.text(0.5, 0.5, r'$\displaystyle H\left(P_{z}\right)$', horizontalalignment='center', fontsize=fontsize_sup,
                     verticalalignment='center', transform=transforms.blended_transform_factory(ax_B_y.transAxes, ax_B_x.transAxes))
 # = -\sum_{i}\left(P_{z}\right)\cdot log\left(P_{z}\right)
caption_B.set_in_layout(False)

tick_H_0_5 = plt.text(-np.log(0.5), 1.2, r'$2H\left(0.5\right)$', horizontalalignment='center', fontsize=fontsize,
                     verticalalignment='bottom', transform=transforms.blended_transform_factory(ax_CM_B.transData, ax_CM_B.transAxes))
caption_A.set_in_layout(False)


temp = 0.005

label_A = plt.text(temp, 1-temp, 'A', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=fig.transFigure, wrap=False)
label_A.set_in_layout(False)

label_B = plt.text(1-temp, 1-temp,
                    'B', horizontalalignment='right', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=fig.transFigure, wrap=False)
label_B.set_in_layout(False)


#%% Save
plt.savefig('SF1'+".svg",dpi = 300, format="svg",transparent=True)
plt.savefig('SF1'+'.png',dpi = 400, format='png')