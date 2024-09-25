#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 06:33:02 2024

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
cmap = plt.cm.summer
# cmap = plt.cm.viridis
cmap = LinearSegmentedColormap.from_list("", [
                                              'peru','goldenrod','olive','olivedrab',
                                              'green',
                                              'limegreen','yellowgreen','palegoldenrod','orange'])

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

step_detials_sec = 26


#%% Load Data
data_2D = np.load(data_path+data_name,allow_pickle=True).flatten()[0]
for k in data_2D.keys():
    locals()[k] = data_2D[k]
out_mean = -1 * data_2D['out_mean']# correct \mu_y-\mu_z vs \mu_z-\mu_y
out_std_disc = data_2D['out_std_disc']
ZHs = data_2D['ZHs']
HSs = data_2D['HSs']

#%%
plt.close('all')

fig = plt.figure(layout='constrained', figsize=(8.9, 8))#latex --> (5.5,9)

gs = fig.add_gridspec(2,2, hspace=0.0, wspace=0.02)
gs.set_width_ratios([1,0.325/4])

ax_1 = fig.add_subplot(gs[0,:])
ax_2 = fig.add_subplot(gs[1,:])

# ax_CM = fig.add_subplot(gs[:,1])

#%% Colorbar
cmap_min = -np.nanmax(np.abs(np.concatenate([out_mean.flatten(),out_std_disc.flatten()]))) # exlude bias
cmap_max = -cmap_min

linthresh=1

temp1 = []
for i in np.arange(np.ceil(np.log10(cmap_max))):
    for ii in np.linspace(0,10,11)[1:]/1:
        temp1.append(ii*10**i)
temp1 = np.array(temp1)[temp1<cmap_max]
# temp1 = np.logspace(-2,np.log10(cmap_max),15)
temp2 = np.concatenate((np.array([1e-8]),np.arange(10)[1:]/10,temp1,np.array([cmap_max])))
tot_levels = np.unique(np.sort(np.concatenate((-temp2,temp2))))
temp = np.concatenate((np.linspace(0,linthresh,100),np.logspace(np.log10(linthresh),np.log10(cmap_max),100)))
tot_levels_f = np.unique(np.concatenate((-temp,temp)))

colornorm = SymLogNorm(linthresh=linthresh, linscale=1,
                       vmin=cmap_min,
                       vmax=cmap_max,
                       base=10)
colormap = ScalarMappable(cmap=cmap, norm=colornorm)

# cbar = fig.colorbar(
#             colormap,
#             cax=ax_CM, orientation='vertical',
#             extend='max',# extendfrac='auto',
#             alpha=0.1
# )


# # ax_CM.set_yscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
# ax_CM.set_yscale('log')

# ax_CM.set_ylim(1e-1,ax_CM.get_ylim()[1])

# ax_CM.set_ylim(ax_CM.get_ylim())
# ax_CM.set_xlim([0,1])

# 
pmeshcolor_span = np.concatenate((np.linspace(0,linthresh),np.logspace(*np.log10([linthresh,cmap_max]))))
pmeshcolor_span = np.unique(np.sort(np.concatenate((-pmeshcolor_span,pmeshcolor_span))))


#%% Ax 1

ax_1.plot(HSs[int(np.floor(out_std_disc.shape[0]/2)),:],out_std_disc[int(np.floor(out_std_disc.shape[0]/2)),:],'k',linewidth=2, zorder = np.inf, label = r'$z_{0}/h=0$')

ax_1.plot(HSs[int(np.floor(out_std_disc.shape[0]/2)),:],HSs[int(np.floor(out_std_disc.shape[0]/2)),:]/2,'tab:red',linewidth=2,linestyle='--', zorder = np.inf, label = r'$\displaystyle\frac{h/\sigma_{y}}{2}$')

t1,t2 = np.meshgrid(np.unique(HSs),pmeshcolor_span)
ax_1.pcolormesh(t1,t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,HSs.shape[0],step_detials_sec):
    if i == 0:
        ax_1.plot(HSs[i,:],out_std_disc[i,:],'k',linewidth=0.25, linestyle = ':', label = r'$z_{0}/h\neq 0$')
    else:
        ax_1.plot(HSs[i,:],out_std_disc[i,:],'k',linewidth=0.25, linestyle = ':')
ax_1.set_xscale('log')
ax_1.set_yscale('log')
# ax_1.set_xlim(0,cmap_max)
ax_1.set_ylim(0.1,ax_1.get_ylim()[1])
ax_1.set_xticklabels([])

ax_1.plot(HSs[int(np.floor(out_std_disc.shape[0]/2)),:],np.sqrt(1+HSs[int(np.floor(out_std_disc.shape[0]/2)),:]**2/12),'tab:blue',linewidth=2,linestyle='--', zorder = np.inf, label = r'$\displaystyle \sqrt{1+\frac{\left(h/\sigma_{y}\right)^2}{12}}$')

ax_1.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                                left=True, labelleft=True, right=False, labelright=False)

ax_1.plot([np.sqrt(6)]*2,ax_1.get_ylim(), color = color_h_crit, linewidth=2, linestyle = ':', zorder = 1)
ax_1.set_ylabel('$\sigma_{z}/\sigma_{y}$')

ax_1.legend(ncols=2)


# ax_1_secx = ax_1.secondary_yaxis(location='right')
ax_1_secx = ax_1.twiny()
ax_1_secx.set_xlim(ax_1.get_xlim())
# ax_1_secx.loglog(1,np.sqrt(6))
ax_1_secx.set_xscale('log')
# ax_1_secx.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
#                       left=False, labelleft=False, right=False, labelright=False,rotation = 0)
ax_1_secx.set_xticks([np.sqrt(6)])
ax_1_secx.set_xticklabels([r'$h_{norm}^{crit}=\sqrt{6}$'])
ax_1_secx.tick_params(axis='x',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_1_secx.set_alpha(d_style_contour['alpha'])

#%% Ax 2

ax_2.plot(HSs[int(np.floor(out_std_disc.shape[0]/2)),:],np.abs(out_std_disc[int(np.floor(out_std_disc.shape[0]/2)),:]-HSs[int(np.floor(out_std_disc.shape[0]/2)),:]/2),'tab:red',linewidth=2,linestyle='--', label = r'$\displaystyle\left|\frac{\sigma_{z}}{\sigma_{y}}-\frac{h/\sigma_{y}}{2}\right|$')

ax_2.plot(HSs[int(np.floor(out_std_disc.shape[0]/2)),:],np.abs(out_std_disc[int(np.floor(out_std_disc.shape[0]/2)),:]-np.sqrt(1+HSs[int(np.floor(out_std_disc.shape[0]/2)),:]**2/12)),'tab:blue',linewidth=2,linestyle='--', label = r'$\displaystyle\left|\frac{\sigma_{z}}{\sigma_{y}}-\sqrt{1+\frac{\left(h/\sigma_{y}\right)^2}{12}}\right|$')

ax_2.legend(ncols=2)

ax_2.set_xscale('log')
ax_2.set_yscale('log')

ax_2.set_xlim(ax_1.get_xlim())
ax_2.set_ylim(2*1e-5,ax_2.get_ylim()[1])
ax_2.set_xlabel('$h/\sigma_{y}$')
ax_2.set_ylabel('$Absolute\ Deviation$')

ax_2.tick_params(which='both',top=True, labeltop=False, bottom=True, labelbottom=True,
                  left=True, labelleft=True, right=False, labelright=False)

ax_2.plot([np.sqrt(6)]*2,ax_2.get_ylim(), color = color_h_crit, linewidth=2, linestyle = ':', zorder = 0)


#%% Text

# fig.canvas.draw()
# fig.set_constrained_layout(False)

temp = 0.005

label_A = plt.text(temp, 1-temp, 'A', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=fig.transFigure, wrap=False)
label_A.set_in_layout(False)

label_B = plt.text(temp, 0,
                    'B', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='bottom', transform=fig.transFigure, wrap=False)
label_B.set_in_layout(False)


#%% Save
plt.savefig('SF2'+".svg",dpi = 300, format="svg",transparent=True)
plt.savefig('SF2'+'.png',dpi = 400, format='png')