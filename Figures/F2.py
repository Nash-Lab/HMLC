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

gs = fig.add_gridspec(2,3, hspace=0.1, wspace=0.02)
gs.set_height_ratios([3.6*2/8,1])
gs.set_width_ratios([1,0.325,1])

subfig_A = fig.add_subfigure(gs[0, :])
subfig_B = fig.add_subfigure(gs[1, 0])
subfig_C = fig.add_subfigure(gs[1, 2])
# subfig_D = fig.add_subfigure(gs[1, 2])
subfig_CM = fig.add_subfigure(gs[1, 1])

# subfig_A.set_facecolor('0.15')
# subfig_B.set_facecolor('0.30')
# subfig_C.set_facecolor('0.45')
# # subfig_D.set_facecolor('0.6')
# subfig_CM.set_facecolor('0.75')


gs_A = subfig_A.add_gridspec(1,1)

gs_B = subfig_B.add_gridspec(2,2)
gs_B.set_height_ratios([gs_subfig_ABC_ratio.max(),gs_subfig_ABC_ratio.min(),])
gs_B.set_width_ratios([gs_subfig_ABC_ratio.min(),gs_subfig_ABC_ratio.max(),])

gs_C = subfig_C.add_gridspec(2,2)
gs_C.set_height_ratios([gs_subfig_ABC_ratio.max(),gs_subfig_ABC_ratio.min(),])
gs_C.set_width_ratios([gs_subfig_ABC_ratio.max(),gs_subfig_ABC_ratio.min(),])

gs_CM = subfig_CM.add_gridspec(1,1)


ax_A = subfig_A.add_subplot(gs_A[0,0])

ax_B = subfig_B.add_subplot(gs_B[0,1])
ax_B_x = subfig_B.add_subplot(gs_B[1, 1])
ax_B_y = subfig_B.add_subplot(gs_B[0,0])

ax_C = subfig_C.add_subplot(gs_C[0,0])
ax_C_x = subfig_C.add_subplot(gs_C[1, 0])
ax_C_y = subfig_C.add_subplot(gs_C[0,1])


ax_CM = subfig_CM.add_subplot(gs_CM[:, :])

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

cbar = subfig_CM.colorbar(
            colormap,
            cax=ax_CM, orientation='vertical',
            extend='both',# extendfrac='auto',
)

# remove cbar, could be improved
temp = cbar.ax.get_children()
print(temp)
temp[3].remove()
temp[2].remove()
temp[1].remove()

ax_CM.set_yscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)

ax_CM.set_ylim(ax_CM.get_ylim())
ax_CM.set_xlim([0,1])

# 
pmeshcolor_span = np.concatenate((np.linspace(0,linthresh),np.logspace(*np.log10([linthresh,cmap_max]))))
pmeshcolor_span = np.unique(np.sort(np.concatenate((-pmeshcolor_span,pmeshcolor_span))))


t1,t2 = np.meshgrid(np.linspace(0.,0.5),pmeshcolor_span)
ax_CM.contourf(t1,t2,t2,levels=tot_levels_f, alpha = alpha_main,zorder=np.inf,cmap = cmap, norm=colornorm)
ax_CM.contour(t1,t2,t2,levels=tot_levels, alpha = alpha_contour, linewidths=0.5,zorder=np.inf,cmap = cmap, norm=colornorm)
t1,t2 = np.meshgrid(np.linspace(0.5,1),pmeshcolor_span)
ax_CM.pcolormesh(t1,t2,t2, norm=colornorm,cmap=cmap,alpha=alpha_sec,transform=transforms.blended_transform_factory(ax_CM.transData, ax_CM.transData),zorder=np.inf)

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
from scipy.stats import norm,rv_discrete,rv_histogram
from scipy.special import erf

N_gates = 4
bins_tot_0 = np.linspace(-2,2,N_gates*2+1)
bins_edges_0 = bins_tot_0[::2]
bins_median_0 = bins_tot_0[1::2]

mu_0 = 0.35
sigma_0 = 0.375
dist_y = norm(mu_0,sigma_0)
y = np.linspace(-2,2,1001)
y_pdf = dist_y.pdf(y)
z = bins_median_0.copy()
z_pmf = np.diff(0.5*(1+erf((bins_edges_0-mu_0)/np.sqrt(2)/sigma_0)))
dist_z = rv_discrete(values=(bins_median_0,z_pmf))
print(z_pmf)

color_z = 'grey'

ax_A1 = ax_A.twinx()
ax_A2 = ax_A.twiny()
ax_A2.tick_params(axis='x', colors=color_z)
ax_A.spines['top'].set_color(color_z) 
ax_A.spines['right'].set_color(color_z)
ax_A1.tick_params(axis='y', colors=color_z)
ax_A.set_xlim([-2,2])
ax_A1.set_xlim([-2,2])
ax_A1.set_ylim([0,1])
ax_A2.set_xlim([-2,2])


ax_A1.bar(z,z_pmf,alpha=0.5,width = 1,color=color_z)
ax_A1.set_ylabel(r'$P\left(z\right)$',color=color_z)
# ax_A2.set_xticks(bins_median_0)
ax_A2.set_xticks(np.concatenate((np.array([ax_A2.get_xlim()[0]]),bins_median_0)))
ax_A2.set_xticklabels([r'$h_{0}$']+[f'${i}$' for i in bins_median_0])
ax_A2.set_xlabel(r'$z$',color=color_z)

ax_A.set_xlabel(r'$y$')
ax_A.set_ylabel(r'$\phi\left(y\right)=dP\left(y\right)/dy$')
ax_A.plot(y,y_pdf,color = 'k', zorder = 1)
ax_A.set_ylim([0, np.max(ax_A.get_ylim())])

for i in bins_edges_0:
    # ax_A.plot([i,i], [0, np.max(ax_A.get_ylim())], linestyle= (0, (10, 10)), color = 'k',linewidth=0.8)
    # ax_A.plot([i,i], [0, np.max(ax_A.get_ylim())], linestyle= (10, (10, 10)), color = color_z,linewidth=0.8)
    ax_A.plot([i,i], [0, np.max(ax_A.get_ylim())], color = color_z,linewidth=0.8)
    
ax_A.set_zorder(2)
ax_A1.set_zorder(1)
ax_A.patch.set_visible(False)


ax_A.annotate("",xy=(0, 0.5), xytext=(mu_0, 0.5),
                  arrowprops=dict(arrowstyle="->",shrinkA=0, shrinkB=0))
ax_A.plot([mu_0],[0.5],marker='|',color='k')
ax_A.annotate("$z_0$",xy=(mu_0/2, 0.51),ha='center',va='bottom')
ax_A.annotate("$\mu_{y}$",xy=(mu_0, 0.45),ha='center',va='top')
ax_A.plot([mu_0,mu_0],[0.5,dist_y.pdf(mu_0)],':',color='k')

ax_A.annotate("h",xy=(0.5, 0.11),ha='center',va='bottom')
ax_A.plot([0,1],[0.1,0.1],marker='|',color='k')

ax_A.annotate('h$=$1.0$;\ z_0=-$'+f'{dist_y.mean():.3f}'+'\n'+'$\mu_z$'+r'$\approx$'+f'{dist_z.mean():.3f}'+'$;\ \sigma_z$'+r'$\approx$'+f'{dist_z.std():.3f}'+'\n'+'$\mu_{y}=$'+f'{dist_y.mean():.3f}'+'$,\sigma_y=$'+f'{dist_y.std():.3f}',
                  xy=(0.02, 0.95), xycoords='axes fraction',
                  ha='left', va='top',
                  bbox=dict(boxstyle='round', fc='w',alpha=0.75))


#%% subfig B
s = 'out_mean'

temp = locals()[s].copy()
temp[np.abs(temp)<1e-4]=0
# ax_B.contourf(ZHs,HSs,temp,levels=tot_levels, alpha = alpha_main,zorder=np.inf,cmap = cmap, norm=colornorm)
ax_B.contourf(ZHs,HSs,temp,levels=tot_levels_f, alpha = 1,zorder=0,cmap = cmap, norm=colornorm)
rect_A = Rectangle((0, 0), width=1, height=1, transform=ax_B.transAxes, alpha = 1-alpha_main, zorder = 1, facecolor='white', edgecolor='black', linewidth = 0.8)
ax_B.add_patch(rect_A)
ax_B.contour(ZHs,HSs,temp,levels=tot_levels, alpha = alpha_contour, linewidths=alpha_main, zorder=2, cmap = cmap, norm=colornorm)

ax_B.set_yscale('log')
ax_B.set_xticklabels([])
ax_B.set_yticklabels([])
ax_B.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True, rotation = 90)


ax_B_secy = ax_B.secondary_yaxis(location='left')
ax_B_secy.set_yticks([np.sqrt(6)])
ax_B_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_B_secy.set_alpha(d_style_contour['alpha'])
ax_B_secy.yaxis.set_ticklabels([])


t1,t2 = np.meshgrid(np.unique(ZHs),pmeshcolor_span)
ax_B_x.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,ZHs.shape[0],step_detials_sec):
    ax_B_x.plot(ZHs[:,i],locals()[s][:,i],'k',linewidth=0.25)
ax_B_x.set_yscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
ax_B_x.set_ylim(cmap_min,cmap_max)
ax_B_x.set_yticklabels([])
ax_B_x.set_xlabel('$z_{0}/h$')
ax_B_x.xaxis.set_label_position("bottom")
ax_B_x.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True)


t1,t2 = np.meshgrid(pmeshcolor_span,np.unique(HSs))
ax_B_y.pcolormesh(t1,t2, t1, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,ZHs.shape[0],step_detials_sec):
    ax_B_y.plot(locals()[s][i,:],HSs[i,:],'k',linewidth=0.25)
ax_B_y.set_xscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
ax_B_y.set_yscale('log')
ax_B_y.set_xlim(cmap_min,cmap_max)
ax_B_y.set_xticklabels([])
ax_B_y.set_ylabel('$h/\sigma_{y}$')
ax_B_y.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True)
ax_B_y.invert_xaxis()

ax_B_y_secy = ax_B_y.secondary_yaxis(location='left')
ax_B_y_secy.set_yticks([np.sqrt(6)])
ax_B_y_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_B_y_secy.set_alpha(d_style_contour['alpha'])
ax_B_y_secy.yaxis.set_ticklabels([])

#%% subfig C
s = 'out_std_disc'

temp = locals()[s].copy()
temp[np.abs(temp)<1e-4]=0
# ax_C.contourf(ZHs,HSs,temp,levels=tot_levels, alpha = alpha_main,zorder=np.inf,cmap = cmap, norm=colornorm)
ax_C.contourf(ZHs,HSs,temp,levels=tot_levels_f, alpha = 1,zorder=0,cmap = cmap, norm=colornorm)
rect_B = Rectangle((0, 0), width=1, height=1, transform=ax_C.transAxes, alpha = 1-alpha_main, zorder = 1, facecolor='white', edgecolor='black', linewidth = 0.8)
ax_C.add_patch(rect_B)
ax_C.contour(ZHs,HSs,temp,levels=tot_levels, alpha = alpha_contour, linewidths=alpha_main, zorder=2, cmap = cmap, norm=colornorm)

ax_C.set_yscale('log')
ax_C.set_xticklabels([])
ax_C.set_yticklabels([])
ax_C.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                              left=False, labelleft=False, right=True, labelright=True,rotation = 90)

ax_C_secy = ax_C.secondary_yaxis(location='right')
ax_C_secy.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                      left=False, labelleft=False, right=True, labelright=True,rotation = 90)
ax_C_secy.set_yticks([np.sqrt(6)])
ax_C_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_C_secy.set_alpha(d_style_contour['alpha'])
ax_C_secy.yaxis.set_ticklabels([])


t1,t2 = np.meshgrid(np.unique(ZHs),pmeshcolor_span)
ax_C_x.pcolormesh(t1, t2, t2, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,ZHs.shape[0],step_detials_sec):
    ax_C_x.plot(ZHs[:,i],locals()[s][:,i],'k',linewidth=0.25)
# ax_C_x.set_xscale('log')
ax_C_x.set_yscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
ax_C_x.set_ylim(0,cmap_max)
ax_C_x.set_yticklabels([])
ax_C_x.set_xlabel('$z_{0}/h$')
ax_C_x.xaxis.set_label_position("bottom")
ax_C_x.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                                left=False, labelleft=False, right=True, labelright=True)


t1,t2 = np.meshgrid(pmeshcolor_span,np.unique(HSs))
ax_C_y.pcolormesh(t1,t2, t1, norm=colornorm,cmap=cmap,alpha=alpha_sec)
for i in range(0,HSs.shape[0],step_detials_sec):
    ax_C_y.plot(locals()[s][i,:],HSs[i,:],'k',linewidth=0.25)
ax_C_y.set_xscale('symlog',base=10,linthresh=1,subs=[2, 3, 4, 5, 6, 7, 8, 9],linscale=1)
ax_C_y.set_yscale('log')
ax_C_y.set_xlim(0,cmap_max)
ax_C_y.set_xticklabels([])
ax_C_y.set_ylabel('$h/\sigma_{y}$')
ax_C_y.yaxis.set_label_position("right")
ax_C_y.tick_params(which='both',top=False, labeltop=False, bottom=True, labelbottom=True,
                                left=False, labelleft=False, right=True, labelright=True)
ax_C_y.invert_xaxis()

ax_C_y_secy = ax_C_y.secondary_yaxis(location='right')
ax_C_y_secy.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                        left=False, labelleft=False, right=True, labelright=True,rotation = 90)
ax_C_y_secy.set_yticks([np.sqrt(6)])
ax_C_y_secy.tick_params(axis='y',direction='out',color=color_h_crit, width=d_style_contour['linewidth'], length = crit_ticks_length)
ax_C_y_secy.set_alpha(d_style_contour['alpha'])
ax_C_y_secy.yaxis.set_ticklabels([])


ax_CM.tick_params(which='both', left=True, labelleft=True, right=True, labelright=True)

#%% Final rearrangement
temp1 = ax_CM.get_yticks(minor=False)
temp2 = ax_CM.get_yticks(minor=True)

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


#%% Text

fig.canvas.draw()
fig.set_constrained_layout(False)

caption_B = plt.text(0.5, 0.5, r'$\displaystyle\frac{\mu_{y}-\mu_{z}}{\sigma_{y}}$', horizontalalignment='center', fontsize=fontsize_sup,
                      verticalalignment='center', transform=transforms.blended_transform_factory(ax_B_y.transAxes, ax_B_x.transAxes))
caption_B.set_in_layout(False)
caption_C = plt.text(0.5, 0.5, r'$\displaystyle\frac{\sigma_{z}}{\sigma_{y}}$', horizontalalignment='center', fontsize=fontsize_sup,
                      verticalalignment='center', transform=transforms.blended_transform_factory(ax_C_y.transAxes, ax_C_x.transAxes))
caption_C.set_in_layout(False)


temp = 0.005

label_A = plt.text(temp, 1-temp, 'A', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='top', transform=fig.transFigure, wrap=False)
label_A.set_in_layout(False)

label_B = plt.figtext(temp, 0.,
                    'B', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='bottom', transform=fig.transFigure, wrap=False, zorder = np.inf)
label_B.set_in_layout(False)

label_C = plt.text(1-temp, 0,
                    'C', horizontalalignment='right', fontsize=fontsize_letter,fontweight='bold',
                    verticalalignment='bottom', transform=fig.transFigure, wrap=False)
label_C.set_in_layout(False)


#%% Save
plt.savefig('F2'+".svg",dpi = 300, format="svg",transparent=True)
plt.savefig('F2'+'.png',dpi = 400, format='png')