#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:46:35 2024

@author: vdoffini
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.cm import plasma,winter,bwr,cividis,twilight_shifted,ScalarMappable,get_cmap
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize,LinearSegmentedColormap,ListedColormap,LogNorm,SymLogNorm
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch,Rectangle,Arc,ConnectionPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
import matplotlib.transforms as transforms
import matplotlib
import pandas as pd
from scipy.stats import multivariate_normal,norm,gaussian_kde


cm = ListedColormap(twilight_shifted(np.linspace(0,31/64, 1024)))
plt.rcParams["font.family"] = "Times New Roman"#https://stackoverflow.com/questions/40734672/how-to-set-the-label-fonts-as-time-new-roman-by-drawparallels-in-python/40734893
fontsize = 13
fontsize_sup = 15
fontsize_letter = 30
plt.rcParams.update({'text.usetex': True})
plt.rcParams['text.latex.preamble'] =r"\usepackage{color} "

plt.rcParams.update({'font.size': fontsize})
plt.rcParams.update({'legend.fontsize': fontsize})
plt.rcParams.update({'axes.labelsize': fontsize})


#%% Parameters

results_path = '../Results/'
summary_results_name = '00_summary_data.npz'

data_path = '../Data/'
mut_data_name = 'mut.csv'

# cmap = plt.cm.plasma
# cmap = plt.cm.RdYlGn_r
# cmap = plt.cm.summer
cmap = plt.cm.viridis

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
color_HTS = 'coral'
cmap_HTS = LinearSegmentedColormap.from_list("", ['black',color_HTS,'white'])
cmap_HTS = LinearSegmentedColormap.from_list("", [cmap_HTS(0.25),color_HTS,cmap_HTS(0.75)])

cmap_LC = cmap_HTS


#%% Functions

# https://stackoverflow.com/questions/49223702/adding-a-legend-to-a-matplotlib-plot-with-a-multicolored-line
class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize,trans):
        x = np.linspace(0,width,self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap,
                     transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]
    
    
#%% Load Data

N_points = 51

def f_temp(x,exp,points = [0,1]):
    points_max = np.max(points)
    points_min = np.min(points)
    if exp>0:
        out = (1-x)**exp
    else:
        out = 1-x**-exp
    
    return out*(points_max-points_min)+points_min

# exps = [10,-10,-10]
# points = np.logspace(*np.log10([0.03,3]),len(exps)+1)[::-1]
# i_s = np.log10(np.logspace(0.01,0.1,51))
min_max = np.sort([0.01,0.25])
i_s = min_max[0]+np.logspace(*np.log10([min_max[0]/100,min_max[1]-min_max[0]]),N_points)[::-1]
out = []
for i in i_s:
    
    exps = [-5,4.5,4.5]

    points = np.linspace(*np.log10([i,1]),len(exps)+1)[::-1]
    # print(points)    
    
    x = np.linspace(0,len(exps),1000)
    
    out_temp = []
    for ii in range(len(exps)):
        if ii == 0:
            temp = x[(x>=0)&(x<=ii+1)]
        else:
            temp = x[(x>ii)&(x<=ii+1)]
        
        temp = (temp-temp.min())/(temp.max()-temp.min())
        
        out_temp.append(f_temp(temp,exps[ii],points[ii:ii+2]))
    
    out_temp = np.concatenate(out_temp)
    out.append(out_temp)
    
    # y_true = 
    
    plt.semilogy(x,10**out_temp)
out = 10**np.array(out)

Ns_tr_norm = x
lcs_median = out
lcs_median_true = out[-1:,:]
lcs_median_ratio = lcs_median/lcs_median_true






with np.load(results_path+summary_results_name,allow_pickle=True) as data:
    for k in data.keys():
        x = data['x']
        y = data['y']
        x_log_norm_rot = data['x_log_norm_rot']
        bins_edges_log_norm_rot = data['bins_edges_log_norm_rot']
        gates = data['gates']
        arrow_coordinates = data['arrow_coordinates']
        text_coordinates = data['text_coordinates']
        sigmax = data['sigmax']
        mus_x = data['mus_x']
        n_gates = data['n_gates']

df_mut = pd.read_csv(data_path+mut_data_name,header=[0,1],index_col=0)

# with np.load('/Users/vdoffini/Data/sorting_error_2D/uni_h_crit_out.npz',allow_pickle=True) as data:
#     Ns_tr_norm = data['Ns_tr_norm']
#     lcs_median = np.median(data['lcs'],axis=[2,3])[3,:,:]
#     lcs_median_true = np.median(data['lcs'],axis=[2,3])[0,:1,:]
#     lcs_median_ratio = lcs_median/lcs_median_true


#%%

plt.close('all')

fig = plt.figure(layout='constrained', figsize=(8.9, 9.5))#latex --> (5.5,9)

gs = fig.add_gridspec(4,2, hspace=0.05, wspace=0.05)
gs.set_height_ratios([0.25,1,1,1])
gs.set_width_ratios([1,1])

subfig_A = fig.add_subfigure(gs[0, :])
subfig_B = fig.add_subfigure(gs[1, 0])
subfig_C = fig.add_subfigure(gs[1, 1])
subfig_D = fig.add_subfigure(gs[2, 0])
subfig_E = fig.add_subfigure(gs[2, 1])
subfig_F = fig.add_subfigure(gs[3, 0])
subfig_G = fig.add_subfigure(gs[3, 1])

# subfig_A.set_facecolor('0.9')
# subfig_B.set_facecolor('0.8')
# subfig_C.set_facecolor('0.7')
# subfig_D.set_facecolor('0.6')
# subfig_E.set_facecolor('0.5')
# subfig_F.set_facecolor('0.4')
# subfig_G.set_facecolor('0.3')

gs_C = subfig_C.add_gridspec(1,8)
gs_D = subfig_D.add_gridspec(1,1)
gs_E = subfig_E.add_gridspec(3,1)
gs_F = subfig_F.add_gridspec(1,1)
gs_G = subfig_G.add_gridspec(1,1)

for i in range(8):
    locals()[f'ax_C_{i}'] = subfig_C.add_subplot(gs_C[0,i])

ax_D = subfig_D.add_subplot(gs_D[0,0])
for i in range(3):
    locals()[f'ax_E_{i}'] = subfig_E.add_subplot(gs_E[i,0])
    locals()[f'ax_E_{i}_secx'] = locals()[f'ax_E_{i}'].secondary_xaxis('top')
    locals()[f'ax_E_{i}_sec'] = locals()[f'ax_E_{i}'].twinx()

ax_F = subfig_F.add_subplot(gs_F[0,0])
ax_G = subfig_G.add_subplot(gs_G[0,0])

#%% subfig A

pos_min = 0.075
pos_max = 1-pos_min

ss = [r'$h$',
      r'$z_{0}$ (or $h_{0}$)',
      r'$\bar{N}_{s}$',
      r'$p_{WT}$',
      r'$WT,\, v,\, n$',
      r'$\mu_{y,i}=f\left(x_{i}\right),\, \sigma_{y}$']

subfig_A_text_positions = []
# [[[x0,y0],
#   [x0,y1]],
#    ...
#  ]

# loop to get the width of each text
for c,s in zip([0.5]*len(ss),ss):    
    temp_txt = plt.text(c, 0.5, s, horizontalalignment='center', fontsize=fontsize_sup,
                        verticalalignment='center', transform=subfig_A.transSubfigure, wrap=False)
    
    #set text to be out frame (needed for constrained_layout)
    temp_txt.set_in_layout(False)
        
    # textbox position (in pixels?)
    t = temp_txt.get_window_extent(renderer = fig.canvas.get_renderer())
    subfig_A_text_positions.append(fig.transFigure.inverted().transform(t))
    
    #remove temporary text
    Artist.remove(temp_txt)

subfig_A_text_positions = np.array(subfig_A_text_positions)
subfig_A_text_widths = np.diff(subfig_A_text_positions[:,:,0]).flatten()
delta_text = (pos_max-pos_min-subfig_A_text_widths.sum())/(len(ss)-1)

pos = pos_min
for i,s in enumerate(ss):
    
    pos += subfig_A_text_widths[i]/2
    
    temp_txt = plt.text(pos, 0.5, s, horizontalalignment='center', fontsize=fontsize_sup,
                        verticalalignment='center', transform=subfig_A.transSubfigure, wrap=False)
    
    #set text to be out frame (needed for constrained_layout)
    temp_txt.set_in_layout(False)
    
    pos = pos + delta_text + subfig_A_text_widths[i]/2
    

#%% subfig B

# ss = ["".join(r'FFFSARG'),r'$\ \rightarrow\ m_{}=0;\ N_{norm}\approx 0.000\ 1$','\n',
#       r'\textbf{a}',"","".join(r'FFSARG'),r'$\ \rightarrow\ m_{}=1;\ N_{norm}\approx 0.179;\ N_{}=\ \ 2$','\n',
#       r'\textbf{g}',"","".join(r'FFSARG'),r'$\ \rightarrow\ m_{}=1;\ N_{norm}\approx 0.290;\ N_{}=\ \ 3$','\n',
#       r'...','\n',
#       r'\textbf{a}',"",r'\textbf{a}',"","".join(r'FSARG'),r'$\ \rightarrow\ m_{}=2;\ N_{norm}\approx 1.011;\ N_{}=\ 30$','\n',
#       r'\textbf{a}',"",r'\textbf{g}',"","".join(r'FSARG'),r'$\ \rightarrow\ m_{}=2;\ N_{norm}\approx 1.023;\ N_{}=\ 31$','\n',
#       r'...','\n',
#       r'\textbf{a}',"",r'\textbf{a}',"",r'\textbf{a}',"","".join(r'SARG'),r'$\ \rightarrow\ m_{}=3;\ N_{norm}\approx 2.000;\ N_{}=365$','\n',
#       r'\textbf{a}',"",r'\textbf{a}',"",r'\textbf{g}',"","".join(r'SARG'),r'$\ \rightarrow\ m_{}=3;\ N_{norm}\approx 2.001;\ N_{}=366$','\n',
#       r'...'
#       ]

# ss = ["".join(r'FFFSARG'),r'$\ \rightarrow\ N_{norm} = 0;\phantom{(-0];}\ m=0$','\n',
#       r'\textbf{a}',"","".join(r'FFSARG'),r'$\ \rightarrow\ N_{norm}\in (0-1];\ m=1$','\n',
#       r'\textbf{g}',"","".join(r'FFSARG'),r'$\ \rightarrow\ N_{norm}\in (0-1];\ m=1$','\n',
#       r'...','\n',
#       r'\textbf{a}',"",r'\textbf{a}',"","".join(r'FSARG'),r'$\ \rightarrow\ N_{norm}\in (1-2];\ m=2$','\n',
#       r'\textbf{a}',"",r'\textbf{g}',"","".join(r'FSARG'),r'$\ \rightarrow\ N_{norm}\in (1-2];\ m=2$','\n',
#       r'...','\n',
#       r'\textbf{a}',"",r'\textbf{a}',"",r'\textbf{a}',"","".join(r'SARG'),r'$\ \rightarrow\ N_{norm}\in (2-3];\ m=3$','\n',
#        r'\textbf{a}',"",r'\textbf{a}',"",r'\textbf{g}',"","".join(r'SARG'),r'$\ \rightarrow\ N_{norm}\in (2-3];\ m=3$','\n',
#       r'...','\n',
#        # r'\textbf{s}',"",r'\textbf{s}',"",r'\textbf{s}',"",r'\textbf{r}',"",r'\textbf{s}',"",r'\textbf{s}',"",r'\textbf{s}',"",r'$\ \rightarrow\ N_{norm}\in (6-7];\ m=7$'
#       # r'\texttt{sssrsss}',r'$\ \rightarrow\ N_{norm}\in (6-7];\ m=7$','\n'
#       ]

ss = [r'\texttt{FFFSARG}',r'$\ \rightarrow\ N_{norm} = 0;\phantom{(-0];}\ m=0$','\n',
      r'\texttt{aFFSARG}',r'$\ \rightarrow\ N_{norm}\in (0-1];\ m=1$','\n',
      r'\texttt{gFFSARG}',r'$\ \rightarrow\ N_{norm}\in (0-1];\ m=1$','\n',
      r'...','\n',
      r'\texttt{aaFSARG}',r'$\ \rightarrow\ N_{norm}\in (1-2];\ m=2$','\n',
      r'\texttt{agFSARG}',r'$\ \rightarrow\ N_{norm}\in (1-2];\ m=2$','\n',
      r'...','\n',
      r'\texttt{aaaSARG}',r'$\ \rightarrow\ N_{norm}\in (2-3];\ m=3$','\n',
      r'...','\n',
      r'\texttt{sssrsss}',r'$\ \rightarrow\ N_{norm}\in (6-7];\ m=7$','\n'
      ]

        
temp_txt = plt.text(0.5, 0.45, ''.join(ss), horizontalalignment='center', fontsize=fontsize_sup,
                    verticalalignment='center', transform=subfig_B.transSubfigure, wrap=False, fontdict={'family': 'monospace'})
temp_txt.set_in_layout(False)


#%% subfig C

p_wt_C = '0.6900000000000001'

y_lim_C = [df_mut.Bin.loc[:,p_wt_C].min()*0.5,
           df_mut.Bin.loc[:,p_wt_C].max()*2,]


for i in range(8):
    locals()[f'ax_C_{i}'].xaxis.set_label_position("top")
    locals()[f'ax_C_{i}'].yaxis.set_label_position("right")

    temp = df_mut.Info.query(f'n_mutations=={i}').index
    if i == 0:
        locals()[f'ax_C_{i}'].plot([0.5],df_mut.Bin.loc[temp,p_wt_C],'k.')
        locals()[f'ax_C_{i}'].set_xlabel(r'$m$',x=0.5, alpha=0)

    else:
        locals()[f'ax_C_{i}'].plot(np.linspace(0,1,temp.shape[0]),df_mut.Bin.loc[temp,p_wt_C],'k.')
    
    locals()[f'ax_C_{i}'].set_yscale('log')
    locals()[f'ax_C_{i}'].set_ylim(y_lim_C)

    # locals()[f'ax_C_{i}'].set_yscale('log')
    if i == 7:
        locals()[f'ax_C_{i}'].set_ylabel(r'$P\left(x_{i}|m,\,n=7,\,v=5\right)$')
    else:
        locals()[f'ax_C_{i}'].set_yticklabels([])
    
    locals()[f'ax_C_{i}'].set_xticks([0.5])
    locals()[f'ax_C_{i}'].set_xticklabels(str(i))
    # locals()[f'ax_C_{i}'].tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
    #                                   left=True, labelleft=True, right=False, labelright=False)
    locals()[f'ax_C_{i}'].tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                                      left=False, labelleft=False, right=True, labelright=True)



#%% subfig D

ax_D.scatter(x[:,0],x[:,1],s=5,c = y/y.max(), cmap = cmap)
ax_D.scatter(x[:,0],x[:,1],s=2,color = 'k')
ax_D.set_aspect('equal')
 
ax_D.set_xscale('log')
ax_D.set_yscale('log')

ax_D.set_ylim(1, np.max([ax_D.get_xlim()[1],ax_D.get_ylim()[1]]))
ax_D.set_xlim(1, np.max([ax_D.get_xlim()[1],ax_D.get_ylim()[1]]))

for i in range(len(gates)):
    ax_D.add_patch(PathPatch(Path(gates[i]), edgecolor = 'grey', alpha = 1, facecolor='none',zorder=np.inf))

ax_D.set_xlabel('Binding (Red)')
ax_D.set_ylabel('Expression (Green)')
ax_D.xaxis.set_label_position("top")

ax_D.annotate('', xy=arrow_coordinates[0,:], xytext=arrow_coordinates[1,:],arrowprops=dict(arrowstyle="<-"), horizontalalignment='center',
              verticalalignment='center',transform=ax_D.transData, annotation_clip=False, clip_on=False)
ax_D.text(text_coordinates[0,0], text_coordinates[0,1], '$y_i$'+' or '+'$z_i$' , horizontalalignment='center',
          verticalalignment='center', rotation=-45)

ax_D.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                 left=True, labelleft=True, right=False, labelright=False)


try:
    ax_D.set_xlim(1, 10**7)
    
    ax_D.scatter(1680,38,s=5,c = [cmap(1.)])
    ax_D.scatter(1680,38,s=2,color = 'k')
    ax_D.scatter(1680,2800,s=5,c = [cmap(0.5)])
    ax_D.scatter(1680,2800,s=2,color = 'k')
    
    ax_D_ins1_x0 = 0.55
    ax_D_ins1_w = 1-ax_D_ins1_x0-0.005
    ax_D_ins1_h_delta = 0.009
    ax_D_ins1_h = (1-3*ax_D_ins1_h_delta)/2
    ax_D_ins1 = ax_D.inset_axes([ax_D_ins1_x0,ax_D_ins1_h_delta*0.5+0.5,ax_D_ins1_w,ax_D_ins1_h],zorder=np.inf, xticks=[], yticks=[])
    ax_D_ins2 = ax_D.inset_axes([ax_D_ins1_x0,ax_D_ins1_h_delta,ax_D_ins1_w,ax_D_ins1_h],zorder=np.inf, xticks=[], yticks=[])
    
    zoom_h = 0.05
    zoom_w = zoom_h/ax_D_ins1_h*ax_D_ins1_w
    ax_D_ins1_zoom_x = 0.445
    ax_D_ins1_zoom_y = 0.79
    ax_D_ins2_zoom_x = ax_D_ins1_zoom_x
    ax_D_ins2_zoom_y = 0.3475
    ax_D_ins1_zoom = ax_D.inset_axes([ax_D_ins1_zoom_x,ax_D_ins1_zoom_y,zoom_w,zoom_h],zorder=np.inf, xticks=[], yticks=[])
    ax_D_ins1_zoom.patch.set_alpha(0.)
    ax_D_ins2_zoom = ax_D.inset_axes([ax_D_ins2_zoom_x,ax_D_ins2_zoom_y,zoom_w,zoom_h],zorder=np.inf, xticks=[], yticks=[])
    ax_D_ins2_zoom.patch.set_alpha(0.)

    
    cp_ins1 = ConnectionPatch((ax_D_ins1_zoom_x,ax_D_ins1_zoom_y+zoom_h), (ax_D_ins1_x0,ax_D_ins1_h+ax_D_ins1_h_delta*0.5+0.5), ax_D.transAxes, ax_D.transAxes)
    cp_ins1.set_zorder(np.inf)
    cp_ins1.set_linestyle((0,(2,2)))
    ax_D.add_artist(cp_ins1)
    
    cp_ins2 = ConnectionPatch((ax_D_ins2_zoom_x,ax_D_ins2_zoom_y), (ax_D_ins1_x0,ax_D_ins1_h_delta), ax_D.transAxes, ax_D.transAxes)
    cp_ins2.set_zorder(np.inf)
    cp_ins2.set_linestyle((0,(2,2)))
    ax_D.add_artist(cp_ins2)
    
    # ax_D_ins1.indicate_inset_zoom(ax_D_ins1_zoom, edgecolor='k')
    # ax_D_ins2.indicate_inset_zoom(ax_D_ins2_zoom, edgecolor='k')

    
    ax_D_ins1_ins_x0 = 0.62
    ax_D_ins1_ins_w = 1-ax_D_ins1_ins_x0-0.015
    ax_D_ins1_ins_h_delta = 0.025
    ax_D_ins1_ins_h = (1-2*ax_D_ins1_ins_h_delta)
    ax_D_ins1_ins = ax_D_ins1.inset_axes([ax_D_ins1_ins_x0,ax_D_ins1_ins_h_delta,ax_D_ins1_ins_w,ax_D_ins1_ins_h],zorder=np.inf, xticks=[], yticks=[])
    ax_D_ins2_ins = ax_D_ins2.inset_axes([ax_D_ins1_ins_x0,ax_D_ins1_ins_h_delta,ax_D_ins1_ins_w,ax_D_ins1_ins_h],zorder=np.inf, xticks=[], yticks=[])
    
    ins_zoom_h = 0.1
    ins_zoom_w = 0.07
    ax_D_ins1_ins_zoom_x = 0.2375
    ax_D_ins1_ins_zoom_y = 0.5375
    ax_D_ins2_ins_zoom_x = 0.305
    ax_D_ins2_ins_zoom_y = 0.618
    ax_D_ins1_ins_zoom = ax_D_ins1.inset_axes([ax_D_ins1_ins_zoom_x,ax_D_ins1_ins_zoom_y,ins_zoom_w,ins_zoom_h],zorder=np.inf, xticks=[], yticks=[])
    ax_D_ins1_ins_zoom.patch.set_alpha(0.)
    ax_D_ins2_ins_zoom = ax_D_ins2.inset_axes([ax_D_ins2_ins_zoom_x,ax_D_ins2_ins_zoom_y,ins_zoom_w,ins_zoom_h],zorder=np.inf, xticks=[], yticks=[])
    ax_D_ins2_ins_zoom.patch.set_alpha(0.)
    
    cp_ins1_ins = ConnectionPatch((ax_D_ins1_ins_zoom_x,ax_D_ins1_ins_zoom_y+ins_zoom_h), (ax_D_ins1_ins_x0,ax_D_ins1_ins_h_delta+ax_D_ins1_ins_h), ax_D_ins1.transAxes, ax_D_ins1.transAxes)
    cp_ins1_ins.set_zorder(np.inf)
    cp_ins1_ins.set_linestyle((0,(2,2)))
    ax_D_ins1.add_artist(cp_ins1_ins)

    cp_ins2_ins = ConnectionPatch((ax_D_ins2_ins_zoom_x,ax_D_ins2_ins_zoom_y), (ax_D_ins1_ins_x0,ax_D_ins1_ins_h_delta), ax_D_ins2.transAxes, ax_D_ins2.transAxes)
    cp_ins2_ins.set_zorder(np.inf)
    cp_ins2_ins.set_linestyle((0,(2,2)))
    ax_D_ins2.add_artist(cp_ins2_ins)
    
    
    
    for ax in [ax_D_ins1_zoom, ax_D_ins2_zoom, ax_D_ins1_ins_zoom, ax_D_ins2_ins_zoom]:
        for spine in ax.spines.values():
            spine.set_linestyle((0,(2,2)))

    
    
    from PIL import Image
    
    img_cell1 = Image.open("./F1_blender/cell1.png")
    img_cell2 = Image.open("./F1_blender/cell2.png")
    img_prot1 = Image.open("./F1_blender/prot1_2.png")
    img_prot2 = Image.open("./F1_blender/prot2_2.png")
    
    ax_D_ins1.imshow(img_cell1, interpolation='nearest', aspect='auto')
    ax_D_ins2.imshow(img_cell2, interpolation='nearest', aspect='auto')
    ax_D_ins1_ins.imshow(img_prot1, interpolation='nearest', aspect='auto')
    ax_D_ins2_ins.imshow(img_prot2, interpolation='nearest', aspect='auto')
    
    
    temp_txt = plt.text(0.05,0.0, r'a', horizontalalignment='left', fontsize=fontsize,
                        verticalalignment='bottom', transform=ax_D_ins1_ins.transAxes, wrap=False)    

    temp_txt.set_in_layout(False)

    ax_D_ins1_ins.annotate('', xy=(0.2,0.35), xytext=(0.15,0.15), xycoords='axes fraction', textcoords='axes fraction', 
                            arrowprops=dict(arrowstyle="->"), horizontalalignment='center', fontsize=fontsize,
                            verticalalignment='center', transform=ax_D_ins1_ins.transAxes, annotation_clip=True, clip_on=True)

    temp_txt = plt.text(0.05,0.975, r'b', horizontalalignment='left', fontsize=fontsize,
                        verticalalignment='top', transform=ax_D_ins1_ins.transAxes, wrap=False)
    
    temp_txt.set_in_layout(False)
    
    ax_D_ins1_ins.annotate('', xy=(0.575,0.8), xytext=(0.225,0.9), xycoords='axes fraction', textcoords='axes fraction', 
                            arrowprops=dict(arrowstyle="->"), horizontalalignment='center', fontsize=fontsize,
                            verticalalignment='center', transform=ax_D_ins1_ins.transAxes, annotation_clip=True, clip_on=True)

    # temp_txt = plt.text(0.02,0.0, r'$ii$', horizontalalignment='left', fontsize=fontsize,
    #                     verticalalignment='bottom', transform=ax_D_ins2_ins.transAxes, wrap=False)    

    # temp_txt.set_in_layout(False)

    # ax_D_ins2_ins.annotate('', xy=(0.2,0.35), xytext=(0.15,0.15), xycoords='axes fraction', textcoords='axes fraction', 
    #                         arrowprops=dict(arrowstyle="->"), horizontalalignment='center', fontsize=fontsize,
    #                         verticalalignment='center', transform=ax_D_ins2_ins.transAxes, annotation_clip=True, clip_on=True)

    temp_txt = plt.text(0.05,0.975, r'c', horizontalalignment='left', fontsize=fontsize,
                        verticalalignment='top', transform=ax_D_ins2_ins.transAxes, wrap=False)
    
    temp_txt.set_in_layout(False)
    
    ax_D_ins2_ins.annotate('', xy=(0.575,0.75), xytext=(0.225,0.9), xycoords='axes fraction', textcoords='axes fraction', 
                            arrowprops=dict(arrowstyle="->"), horizontalalignment='center', fontsize=fontsize,
                            verticalalignment='center', transform=ax_D_ins2_ins.transAxes, annotation_clip=True, clip_on=True)
        


except:
    1

#%% subfig E

y_space = np.linspace(bins_edges_log_norm_rot.min(),bins_edges_log_norm_rot.max(),100)
y_space_norm = n_gates*(y_space-y_space.min())/(y_space.max()-y_space.min())
bins_edges_log_norm_rot_norm = np.arange(n_gates[0]+1)

    # locals()[f'ax_E_{i}'] = subfig_E.add_subplot(gs_E[i,0])
    # locals()[f'ax_E_{i}_secx'] = locals()[f'ax_E_{i}'].secondary_xaxis('top')
    # locals()[f'ax_E_{i}_sec'] = locals()[f'ax_E_{i}'].twinx()

for i,mu in enumerate(mus_x):
    locals()[f'ax_E_{i}'].set_xlim(bins_edges_log_norm_rot_norm.min(),bins_edges_log_norm_rot_norm.max())
    
    gkde = gaussian_kde(x_log_norm_rot[y==i,0])
    # locals()[f'ax_E_{i}'].plot(y_space_norm, gkde.pdf(y_space), linestyle='--', color=cmap(i/(mus_x.size-1+1e-8)), lw=1.5, path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()], zorder = 10000)
    locals()[f'ax_E_{i}'].plot(y_space_norm, norm(mu,sigmax).pdf(y_space), linestyle='--', color=cmap(i/(mus_x.size-1+1e-8)), lw=1.5, path_effects=[pe.Stroke(linewidth=2.5, foreground='k'), pe.Normal()], zorder = 10000)
    
    temp = np.histogram(x_log_norm_rot[y==i,0],bins_edges_log_norm_rot)[0]
    temp = temp/temp.sum()
    
    color_temp = list(cmap(i/(mus_x.size-1+1e-8)))
    color_temp[-1] = 0.7

    locals()[f'ax_E_{i}_sec'].bar(bins_edges_log_norm_rot_norm[:-1]+0.5,temp, width=1, color=color_temp, zorder = 0)
    
    locals()[f'ax_E_{i}_sec'].set_zorder(1)
    locals()[f'ax_E_{i}'].set_zorder(2)
    locals()[f'ax_E_{i}'].patch.set_visible(False)
    
    locals()[f'ax_E_{i}'].set_ylim(0,None)
    locals()[f'ax_E_{i}_sec'].set_ylim(0,1)
    
    if i == 0:
        locals()[f'ax_E_{i}_secx'].set_xlabel('$z_{i}$')
        locals()[f'ax_E_{i}_secx'].set_xticks(bins_edges_log_norm_rot_norm[:-1]+0.5)
    else:
        locals()[f'ax_E_{i}_secx'].set_xticks(bins_edges_log_norm_rot_norm[:-1]+0.5)
        locals()[f'ax_E_{i}_secx'].set_xticklabels([])
               
    if i == 1:
        locals()[f'ax_E_{i}'].set_ylabel(r'$\phi\left(y_{i}\right)=dP\left(y_{i}\right)/dy_{i}$')
        locals()[f'ax_E_{i}_sec'].set_ylabel(r'$P\left(z_{i}\right)$')

    if i == 2:
        locals()[f'ax_E_{i}'].set_xlabel('$y_{i}$')
        locals()[f'ax_E_{i}'].set_xticks(bins_edges_log_norm_rot_norm)
    else:
        locals()[f'ax_E_{i}'].set_xticks(bins_edges_log_norm_rot_norm)
        locals()[f'ax_E_{i}'].set_xticklabels([])
    
    for ii in bins_edges_log_norm_rot_norm[1:-1]:
        locals()[f'ax_E_{i}_sec'].plot([ii,ii],[0,1],color='grey',linewidth=0.8)

y_lim = np.max([locals()[f'ax_E_{i}'].get_ylim()[1] for i,mu in enumerate(mus_x)])
for i,mu in enumerate(mus_x):
    locals()[f'ax_E_{i}'].set_ylim(0,y_lim)


#%% subfig F

temp = 0.1
ax_F.text(temp+0.075, 0.35, r'$\mu_{y}$', horizontalalignment='center', fontsize=fontsize_sup,
                verticalalignment='center', transform=ax_F.transAxes, wrap=False)
ax_F.annotate(r'$\mu_{z}$', xy=(temp+0.11, 0.35), xytext=(temp+0.225, 0.35), xycoords='axes fraction', textcoords='axes fraction', 
              arrowprops=dict(arrowstyle="->"), horizontalalignment='center', fontsize=fontsize_sup,
              verticalalignment='center', transform=ax_F.transAxes, annotation_clip=True, clip_on=True)
ax_F.annotate(r'$\bar{y}$', xy=(temp+0.075, 0.3), xytext=(temp+0.075, 0.1), xycoords='axes fraction', textcoords='axes fraction', 
              arrowprops=dict(arrowstyle="->"), horizontalalignment='center', fontsize=fontsize_sup,
              verticalalignment='center', transform=ax_F.transAxes, annotation_clip=True, clip_on=True)
ax_F.annotate(r'$\bar{z}$', xy=(temp+0.11, 0.3), xytext=(temp+0.225, 0.1), xycoords='axes fraction', textcoords='axes fraction', 
              arrowprops=dict(arrowstyle="->"), horizontalalignment='center', fontsize=fontsize_sup,
              verticalalignment='center', transform=ax_F.transAxes, annotation_clip=True, clip_on=True)

temp_d = 0.1
fig_width, fig_height = fig.get_size_inches()
_,_,temp_w,temp_h = ax_F.get_position().bounds
arc = Arc((0.175-temp_d/2, 0.35+temp_d*1.5/2),
          temp_d, temp_d*1.4232209737827715,
          angle=0, theta1=0, theta2=270,
           linewidth=1, color='k',
          transform=ax_F.transAxes)
ax_F.add_patch(arc)
ax_F.annotate(r'', xy=(0.175, 0.35+temp_d*1.5/2-1e-2), xytext=(0.175, 0.35+temp_d*1.5/2), xycoords='axes fraction', textcoords='axes fraction', 
              arrowprops=dict(arrowstyle="->"), horizontalalignment='center', fontsize=fontsize_sup,
              verticalalignment='center', transform=ax_F.transAxes, annotation_clip=True, clip_on=True)



ax_F.set_ylabel(r'$MAE_{\mu_y\leftarrow k\ =\ \mu_y\ |\ \mu_z\ |\ \bar{y}\ |\ \bar{z}}^{test}$')
ax_F.set_xlabel('$N_{norm}^{train}$')
ax_F.set_yscale('log')

ax_F.xaxis.set_label_position("top")
ax_F.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                 left=True, labelleft=True, right=False, labelright=False)

for i in range(51):
    ax_F.semilogy(Ns_tr_norm,lcs_median[i],color=cmap_LC(i/50))
ax_F.semilogy(Ns_tr_norm,lcs_median_true[0],':',color='k')


#%% subfig G

ax_G.set_ylabel(r'$MAE_{\mu_y\leftarrow k}^{test}\ /\ MAE_{\mu_y\leftarrow \mu_y}^{test}$')
ax_G.set_xlabel('$N_{norm}^{train}$')
ax_G.set_yscale('log')

ax_G.xaxis.set_label_position("top")
ax_G.yaxis.set_label_position("right")
ax_G.tick_params(which='both',top=True, labeltop=True, bottom=False, labelbottom=False,
                 left=False, labelleft=False, right=True, labelright=True)

for i in range(51):
    ax_G.semilogy(Ns_tr_norm,lcs_median_ratio[i],color=cmap_LC(i/50))
ax_G.semilogy(Ns_tr_norm,[1]*len(Ns_tr_norm),':',color='k')


temp_lc_x = np.arange(51)
temp_lc_y = np.array([0 for i in temp_lc_x])
temp_lc_points = np.array([temp_lc_x, temp_lc_y]).T.reshape(-1, 1, 2)
temp_lc_segments = np.concatenate([temp_lc_points[:-1], temp_lc_points[1:]], axis=1)

lc = LineCollection(temp_lc_segments, cmap=cmap_LC,
                    norm=plt.Normalize(0, temp_lc_x.max()), linewidth=3)
lc.set_array(temp_lc_x)


# ax_G.legend(handles=[
#                      Line2D([0], [0], color=cmap_LC(0.5), lw=2, label=r'$k= \bar{z}\left(\uparrow \bar{N}_{s}\right)$'),
#                      LineCollection(temp_lc_segments, cmap=cmap_LC, label=r'$k= \bar{z}\left(\uparrow \bar{N}_{s}\right)$'),
#                      Line2D([0], [0], color='k', lw=2, linestyle=':', label=r'$k= \mu_{y}$'),
#                      ],
#     handler_map={lc: HandlerColorLineCollection(numpoints=4)}, framealpha=1,
#     handlelength=0.75)
ax_G.legend([
             lc,
             Line2D([0], [0], color='k', lw=2, linestyle=':'),
             ],
            [
             r'$k= \bar{z},\, HTS\left(-\rightarrow+\right)$',
             r'$k= \mu_{y}$'
             ],
    
    handler_map={lc: HandlerColorLineCollection(numpoints=51)}, framealpha=0,
    handlelength=2)


#%% Text

fig.canvas.draw()

xlabel_C_3 = locals()['ax_C_3'].text(locals()['ax_C_0'].get_position().x0 + (locals()['ax_C_0'].get_position().x0 + locals()[f'ax_C_{gs_C.get_geometry()[1]-1}'].get_position().x0 + locals()[f'ax_C_{gs_C.get_geometry()[1]-1}'].get_position().width)/2,
                                     1.175, r'$m$', horizontalalignment='center', fontsize=fontsize,
                                     verticalalignment='center', transform=transforms.blended_transform_factory(subfig_C.transSubfigure,locals()['ax_C_0'].transAxes), wrap=False)
xlabel_C_3.set_in_layout(False)


temp = 0.005

label_A = plt.text(temp, 1-temp, 'A', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                   verticalalignment='top', transform=fig.transFigure, wrap=False)
label_A.set_in_layout(False)

label_B = plt.text(temp, (subfig_B.bbox.y0+subfig_B.bbox.height)/fig.bbox.height-temp,
                   'B', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                   verticalalignment='top', transform=fig.transFigure, wrap=False)
label_B.set_in_layout(False)

label_D = plt.text(temp, (subfig_D.bbox.y0+subfig_D.bbox.height)/fig.bbox.height-temp,
                   'D', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                   verticalalignment='top', transform=fig.transFigure, wrap=False)
label_D.set_in_layout(False)

label_F = plt.text(temp, (subfig_F.bbox.y0+subfig_F.bbox.height)/fig.bbox.height-temp,
                   'F', horizontalalignment='left', fontsize=fontsize_letter,fontweight='bold',
                   verticalalignment='top', transform=fig.transFigure, wrap=False)
label_F.set_in_layout(False)


label_C = plt.text(1-temp, (subfig_C.bbox.y0+subfig_C.bbox.height)/fig.bbox.height-temp,
                   'C', horizontalalignment='right', fontsize=fontsize_letter,fontweight='bold',
                   verticalalignment='top', transform=fig.transFigure, wrap=False)
label_C.set_in_layout(False)

label_E = plt.text(1-temp, (subfig_E.bbox.y0+subfig_E.bbox.height)/fig.bbox.height-temp,
                   'E', horizontalalignment='right', fontsize=fontsize_letter,fontweight='bold',
                   verticalalignment='top', transform=fig.transFigure, wrap=False)
label_E.set_in_layout(False)

label_G = plt.text(1-temp, (subfig_G.bbox.y0+subfig_G.bbox.height)/fig.bbox.height-temp,
                   'G', horizontalalignment='right', fontsize=fontsize_letter,fontweight='bold',
                   verticalalignment='top', transform=fig.transFigure, wrap=False)
label_G.set_in_layout(False)


#%% Save
plt.savefig('F1'+".svg",dpi = 300, format="svg",transparent=True)
plt.savefig('F1'+'.png',dpi = 400, format='png')