#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:38:09 2024

@author: vdoffini
"""
import numpy as np
from scipy.stats import norm,rv_discrete,rv_histogram
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import CubicSpline
import warnings
import sys

#%% Parameters
'''
ToDo:
    test var_multiplier = np.sqrt(2/np.pi), which is E[|N(0,sigma)|] see --> https://en.wikipedia.org/wiki/Half-normal_distribution
    implement str_variable = entropy
'''

str_variable = 'error'
var_multiplier = np.sqrt(2/np.pi)
str_distr = 'disc'

bool_plot = False # if plot or not

bool_save = True # if save data (not plot) or not
save_path = '../Results/'
save_name = f'02_bias_vs_variance_{str_variable}.npy'

hs_crit = np.sqrt(6)

#%% Derived parameters
mu=0 #do not change
sigma = 1. #do not change
pop_cont = norm(mu,sigma)

data_2D_hist = np.load(save_path+'01_2D_hist.npy',allow_pickle=True).flatten()[0]
HSs = data_2D_hist['HSs']#[::5,:][:,::5]
n_grid = HSs.shape[0] # nuber of grid points (n_grid * n_grid)
if np.unique(HSs[:,0]).size < n_grid:
    hs_s = HSs[0,:]
else:
    hs_s = HSs[:,0]

multiplier4maxBin = data_2D_hist['multiplier4maxBin'][0]
max_bin = HSs.max()*multiplier4maxBin

#%% Functions

def h_norm(sigma):
    return 0.5*(1+np.log(2*np.pi))+np.log(sigma)

def f_out(inputs, str_return,str_distr = str_distr, str_variable = str_variable, max_bin = max_bin, pop_cont = pop_cont, var_multiplier = var_multiplier):
    zh0,hs,N = inputs
    
    b = hs*sigma*np.ceil(max_bin/hs)
    bins_tot = zh0*hs*sigma + mu + np.concatenate((-np.linspace(0,b,np.ceil(max_bin/hs).astype(int)*2+1)[1:][::-1],
                                                   np.linspace(0,b,np.ceil(max_bin/hs).astype(int)*2+1)))
    bins_edges = bins_tot[::2]
    bins_median = bins_tot[1::2]
    p_temp = np.diff(pop_cont.cdf(bins_edges))
    
    if str_distr.lower() == 'disc':
        pop_aprox = rv_discrete(values=(bins_median,p_temp))
    elif str_distr.lower() == 'hist':
        pop_aprox = rv_histogram(histogram=(p_temp,bins_edges))
    else:
        raise ValueError(f'str_distr should be equal to "disc" or "hist". Current value: "{str_distr}".')

    if str_variable.lower() == 'error':
        if str_return.lower() == 'bias':
            return np.abs(pop_aprox.mean() - pop_cont.mean())
        elif str_return.lower() == 'var':
            return pop_aprox.std()/np.sqrt(N)*var_multiplier
        else:
            raise ValueError(f'str_return should be equal to "bias" or "var". Current value: "{str_return}".')
    elif str_variable.lower() == 'entropy':
        if str_return.lower() == 'bias':
            return pop_aprox.entropy()-pop_cont.entropy()
        elif str_return.lower() == 'var':
            return h_norm(pop_aprox.std()/np.sqrt(N))
        else:
            raise ValueError(f'str_return should be equal to "bias" or "var". Current value: "{str_return}".')
    else:
        raise ValueError(f'str_variable should be equal to "error" or "entropy". Current value: "{str_variable}".')

def quad_correction_points(h,bool_macos=sys.platform == 'darwin'):
    if bool_macos:
        return None
    else:
        if h>10:
            return 10/h
        else:
            return None
    

#%% Calculate n_crit and Ns
#   -> n_crit is the N_sample at which bias==var & hs==hs_crit

n_crit_out = minimize_scalar(lambda y: np.abs(quad(lambda x:x*f_out([x,hs_crit*sigma,10**y], str_return = 'bias'), 0.,0.5)[0]
                                              -quad(lambda x:x*f_out([x,hs_crit*sigma,10**y], str_return = 'var'), 0.,0.5)[0])
                             ,bounds=[0,10])

if n_crit_out.success == False:
    warnings.warn(f'n_crit calulation got a problem. Message: "{n_crit_out.message}"')

n_crit = 10**n_crit_out.x

n_s_span = (0,n_crit_out.x*2)# min and distance between (true) mean and closest gate edge / gates width

n_s = np.logspace(*n_s_span,n_grid)

_,Ns = np.meshgrid(hs_s,n_s)    

#%% Calculate contour_crit
#   -> contour_crit is the value of bias+var @ hs_crit*sigma and n_crit
contour_crit = (quad(lambda x:x*f_out([x,hs_crit*sigma,n_crit], str_return = 'bias'), 0.,0.5)[0]*2+
                quad(lambda x:x*f_out([x,hs_crit*sigma,n_crit], str_return = 'var'), 0.,0.5)[0]*2)

#%% 2D grid
#   -> calculate bias, var and bias+var for each value in HSs and Ns

bias = []
bias_integral_error = []
var = []
var_integral_error = []
bias_plus_var = []
bias_plus_var_integral_error = []
for i,(hs,n) in enumerate(zip(HSs.flatten(),
                              Ns.flatten())):
    h = hs*sigma
    
    print(i,(hs,n))
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias.append(temp[0]*2)
    bias_integral_error.append(temp[1]*2)
    print(f'b={temp[0]}')
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var.append(temp[0]*2)
    var_integral_error.append(temp[1]*2)
    print(f'v={temp[0]*2}')
    bias_plus_var.append(bias[-1]+var[-1])
    bias_plus_var_integral_error.append(bias_integral_error[-1]+var_integral_error[-1])
    

bias = np.array(bias).reshape(HSs.shape)
bias_integral_error = np.array(bias_integral_error).reshape(HSs.shape)
print(f'max bias_integral_error: {np.abs(bias_integral_error).max()}')
var = np.array(var).reshape(HSs.shape)
var_integral_error = np.array(var_integral_error).reshape(HSs.shape)
print(f'max var_integral_error: {np.abs(var_integral_error).max()}')
bias_plus_var = np.array(bias_plus_var).reshape(HSs.shape)
bias_plus_var_integral_error = np.array(bias_plus_var_integral_error).reshape(HSs.shape)
print(f'max bias_plus_var_integral_error: {np.abs(bias_plus_var_integral_error).max()}')

#%% h critical
#   -> calculate bias, var and bias+var @ hs_crit*sigma (changing N_sample)
bias_h_crit = []
var_h_crit = []
for i,n in enumerate(n_s):
    h = hs_crit*sigma
    
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias_h_crit.append([n,h,temp[0]*2])
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var_h_crit.append([n,h,temp[0]*2])
    
    print(i,(n,h))

bias_h_crit = np.array(bias_h_crit)
var_h_crit = np.array(var_h_crit)
bias_plus_var_h_crit = bias_h_crit.copy()
bias_plus_var_h_crit[:,-1] = bias_h_crit[:,-1] + var_h_crit[:,-1]

#%% n critical
#   -> calculate bias, var and bias+var @ n_crit (changing h)
bias_n_crit = []
var_n_crit = []
for i,hs in enumerate(hs_s):
    h = hs*sigma
    n = n_crit
    
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias_n_crit.append([n,h,temp[0]*2])
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var_n_crit.append([n,h,temp[0]*2])
    
    print(i,(n,h))

bias_n_crit = np.array(bias_n_crit)
var_n_crit = np.array(var_n_crit)
bias_plus_var_n_crit = bias_n_crit.copy()
bias_plus_var_n_crit[:,-1] = bias_n_crit[:,-1] + var_n_crit[:,-1]

#%% bias == var
#   -> calculate n and h at which bias==var (non-equidistant points)

bias_eq = []
var_eq = []
bias_plus_var_eq = []
t=[]
for i,n in enumerate(n_s):
    
    def f_temp(inputs):
        h,n = inputs
        temp1 = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)[0]*2
        temp2 = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5, points=quad_correction_points(h))[0]*2
        return np.abs(temp1-temp2)
    
    t.append(minimize_scalar(lambda h: f_temp([h,n]),bounds=[(hs_s*sigma).min(),(hs_s*sigma).max()]))
    h = t[-1]['x']
    
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias_eq.append([n,h,temp[0]*2])
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var_eq.append([n,h,temp[0]*2])
    
    print(i,(n,h))

bias_eq = np.array(bias_eq)
var_eq = np.array(var_eq)
bias_plus_var_eq = bias_eq.copy()
bias_plus_var_eq[:,-1] = bias_eq[:,-1] + var_eq[:,-1]

#%% bias == var (equidistant points 4 ML, it needs the previous cell to estimate the derivative)
#   -> calculate n and h at which bias==var (equidistant points)

temp_n = np.log10(var_eq[:,0])
temp_n_crit = np.log10(n_crit)
temp_n_crit = (temp_n_crit-temp_n.min())/(temp_n.max()-temp_n.min())
temp_n = (temp_n-temp_n.min())/(temp_n.max()-temp_n.min())

temp_h = np.log10(var_eq[:,1])
temp_h_crit = np.log10(hs_crit*sigma)
temp_h_crit = (temp_h_crit-temp_h.min())/(temp_h.max()-temp_h.min())
temp_h = (temp_h-temp_h.min())/(temp_h.max()-temp_h.min())
cs_n = CubicSpline(temp_n,temp_h)

if bool_plot:
    plt.figure()
    xx_ns = np.linspace(0,1,1000)
    plt.plot(xx_ns,cs_n(xx_ns))
#https://stackoverflow.com/questions/44962794/how-to-integrate-arc-lengths-using-python-numpy-and-scipy
s_n = quad(lambda n: np.sqrt(1+cs_n.derivative(1)(n)**2),temp_n_crit,temp_n.max())

norm_len = s_n[0]
norm_delta = norm_len/n_grid

norm_ns = [0]
for i in range(n_grid):
    f_temp = lambda x: np.abs(norm_delta - quad(lambda n: np.sqrt(1+cs_n.derivative(1)(n)**2),norm_ns[-1],x)[0])
    temp = minimize_scalar(f_temp, bounds=[norm_ns[-1],temp_n.max()*2])['x']
    if temp < 1:
        norm_ns.append(temp)
    else:
        break
norm_ns = np.array(norm_ns)

temp_n = np.log10(var_eq[:,0])
ns_equidistant = 10**(norm_ns*(temp_n.max()-temp_n.min())+temp_n.min())


bias_eq_equidistant = []
var_eq_equidistant = []
bias_plus_var_eq_equidistant = []
t=[]
for i,n in enumerate(n_s):
    
    def f_temp(inputs):
        h,n = inputs
        temp1 = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)[0]*2
        temp2 = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5, points=quad_correction_points(h))[0]*2
        return np.abs(temp1-temp2)
    
    t.append(minimize_scalar(lambda h: f_temp([h,n]),bounds=[(hs_s*sigma).min(),(hs_s*sigma).max()]))
    h = t[-1]['x']
    
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias_eq_equidistant.append([n,h,temp[0]*2])
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var_eq_equidistant.append([n,h,temp[0]*2])
    
    print(i,(n,h))
    
bias_eq_equidistant = np.array(bias_eq_equidistant)
var_eq_equidistant = np.array(var_eq_equidistant)
bias_plus_var_eq_equidistant = bias_eq_equidistant.copy()
bias_plus_var_eq_equidistant[:,-1] = bias_eq_equidistant[:,-1] + var_eq_equidistant[:,-1]


#%% contour_crit
#   -> calculate n and h at which bias+var == bias+var @n_crit&hs_crit*sigma (non-equidistant points)

bias_contour_crit = []
var_contour_crit = []
bias_plus_var_contour_crit = []
t=[]
def f_temp(inputs, contour_crit = contour_crit):
    h,n = inputs
    temp1 = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)[0]*2
    temp2 = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5, points=quad_correction_points(h))[0]*2
    return np.abs(np.log10(contour_crit)-np.log10(temp1+temp2))

for i,hs in enumerate(hs_s[hs_s<=hs_crit]):
    h = hs*sigma

    t.append(minimize_scalar(lambda n: f_temp([h,n]),bounds=[(n_s).min(),(n_s).max()]))
    
    n = t[-1]['x']

    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias_contour_crit.append([n,h,temp[0]*2])
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var_contour_crit.append([n,h,temp[0]*2])
        
    print(i,(n,h))

for ii,n in enumerate(n_s[n_s>n_crit]):
    t.append(minimize_scalar(lambda h: f_temp([h,n]),bounds=[(hs_s*sigma).min(),(hs_s*sigma).max()]))
    h = t[-1]['x']
    
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias_contour_crit.append([n,h,temp[0]*2])
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var_contour_crit.append([n,h,temp[0]*2])
        
    print(i+ii,(n,h))


bias_contour_crit = np.array(bias_contour_crit)
var_contour_crit = np.array(var_contour_crit)
bias_plus_var_contour_crit = bias_contour_crit.copy()
bias_plus_var_contour_crit[:,-1] = bias_contour_crit[:,-1] + var_contour_crit[:,-1]

#%% contour_crit (equidistant points 4 ML, it needs the previous cell to estimate the derivative)
#   -> calculate n and h at which bias+var == bias+var @n_crit&hs_crit*sigma (non-equidistant points)

temp_n = np.log10(var_contour_crit[:,0])
temp_n_crit = np.log10(n_crit)
temp_n_crit = (temp_n_crit-temp_n.min())/(temp_n.max()-temp_n.min())
temp_n = (temp_n-temp_n.min())/(temp_n.max()-temp_n.min())

temp_h = np.log10(var_contour_crit[:,1])
temp_h_crit = np.log10(hs_crit*sigma)
temp_h_crit = (temp_h_crit-temp_h.min())/(temp_h.max()-temp_h.min())
temp_h = (temp_h-temp_h.min())/(temp_h.max()-temp_h.min())
cs_n = CubicSpline(temp_n,temp_h)
cs_h = CubicSpline(temp_h,temp_n)

if bool_plot:
    plt.figure()
    xx_ns = np.linspace(0,1,1000)
    plt.plot(xx_ns,cs_n(xx_ns))
    xx_hs = np.linspace(0,1,1000)
    plt.plot(cs_h(xx_hs),xx_hs)
    plt.plot(temp_n,temp_h,'.')

#https://stackoverflow.com/questions/44962794/how-to-integrate-arc-lengths-using-python-numpy-and-scipy
s_n = quad(lambda n: np.sqrt(1+cs_n.derivative(1)(n)**2),temp_n_crit,temp_n.max())
s_h = quad(lambda h: np.sqrt(1+cs_h.derivative(1)(h)**2),temp_h.min(),temp_h_crit)

norm_len = s_h[0]+s_n[0]
norm_delta = norm_len/n_grid

norm_ns = [temp_n_crit]
for i in range(n_grid):
    f_temp = lambda x: np.abs(norm_delta - quad(lambda n: np.sqrt(1+cs_n.derivative(1)(n)**2),norm_ns[-1],x)[0])
    temp = minimize_scalar(f_temp, bounds=[norm_ns[-1],temp_n.max()*2])['x']
    if temp < 1:
        norm_ns.append(temp)
    else:
        break
norm_ns = np.array(norm_ns)

norm_hs = [temp_h_crit]
for i in range(n_grid):
    f_temp = lambda x: np.abs(norm_delta - quad(lambda h: np.sqrt(1+cs_h.derivative(1)(h)**2),x,norm_hs[-1])[0])
    temp = minimize_scalar(f_temp, bounds=[-temp_h.max(),norm_ns[-1]])['x']
    if temp > 0:
        norm_hs.append(temp)
    else:
        break
norm_hs = np.array(norm_hs)[::-1]

temp_n = np.log10(var_contour_crit[:,0])
ns_equidistant = 10**(norm_ns*(temp_n.max()-temp_n.min())+temp_n.min())

temp_h = np.log10(var_contour_crit[:,1])
hs_equidistant = 10**(norm_hs*(temp_h.max()-temp_h.min())+temp_h.min())/sigma

bias_contour_crit_equidistant = []
var_contour_crit_equidistant = []
bias_plus_var_contour_crit_equidistant = []
t=[]
def f_temp(inputs, contour_crit = contour_crit):
    h,n = inputs
    temp1 = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)[0]*2
    temp2 = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5, points=quad_correction_points(h))[0]*2
    return np.abs(np.log10(contour_crit)-np.log10(temp1+temp2))

if bool_plot:
    plt.figure()
    
for i,hs in enumerate(hs_equidistant[:-1]):
    h = hs*sigma

    t.append(minimize_scalar(lambda n: f_temp([h,n]),bounds=[(n_s).min(),(n_s).max()]))
    
    n = t[-1]['x']

    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias_contour_crit_equidistant.append([n,h,temp[0]*2])
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var_contour_crit_equidistant.append([n,h,temp[0]*2])
    
    if bool_plot:
        plt.plot(n,h,'.k')
        plt.pause(0.001)
    
    print(i,(n,h))

for ii,n in enumerate(ns_equidistant):
    t.append(minimize_scalar(lambda h: f_temp([h,n]),bounds=[(hs_s*sigma).min(),(hs_s*sigma).max()]))
    h = t[-1]['x']
    
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'bias'), 0.,0.5)
    bias_contour_crit_equidistant.append([n,h,temp[0]*2])
    temp = quad(lambda x:x*f_out([x,h,n], str_return = 'var'), 0.,0.5)
    var_contour_crit_equidistant.append([n,h,temp[0]*2])
    
    if bool_plot:
        plt.plot(n,h,'.k')
        plt.pause(0.001)
    
    print(i+ii,(n,h))


bias_contour_crit_equidistant = np.array(bias_contour_crit_equidistant)
var_contour_crit_equidistant = np.array(var_contour_crit_equidistant)
bias_plus_var_contour_crit_equidistant = bias_contour_crit_equidistant.copy()
bias_plus_var_contour_crit_equidistant[:,-1] = bias_contour_crit_equidistant[:,-1] + var_contour_crit_equidistant[:,-1]


#%% Plot

if bool_plot:
    if str_variable.lower() == 'error':
        norm_min = np.nanmin(np.log10(var))
        norm_max = np.nanmax(np.log10(bias_plus_var))
    else:
        norm_min = var.min()
        norm_max = bias_plus_var.max()

    color_h_crit = 'green'
    color_n_crit = 'blue'
    color_eq = 'red'
    color_contour_crit = 'black'
    
    d_style = {'bias':{'linestyle':'--','dashes':[3, 3]},
               'var':{'linestyle':'--','dashes':[1, 1, 1, 1]},
               'bias_plus_var':{'linestyle':'-'}}
    
    fig,axs = plt.subplots(2,2)
    for s in ['bias','var','bias_plus_var']:
        norm_cmap = ScalarMappable(Normalize(vmin=norm_min, vmax=norm_max))
        
        if str_variable.lower() == 'error':
            temp = np.log10(locals()[s])
        else:
            temp = locals()[s]

        fig_temp,ax_temp = plt.subplots(1,1)
        locals()[f'fig_{s}'] = fig_temp
        locals()[f'ax_{s}'] = ax_temp
        locals()[f'ax_{s}'].contourf(Ns,HSs,temp,levels=100,alpha=0.5,zorder=np.inf)
        locals()[f'ax_{s}'].loglog()
        plt.colorbar(norm_cmap,ax = locals()[f'ax_{s}'])
        plt.pause(0.01)
        locals()[f'ax_{s}'].plot(bias_h_crit[:,0],bias_h_crit[:,1],color=color_h_crit,linewidth=1,**d_style[s])
        locals()[f'ax_{s}'].plot(bias_n_crit[:,0],bias_n_crit[:,1],color=color_n_crit,linewidth=1,**d_style[s])
        locals()[f'ax_{s}'].plot(bias_eq[:,0],bias_eq[:,1],color=color_eq,linewidth=1,**d_style[s])
        locals()[f'ax_{s}'].plot(bias_eq_equidistant[:,0],bias_eq_equidistant[:,1],'.',color=color_eq)
        locals()[f'ax_{s}'].plot(bias_contour_crit[:,0],bias_contour_crit[:,1],color=color_contour_crit,linewidth=1,**d_style[s])
        locals()[f'ax_{s}'].plot(bias_contour_crit_equidistant[:,0],bias_contour_crit_equidistant[:,1],'.',color=color_contour_crit)

        axs[0,0].loglog(locals()[f'{s}_eq'][:,0],locals()[f'{s}_eq'][:,2],color = color_eq, **d_style[s])
        axs[0,1].loglog(locals()[f'{s}_contour_crit_equidistant'][:,1],locals()[f'{s}_contour_crit_equidistant'][:,2],color = color_contour_crit, **d_style[s])
        # axs[0,1].semilogy(locals()[f'{s}_contour_crit_equidistant'][:,2],color = color_contour_crit, **d_style[s])    
        axs[1,0].loglog(locals()[f'{s}_h_crit'][:,0],locals()[f'{s}_h_crit'][:,2],color = color_h_crit, **d_style[s])
        axs[1,1].loglog(locals()[f'{s}_n_crit'][:,1],locals()[f'{s}_n_crit'][:,2],color = color_n_crit, **d_style[s])
        axs[0,1].set_xlim(axs[1,1].get_xlim())

    for ax in axs.flatten():
        ax.set_ylim(10**norm_min,10**norm_max)


#%% Save

if bool_save:
    d_out = {}
    d_out['multiplier4maxBin'] = np.array([multiplier4maxBin])
    d_out['hs_crit'] = np.array([hs_crit])
    d_out['n_crit'] = np.array([n_crit])
    d_out['hs_crit'] = np.array([hs_crit])
    d_out['contour_crit'] = np.array([contour_crit])
    d_out['mu'] = np.array([mu])
    d_out['sigma'] = np.array([sigma])
    d_out['var_multiplier'] = np.array([var_multiplier])
    
    bias
    var
    bias_plus_var

    ss_out= ['HSs',
             'Ns',
             'bias',
             'var',
             'bias_plus_var',
             'bias_h_crit',
             'var_h_crit',
             'bias_plus_var_h_crit',
             'bias_n_crit',
             'var_n_crit',
             'bias_plus_var_n_crit',
             'bias_eq',
             'var_eq',
             'bias_plus_var_eq',
             'bias_eq_equidistant',
             'var_eq_equidistant',
             'bias_plus_var_eq_equidistant',
             'bias_contour_crit',
             'var_contour_crit',
             'bias_plus_var_contour_crit',
             'bias_contour_crit_equidistant',
             'var_contour_crit_equidistant',
             'bias_plus_var_contour_crit_equidistant',
             ]
    for s in ss_out:
        d_out[s] = locals()[s]
    
    try:
        np.save(save_path+save_name,d_out)
    except:
        np.save('./'+save_name,d_out)
        warnings.warn('Data saved on "./"')

