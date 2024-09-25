#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 15:36:15 2024

@author: vdoffini
"""

import numpy as np
from scipy.stats import norm,rv_histogram
from scipy.special import erf
#%% Data Lower Part
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
dist_z = rv_histogram(histogram=(z_pmf,bins_edges_0))
