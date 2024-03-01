# -*- coding: utf-8 -*-
"""
Plot Figure 4 (Pre-Industrial - Albedo)

@author: Daniel Gunning 
"""

import numpy as np
import proplot as pplt
from matplotlib.font_manager import FontProperties
import os
import pickle
import xarray as xr

output_path  = os.path.dirname(os.getcwd())
script_path  = os.path.dirname(output_path)
input_path   = script_path + '/input'

# change directory
os.chdir(script_path)

from utilities import *

#---------
# EBM data
#---------

# load pre-industrial results
#----------------------------

# load data
with open('output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
    pi_dict = pickle.load(f)
pi  = pi_dict['StateYear']
Var = pi_dict['Var']
INPUT = pi_dict['Input']
    
ebm = {}
noresm = {}
era5 = {}

# annual and zonal mean
#----------------------

# average
ebm["alpha_toa"] = pi["rsut"].mean(axis=1)/pi["I"].mean(axis=1)    # planetary
ebm["alpha_toa_global"] = round(global_pymean(ebm["alpha_toa"], Var), 2)

ebm["alpha_boa"] = pi["rsus"].mean(axis=1)/pi["rsds"].mean(axis=1) # surface
ebm["alpha_boa_global"] = round(global_pymean(ebm["alpha_boa"], Var), 2)

# land
ebm["alpha_toa_land"] = pi["rsut_land"].mean(axis=1)/pi["I"].mean(axis=1) # planetary
ebm["alpha_boa_land"] = pi["rsus_land"].mean(axis=1)/pi["rsds_land"].mean(axis=1) # surface

# ocean
ebm["alpha_toa_ocean"] = pi["rsut_ocean"].mean(axis=1)/pi["I"].mean(axis=1) # planetary
ebm["alpha_boa_ocean"] = pi["rsus_ocean"].mean(axis=1)/pi["rsds_ocean"].mean(axis=1) # surface

#-----------------
# Add NorESM2 data
#-----------------

# load annual data
noresm2_lat     = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=0)
noresm2_rsdt    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=5)
noresm2_rsut    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=6)
noresm2_rsds    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=7)
noresm2_rsus    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=8)

# interpolate data
noresm2_rsdt = np.interp(Var['lat'], noresm2_lat, noresm2_rsdt)
noresm2_rsut = np.interp(Var['lat'], noresm2_lat, noresm2_rsut)
noresm2_rsds = np.interp(Var['lat'], noresm2_lat, noresm2_rsds)
noresm2_rsus = np.interp(Var['lat'], noresm2_lat, noresm2_rsus)

# albedo
noresm2_alphat = noresm2_rsut / noresm2_rsdt
noresm2_alphas = noresm2_rsus / noresm2_rsds

# global_mean 
noresm2_alphat_global = global_mean2(noresm2_alphat, Var["lat"], Var["dlat"])
noresm2_alphas_global = global_mean2(noresm2_alphas, Var["lat"], Var["dlat"])

#---------------------------
# ERA5 1940-1970 climatology
#---------------------------

# load annual data
era5_lat     = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=0)
era5_rsdt    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=5)
era5_rsut    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=6)
era5_rsds    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=7)
era5_rsus    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=8)

# interpolate data
era5_rsdt = np.interp(Var['lat'], era5_lat, era5_rsdt)
era5_rsut = np.interp(Var['lat'], era5_lat, -era5_rsut)
era5_rsds = np.interp(Var['lat'], era5_lat, era5_rsds)
era5_rsus = np.interp(Var['lat'], era5_lat, era5_rsus)

# albedo
era5_alphat = era5_rsut / era5_rsdt
era5_alphas = era5_rsus / era5_rsds

# global_mean 
era5_alphat_global = global_mean2(era5_alphat, Var["lat"], Var["dlat"])
era5_alphas_global = global_mean2(era5_alphas, Var["lat"], Var["dlat"])

#--------------
# Plot figure 3 -- v1
#--------------

# constants
#----------

ebm_color = "black"
ebm_lw    = 1.
ebm_ls    = "-"

noresm_color = "blue9"
noresm_lw    = 1.
noresm_ls    = "-"

era5_color = "red9"
era5_lw    = 1.
era5_ls    = "-"

ebm_minus_noresm_color = "blue9"
ebm_minus_noresm_lw    = 1.
ebm_minus_noresm_ls    = "-."

ebm_minus_era5_color = "red9"
ebm_minus_era5_lw    = 1.
ebm_minus_era5_ls    = "-."

legend_fs = 7.
legend_fns = "bold"


# formating
#----------

# shape
shape = [  
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4],]

# figure and axes
fig, axs = pplt.subplots(shape, figsize = (8,4), sharey = False, sharex = False, grid = False)


# fonts
axs.format(ticklabelsize=7, ticklabelweight='normal', 
           ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', 
            titlesize=8, titleweight='normal',)

# x-axis
axs.format(xlim = (-90, 90),
           xlocator = np.arange(-90, 120, 30),
           xminorlocator = np.arange(-90, 100, 10),)


# format top subplots
for i in np.arange(0,1+1):
    
    # y-axis
    axs[i].format(ylim = (0, 1), yminorlocator=[], ylocator = np.arange(0, 1+0.1, 0.1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")

    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)
    
# format bottom subplots  
for i in np.arange(2,3+1):
    
    # y-axis
    axs[i].format(ylim = (-0.2, 0.2), yminorlocator=[], ylocator = np.arange(-0.2, 0.2+0.1, 0.1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")

    # x-axis
    axs[i].format(xlabel = "Latitude", xformatter='deglat')

# titles
axs[0].format(title = r'(a) PI Planetary Albedo',titleloc = 'left')
axs[1].format(title = r'(b) PI Surface Albedo',titleloc = 'left')
axs[2].format(title = r'(c) Difference in Planetary Albedo', titleloc = "left")
axs[3].format(title = r'(d) Difference in Surface Albedo', titleloc = "left")

    
# plot planetary albedo
#----------------------

# lines
ebm_toa_line    = axs[0].plot(Var["lat"], ebm["alpha_toa"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls)
noresm_toa_line = axs[0].plot(Var["lat"], noresm2_alphat, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls)
era5_toa_line   = axs[0].plot(Var["lat"], era5_alphat, color = era5_color, lw = era5_lw, linestyle = era5_ls)

# legend
axs[0].legend(handles = [ebm_toa_line, noresm_toa_line, era5_toa_line],
              labels  = ["ZEMBA", "NorESM2", "ERA5"], frameon = False,
              loc = "c", bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})

# plot difference in planetary albedo
#------------------------------------

# lines
ebm_minus_noresm_toa_line = axs[2].plot(Var["lat"], ebm["alpha_toa"] - noresm2_alphat, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_toa_line   = axs[2].plot(Var["lat"], ebm["alpha_toa"] - era5_alphat, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# legend
axs[2].legend(handles = [ebm_minus_noresm_toa_line, ebm_minus_era5_toa_line],
              labels  = ["ZEMBA - NorESM2", "ZEMBA - ERA5"],
              ncols = 2, loc = "uc", frameon = False, 
              bbox_to_anchor=(0.5, 0.2), prop={'size':legend_fs}) 



# plot surface albedo
#--------------------

# lines
ebm_boa_line    = axs[1].plot(Var["lat"], ebm["alpha_boa"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
noresm_boa_line = axs[1].plot(Var["lat"], noresm2_alphas, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["noresm"])
era5_boa_line   = axs[1].plot(Var["lat"], era5_alphas, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])

# legend
axs[1].legend(handles = [ebm_boa_line, noresm_boa_line, era5_boa_line],
              labels  = ["ZEMBA", "NorESM2", "ERA5"], frameon = False,
              loc = "c", bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})


# plot differences in surface albedo
#-------------------------------------
  
# lines
ebm_minus_noresm_boa_line = axs[3].plot(Var["lat"], ebm["alpha_boa"] - noresm2_alphas, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_boa_line   = axs[3].plot(Var["lat"], ebm["alpha_boa"] - era5_alphas, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

axs[3].legend(handles = [ebm_minus_noresm_boa_line, ebm_minus_era5_boa_line],
              labels  = ["ZEMBA - NorESM2", "ZEMBA - ERA5"],
              ncols = 2, loc = "uc", frameon = False, 
              bbox_to_anchor=(0.5, 0.2), prop={'size': legend_fs}) 


fig.save(os.getcwd() +"/output/plots/f04.png", dpi = 400)
fig.save(os.getcwd() +"/output/plots/f04.pdf", dpi = 400)
