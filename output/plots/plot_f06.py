# -*- coding: utf-8 -*-
"""
Plot Figure 5 (Pre-Industrial - Results - Radiation)

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
ebm_rsdtnet = pi['rsdtnet'].mean(axis=1)  # TOA net shortwave
ebm_rlut    = pi['rlut'].mean(axis=1)     # TOA outgoing longwave
ebm_rtnet   = pi['rtnet'].mean(axis=1)    # TOA net total radiation
ebm_rsdsnet = pi['rsdsnet'].mean(axis=1)  # BOA net shortwave
ebm_rlus    = -pi['rldsnet'].mean(axis=1) # BOA net outgoing longwave
ebm_rsnet   = pi['rsnet'].mean(axis=1)    # BOA net total radiation

ebm_rtnet_global   = global_mean2(ebm_rtnet, Var["lat"], Var["dlat"])    # global mean BOA net total radiation
ebm_rsnet_global   = global_mean2(ebm_rsnet, Var["lat"], Var["dlat"])    # global mean BOA net total radiation

#-----------------
# Add NorESM2 data
#-----------------

# load annual data
noresm2_lat     = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=0)
noresm2_rsdt    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=5)
noresm2_rsut    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=6)
noresm2_rsds    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=7)
noresm2_rsus    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=8)
noresm2_rlut    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=9)
noresm2_rlds    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=10)
noresm2_rlus    = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=11)

# interpolate data
noresm2_rsdt = np.interp(Var['lat'], noresm2_lat, noresm2_rsdt)
noresm2_rsut = np.interp(Var['lat'], noresm2_lat, noresm2_rsut)
noresm2_rsds = np.interp(Var['lat'], noresm2_lat, noresm2_rsds)
noresm2_rsus = np.interp(Var['lat'], noresm2_lat, noresm2_rsus)
noresm2_rlut = np.interp(Var['lat'], noresm2_lat, noresm2_rlut)
noresm2_rlds = np.interp(Var['lat'], noresm2_lat, noresm2_rlds)
noresm2_rlus = np.interp(Var['lat'], noresm2_lat, noresm2_rlus)

# net shortwave
noresm2_rsdtnet = noresm2_rsdt - noresm2_rsut
noresm2_rsdsnet = noresm2_rsds - noresm2_rsus

# net longwave
noresm2_rldsnet = noresm2_rlds - noresm2_rlus

# net total
noresm2_rtnet = noresm2_rsdtnet - noresm2_rlut
noresm2_rsnet = noresm2_rsdsnet + noresm2_rldsnet


#--------------
# Add ERA5 data
#--------------

# load annual data
era5_lat     = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=0)
era5_rsdt    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=5)
era5_rsut    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=6)
era5_rsds    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=7)
era5_rsus    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=8)
era5_rlut    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=9)
era5_rlds    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=10)
era5_rlus    = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=11)

# interpolate data
era5_rsdt = np.interp(Var['lat'], era5_lat, era5_rsdt)
era5_rsut = np.interp(Var['lat'], era5_lat, -era5_rsut)
era5_rsds = np.interp(Var['lat'], era5_lat, era5_rsds)
era5_rsus = np.interp(Var['lat'], era5_lat, era5_rsus)
era5_rlut = np.interp(Var['lat'], era5_lat, -era5_rlut)
era5_rlds = np.interp(Var['lat'], era5_lat, era5_rlds)
era5_rlus = np.interp(Var['lat'], era5_lat, era5_rlus)

# net shortwave
era5_rsdtnet = era5_rsdt - era5_rsut
era5_rsdsnet = era5_rsds - era5_rsus

# net longwave
era5_rldsnet = era5_rlds - era5_rlus

# net total
era5_rtnet = era5_rsdtnet - era5_rlut
era5_rsnet = era5_rsdsnet + era5_rldsnet


#------------
# Plot figure 
#------------

# constants
#----------

ebm_color = "black"
ebm_lw    = 1

noresm_color = "blue9"
noresm_lw    = 1

era5_color = "red9"
era5_lw    = 1

ebm_minus_noresm_color = "blue9"
ebm_minus_noresm_lw    = 1
ebm_minus_noresm_ls    = "-."

ebm_minus_era5_color = "red9"
ebm_minus_era5_lw    = 1
ebm_minus_era5_ls    = "-."

ebm_ls = "-"
noresm_ls = ":"
era5_ls = "-."

asr_ls = "-"
olw_ls = ":"
net_ls = "-."

legend_fs=7.


# formating
#----------

# shape
shape = [  
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [3, 3, 3, 3, 4, 4, 4, 4],]

# figure and axes
fig, axs = pplt.subplots(shape, figsize = (8,4), sharey = False, sharex = False, grid = False)


# fonts
axs.format(ticklabelsize=8., ticklabelweight='normal', 
           ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', 
            titlesize=7, titleweight='normal',)

# x-axis
locatorx      = np.arange(-90, 120, 30)
minorlocatorx = np.arange(-90, 100, 10)
axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))

# format top left subplot
for i in np.arange(0,1):
    
    # y-axis
    axs[i].format(ylim = (-150, 350), yminorlocator=[], ylocator = np.arange(-150, 350+50, 50))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)

# format top left subplot
for i in np.arange(1,2):
    
    # y-axis
    axs[i].format(ylim = (-100, 250), yminorlocator=[], ylocator = np.arange(-100, 250+50, 50))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)
    
# format bottom subplots
for i in np.arange(2,3+1):
    
    # y-axis
    axs[i].format(ylim = (-50, 50), yminorlocator=[], ylocator = np.arange(-50, 50+25, 25))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].format(xlabel = "Latitude", xformatter='deglat')
    
    

# titles
axs[0].format(title = r'(a) Radiative fluxes ($W/m^{2}$) at the top of the atmosphere', titleloc = 'left')
axs[1].format(title = r'(b) Radiative fluxes ($W/m^{2}$) at the surface', titleloc = 'left')
axs[2].format(title = r'(c) Difference in net radiative fluxes ($W/m^{2}$) at the top of the atmosphere', titleloc = 'left')
axs[3].format(title = r'(d) Difference in net radiative fluxes ($W/m^{2}$) at the surface', titleloc = 'left')



    
# plot TOA radiative fluxes
#--------------------------

ebm_toa_asr = axs[0].plot(Var["lat"], ebm_rsdtnet, color = ebm_color, lw = ebm_lw, ls = asr_ls) # Net shortwave at TOA
ebm_toa_olw = axs[0].plot(Var["lat"], ebm_rlut, color = ebm_color, lw = ebm_lw, ls = olw_ls)   # Outgoing longwave radiation at TOA
ebm_toa_net = axs[0].plot(Var["lat"], ebm_rtnet, color = ebm_color, lw = ebm_lw, ls = net_ls)   # Net radiation at TOA

noresm_toa_asr = axs[0,0].plot(Var["lat"], noresm2_rsdtnet, color = noresm_color, lw = noresm_lw, ls = asr_ls)  # Net shortwave at TOA
noresm_toa_olw = axs[0,0].plot(Var["lat"], noresm2_rlut, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
noresm_toa_net = axs[0,0].plot(Var["lat"], noresm2_rtnet, color = noresm_color, lw = noresm_lw, ls = net_ls)     # Net radiation at TOA

era5_toa_asr = axs[0,0].plot(Var["lat"], era5_rsdtnet, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at TOA
era5_toa_olw = axs[0,0].plot(Var["lat"], era5_rlut, color = era5_color, lw = era5_lw, ls = olw_ls)      # Outgoing longwave radiation at TOA
era5_toa_net = axs[0,0].plot(Var["lat"], era5_rtnet, color = era5_color, lw = era5_lw, ls = net_ls)     # Net radiation at TOA


# legend
axs[0].legend(handles = [ebm_toa_asr, noresm_toa_asr, era5_toa_asr],
              labels  = ["ZEMBA (ASR)", "NorESM2 (ASR)", "ERA5 (ASR)"], frameon = False,
              loc = "ul", bbox_to_anchor=(0.02, 0.97), ncols = 1, prop={'size':legend_fs})  

axs[0].legend(handles = [ebm_toa_olw, noresm_toa_olw, era5_toa_olw],
              labels  = ["ZEMBA (OLR)", "NorESM2 (OLR)", "ERA5 (OLR)"], frameon = False,
              loc = "c", bbox_to_anchor=(0.5, 0.6), ncols = 1, prop={'size':legend_fs})   

axs[0].legend(handles = [ebm_toa_net, noresm_toa_net, era5_toa_net],
              labels  = ["ZEMBA (NET)", "NorESM2 (NET)", "ERA5 (NET)"], frameon = False,
              loc = "lc", bbox_to_anchor=(0.5, 0.1), ncols = 1, prop={'size':legend_fs})  

# plot difference in net radiative fluxes at TOA
#-----------------------------------------------

ebm_minus_noresm_toa_line = axs[2].plot(Var["lat"],  ebm_rtnet - noresm2_rtnet, color = ebm_minus_noresm_color, ls = ebm_minus_noresm_ls, lw = ebm_minus_noresm_lw)
ebm_minus_era5_toa_line = axs[2].plot(Var["lat"],  ebm_rtnet - era5_rtnet, color = ebm_minus_era5_color, ls = ebm_minus_era5_ls, lw = ebm_minus_era5_lw)

# legend
axs[2].legend(handles = [ebm_minus_noresm_toa_line, ebm_minus_era5_toa_line],
              labels  = ["ZEMBA − NorESM2 (NET)", "ZEMBA − ERA5 (NET)"], frameon = False,
              loc = "uc", bbox_to_anchor=(0.5, 0.98), ncols = 2, prop={'size':legend_fs}) 

# plot SFC radiative fluxes
#--------------------------

ebm_boa_asr = axs[1].plot(Var["lat"], ebm_rsdsnet, color = ebm_color, lw = ebm_lw, ls = asr_ls) # Net shortwave at boa
ebm_boa_olw = axs[1].plot(Var["lat"], ebm_rlus, color = ebm_color, lw = ebm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
ebm_boa_net = axs[1].plot(Var["lat"], ebm_rsnet, color = ebm_color, lw = ebm_lw, ls = net_ls)   # Net radiation at boa

noresm_boa_asr = axs[1].plot(Var["lat"], noresm2_rsdsnet, color = noresm_color, lw = noresm_lw, ls = asr_ls) # Net shortwave at boa
noresm_boa_olw = axs[1].plot(Var["lat"], noresm2_rlus-noresm2_rlds, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
noresm_boa_net = axs[1].plot(Var["lat"], noresm2_rsnet, color = noresm_color, lw = noresm_lw, ls = net_ls)    # Net radiation at boa

era5_boa_asr = axs[1].plot(Var["lat"], era5_rsdsnet, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at boa
era5_boa_olw = axs[1].plot(Var["lat"], era5_rlus-era5_rlds, color = era5_color, lw = era5_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
era5_boa_net = axs[1].plot(Var["lat"], era5_rsnet, color = era5_color, lw = era5_lw, ls = net_ls)   # Net radiation at boa

# legend
axs[1].legend(handles = [ebm_boa_asr, noresm_boa_asr, era5_boa_asr],
              labels  = ["ZEMBA (ASR)", "NorESM2 (ASR)", "ERA5 (ASR)"], frameon = False,
              loc = "ul", bbox_to_anchor=(0.02, 0.97), ncols = 1, prop={'size':legend_fs})  

axs[1].legend(handles = [ebm_boa_olw, noresm_boa_olw, era5_boa_olw],
              labels  = ["ZEMBA (OLR)", "NorESM2 (OLR)", "ERA5 (OLR)"], frameon = False,
              loc = "lc", bbox_to_anchor=(0.5, 0.1), ncols = 1, prop={'size':legend_fs}) 


axs[1].legend(handles = [ebm_boa_net, noresm_boa_net, era5_boa_net],
              labels  = ["ZEMBA (NET)", "NorESM2 (NET)", "ERA5 (NET)"], frameon = False,
              loc = "c", bbox_to_anchor=(0.5, 0.57), ncols = 1, prop={'size':legend_fs}) 


# plot difference in net radiative fluxes at surface
#----------------------------------------------------

ebm_minus_noresm_sfc_line = axs[3].plot(Var["lat"],  ebm_rsnet - noresm2_rsnet, color = ebm_minus_noresm_color, ls = ebm_minus_noresm_ls, lw = ebm_minus_noresm_lw)
ebm_minus_era5_sfc_line = axs[3].plot(Var["lat"],  ebm_rsnet - era5_rsnet, color = ebm_minus_era5_color, ls = ebm_minus_era5_ls, lw = ebm_minus_era5_lw)

# legend
axs[3].legend(handles = [ebm_minus_noresm_sfc_line, ebm_minus_era5_sfc_line],
              labels  = ["ZEMBA − NorESM2 (NET)", "ZEMBA − ERA5 (NET)"], frameon = False,
              loc = "uc", bbox_to_anchor=(0.5, 0.98), ncols = 2, prop={'size':legend_fs})  
 
fig.save(os.getcwd()+"/output/plots/f06.png", dpi = 400)
fig.save(os.getcwd()+"/output/plots/f06.pdf", dpi = 400)


