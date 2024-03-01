# -*- coding: utf-8 -*-
"""
Plot Figure 2 (Pre-Industrial - Air Temperatures)

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
    
# annual mean temperature
#------------------------

# atmosphere
ebm["tas_annual"]          = pi["Ta"].mean(axis=1) - Var["K"] - (Var["lr"] * Var["mean_height"])
ebm["tas_global_mean"]     = round(global_pymean(ebm["tas_annual"], Var), 2)

# land
ebm["tl_annual"]          = pi["Tl"].mean(axis=1) - Var["K"] - (Var["lr"] * INPUT["land_height"])

# surface ocean
ebm["tos_annual"]         = pi["Tos"].mean(axis=1) - Var["K"]

# june-july-august
#-----------------

# atmosphere
ebm["tas_jja"] = pi["Ta"][:,151:242+1].mean(axis=1) - Var["K"] - (Var["lr"] * Var["mean_height"])

# land
ebm["tl_jja"]  = pi["Tl"][:,151:242+1].mean(axis=1) - Var["K"] - (Var["lr"] * INPUT["land_height"])

# surface ocean
ebm["tos_jja"] = pi["Tos"][:,151:242+1].mean(axis=1) - Var["K"] 

# december-january-february
#--------------------------

# atmosphere
ebm["tas_djf"] = ((np.append(pi["Ta"][:,0:58+1], pi["Ta"][:,334:], axis = 1)).mean(axis=1)) - Var["K"] - (Var["lr"] * Var["mean_height"])

# land
ebm["tl_djf"] = ((np.append(pi["Tl"][:,0:58+1], pi["Tl"][:,334:], axis = 1)).mean(axis=1)) - Var["K"] - (Var["lr"] * INPUT["land_height"])

# surface ocean
ebm["tos_djf"] = ((np.append(pi["Tos"][:,0:58+1], pi["Tos"][:,334:], axis = 1)).mean(axis=1)) - Var["K"]

#-------------
# NorESM2 data
#-------------

# load data
noresm2_lat             = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=0)
noresm2_tas_annual      = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=1)
noresm2_tas_monthly_nc  = np.loadtxt(script_path+'/other_data/noresm2/noresm2_t2m_monthly.txt', skiprows=5, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])

# interpolate data
noresm2_tas_annual  = np.interp(Var['lat'], noresm2_lat, noresm2_tas_annual) - Var['K']
noresm2_tas_monthly = np.zeros((Var['lat'].size, 12)) 
for i in np.arange(0,12):
    noresm2_tas_monthly[:,i]=np.interp(Var['lat'], noresm2_lat, noresm2_tas_monthly_nc[:,i])- Var['K']
    
# global_mean 
noresm2_tas_global   = global_mean2(noresm2_tas_annual, Var["lat"], Var["dlat"])

# JJA mean
noresm2_tas_jja = ( ((noresm2_tas_monthly[:,5]*30.) +
                 (noresm2_tas_monthly[:,6]*31.) +
                 (noresm2_tas_monthly[:,7]*31.))
                
                /
                (30.+31.+31.) 
            
                ) 

# DJF mean
noresm2_tas_djf = ( ((noresm2_tas_monthly[:,11]*31.) +
                 (noresm2_tas_monthly[:,0]*31.) +
                 (noresm2_tas_monthly[:,1]*28.))
                
                /
                (31.+31.+28.) 
            
                ) 

#----------
# ERA5 data
#----------

# load data
era5_lat             = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=0)
era5_tas_annual      = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=1)
era5_tas_monthly_nc  = np.loadtxt(script_path+'/other_data/era5/era5_t2m_monthly.txt', skiprows=5, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])

# interpolate data
era5_tas_annual  = np.interp(Var['lat'], era5_lat, era5_tas_annual) - Var['K']
era5_tas_monthly = np.zeros((Var['lat'].size, 12)) 
for i in np.arange(0,12):
    era5_tas_monthly[:,i]=np.interp(Var['lat'], era5_lat, era5_tas_monthly_nc[:,i]) - Var['K']
    
# global_mean 
era5_tas_global   = global_mean2(era5_tas_annual, Var["lat"], Var["dlat"])

# JJA mean
era5_tas_jja = ( ((era5_tas_monthly[:,5]*30.) +
                 (era5_tas_monthly[:,6]*31.) +
                 (era5_tas_monthly[:,7]*31.))
                
                /
                (30.+31.+31.) 
            
                ) 

# DJF mean
era5_tas_djf = ( ((era5_tas_monthly[:,11]*31.) +
                 (era5_tas_monthly[:,0]*31.) +
                 (era5_tas_monthly[:,1]*28.))
                
                /
                (31.+31.+28.) 
            
                ) 


#------------------------------------------------------------------------------
# Plot -- Version 1
#------------------------------------------------------------------------------

# colors and line widths
#-----------------------

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
ebm_minus_era5_ls    = ":"

# legend font size
legend_fs = 7.
legend_fnt = "bold"

# formating
#----------

# shape
shape = [  
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
        [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],]

# figure and axes
fig, axs = pplt.subplots(shape, figsize = (10,4), sharey = False, sharex = False, grid = False)


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
for i in np.arange(0,2+1):
    
    # y-axis
    axs[i].format(ylim = (-60, 30), yminorlocator=np.arange(-60, 30, 5), ylocator = np.arange(-60, 30+10, 10))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")

    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)
    
# format bottom subplots  
for i in np.arange(3,5+1):
    
    # y-axis
    axs[i].format(ylim = (-8, 8), yminorlocator=[], ylocator = np.arange(-8, 8+4, 4))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")

    # x-axis
    axs[i].format(xlabel = "Latitude", xformatter='deglat')

# titles
axs[0].format(title = r'(a) PI Annual Zonal Temperatures ($^{\circ}$C)',titleloc = 'left')
axs[1].format(title = r'(b) PI DJF Zonal Temperatures ($^{\circ}$C)',titleloc = 'left')
axs[2].format(title = r'(c) PI JJA Zonal Temperatures ($^{\circ}$C)', titleloc = "left")
axs[3].format(title = r'(d) Difference in Annual Zonal Temperatures ($^{\circ}$C)', titleloc = "left")
axs[4].format(title = r'(e) Difference in DJF Zonal Temperatures ($^{\circ}$C)', titleloc = "left")
axs[5].format(title = r'(f) Difference in JJA Zonal Temperatures ($^{\circ}$C)', titleloc = "left")
 

# plot annual mean temperatures
#------------------------------

# lines
ebm_annual_line    = axs[0].plot(Var["lat"], ebm["tas_annual"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls)
era5_annual_line   = axs[0].plot(Var["lat"], era5_tas_annual, color = era5_color, lw = era5_lw, linestyle = era5_ls)
noresm_annual_line = axs[0].plot(Var["lat"], noresm2_tas_annual, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls)

# legends
axs[0].legend(handles = [ebm_annual_line, noresm_annual_line, era5_annual_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})

# plot differences
ebm_minus_noresm_annual_line = axs[3].plot(Var["lat"], ebm["tas_annual"] - noresm2_tas_annual, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_annual_line   = axs[3].plot(Var["lat"], ebm["tas_annual"] - era5_tas_annual, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# difference legends
axs[3].legend(handles = [ebm_minus_noresm_annual_line, ebm_minus_era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})


# plot december - january - february
#-----------------------------------


# lines
ebm_djf_line    = axs[1].plot(Var["lat"], ebm["tas_djf"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
noresm_djf_line = axs[1].plot(Var["lat"], noresm2_tas_djf, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
era5_djf_line   = axs[1].plot(Var["lat"], era5_tas_djf, color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# legend
axs[1].legend(handles = [ebm_djf_line, noresm_djf_line, era5_djf_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})

# plot difference
ebm_minus_noresm_djf_line = axs[4].plot(Var["lat"], ebm["tas_djf"] - noresm2_tas_djf, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_djf_line   = axs[4].plot(Var["lat"], ebm["tas_djf"] - era5_tas_djf, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# differences legend
axs[4].legend(handles = [ebm_minus_noresm_djf_line, ebm_minus_era5_djf_line, era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})


# plot june - july - august
#--------------------------


# lines
ebm_jja_line    = axs[2].plot(Var["lat"], ebm["tas_jja"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
noresm_jja_line = axs[2].plot(Var["lat"], noresm2_tas_jja, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
era5_jja_line   = axs[2].plot(Var["lat"], era5_tas_jja, color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# legend
axs[2].legend(handles = [ebm_jja_line, noresm_jja_line, era5_jja_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})


# plot difference
ebm_minus_noresm_jja_line = axs[5].plot(Var["lat"], ebm["tas_jja"] - noresm2_tas_jja, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_jja_line   = axs[5].plot(Var["lat"], ebm["tas_jja"] - era5_tas_jja, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# differences legend
leg = axs[5].legend(handles = [ebm_minus_noresm_jja_line, ebm_minus_era5_jja_line, era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})


fig.save(os.getcwd()+"/output/plots/f02.png", dpi = 400)
fig.save(os.getcwd()+"/output/plots/f02.pdf", dpi = 400)

