# -*- coding: utf-8 -*-
"""
Plot Figure (2x CO2 and 2% Inso)

@author: Daniel Gunning 
"""

import numpy as np
import proplot as pplt
from matplotlib.font_manager import FontProperties
import os
import pickle
import xarray as xr
import matplotlib.colors as colors

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
with open(os.getcwd()+'\\output\\equilibrium\\pi_moist_res5.0.pkl', 'rb') as f:
    pi = pickle.load(f)['StateYear']
with open(os.getcwd()+'\\output\\equilibrium\\pi_moist_res5.0.pkl', 'rb') as f:
    Var = pickle.load(f)['Var']
    
# load 2x CO2 results
#--------------------

# load data
with open(os.getcwd()+'\\output\\equilibrium\\2xCO2_moist_res5.0.pkl', 'rb') as f:
    dco2 = pickle.load(f)['StateYear']
    
# load 2% Inso results
#---------------------

# load data
with open(os.getcwd()+'\\output\\equilibrium\\2%Inso_moist_res5.0.pkl', 'rb') as f:
    dinso = pickle.load(f)['StateYear']
    
    
ebm = {}
   
    
# annual mean temperature
#------------------------

# pi
ebm["pi_tas"]          = pi["Ta"] 
ebm["pi_tas_global"]   = round(global_pymean(ebm["pi_tas"].mean(axis=1), Var), 1)

# 2x CO2
ebm["2xCO2_tas"]          = dco2["Ta"] 
ebm["2xCO2_tas_global"]   = global_pymean(ebm["2xCO2_tas"].mean(axis=1), Var)

# 2% Inso
ebm["2%Inso_tas"]          = dinso["Ta"] 
ebm["2%Inso_tas_global"]   = global_pymean(ebm["2%Inso_tas"].mean(axis=1), Var)


#------------------------------------------------------------------------------
# Plot -- Version 1
#------------------------------------------------------------------------------

# formating
#----------

# figure shape
shape = [  
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4],
        [3, 3, 3, 4, 4, 4]]


# figure and axes
fig, axs = pplt.subplots(shape, figsize = (8,4), 
                         
                         sharey = False, sharex = False, 
                         
                         hspace=1.8, grid = False)

# variables
x2CO2 = ebm["2xCO2_tas"].mean(axis=1) - ebm["pi_tas"].mean(axis=1)
Inso2 = ebm["2%Inso_tas"].mean(axis=1) - ebm["pi_tas"].mean(axis=1)

# normalized variables
x2CO2_norm = (ebm["2xCO2_tas"].mean(axis=1) - ebm["pi_tas"].mean(axis=1))/(ebm["2xCO2_tas_global"]-ebm["pi_tas_global"])
Inso2_norm = (ebm["2%Inso_tas"].mean(axis=1) - ebm["pi_tas"].mean(axis=1))/(ebm["2%Inso_tas_global"]-ebm["pi_tas_global"])

# format contour plots        
axs[1].format(ylim = (-90, 90), yformatter = 'deglat', ylocator = np.arange(-90, 120, 30), yminorlocator = np.arange(-90, 100, 10))
axs[1].format(xlim = (1, 365), xlocator = np.arange(15, 365, 30), xminorlocator = [],
           xticklabels = ["J", "F", "M", 
                          "A", "M", "J", 
                          "J", "A", "S", 
                          "O", "N", "D"])
axs[3].format(ylim = (-90, 90), yformatter = 'deglat', ylocator = np.arange(-90, 120, 30), yminorlocator = np.arange(-90, 100, 10))
axs[3].format(xlim = (1, 365), xlocator = np.arange(15, 365, 30), xminorlocator = [],
           xticklabels = ["J", "F", "M", 
                          "A", "M", "J", 
                          "J", "A", "S", 
                          "O", "N", "D"],
           xlabel = 'Month')

# format line plots
axs[0].format(xlim = (-90, 90), xformatter = 'deglat', xlocator = np.arange(-90, 120, 30), xminorlocator = np.arange(-90, 100, 10))
axs0=axs[0].twinx()
axs[0].format(ylim = (0, x2CO2.max()*1.05), ylocator = np.arange(0, 13+1, 1), yminorlocator = [])
axs0.format(ylim = (0, x2CO2_norm.max()*1.05), ylocator = np.arange(0, 5+1, 1), yminorlocator = [])

axs[2].format(xlim = (-90, 90), xformatter = 'deglat', xlocator = np.arange(-90, 120, 30), xminorlocator = np.arange(-90, 100, 10), xlabel = 'Latitude')
axs2=axs[2].twinx()
axs[2].format(ylim = (0, Inso2.max()*1.1), ylocator = np.arange(0, 13+1, 1), yminorlocator = [])
axs2.format(ylim = (0, Inso2_norm.max()*1.1),ylocator = np.arange(0, 5+1, 1), yminorlocator = [])

axs[2].format(xformatter='deglat')

# fonts
axs.format(ticklabelsize=7, ticklabelweight='normal', 
            ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', 
            titlesize=8, titleweight='normal',)
axs0.format(ticklabelsize=7, ticklabelweight='normal', 
            ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', 
            titlesize=8, titleweight='normal',)
axs2.format(ticklabelsize=7, ticklabelweight='normal', 
            ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', 
            titlesize=8, titleweight='normal',)

# hide x-axis
for i in np.arange(0,1+1):
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False) 


# mesh grid
days, lat = np.meshgrid(Var["ndays"], Var["lat"])
 
# v max and min values
vmax = 25
vmin = -5

# titles
axs[0].format(title = r'(a) 2xCO$_{2}$ - PI zonal and annual mean temperature ($^{\circ}$C)', titleloc = 'left')
axs[1].format(title = r'(b) 2xCO$_{2}$ - PI zonal mean temperature ($^{\circ}$C)', titleloc = 'left')
axs[2].format(title = r'(c) S$_{o}$+2$\%$ - PI zonal and annual mean temperature ($^{\circ}$C)', titleloc = 'left')
axs[3].format(title = r'(d) S$_{o}$+2$\%$ - PI zonal mean temperature ($^{\circ}$C)', titleloc = 'left')

# colormap
cmap = 'vik'

# plot 2x CO2 - PI
#-----------------

# 2x CO2 annual
pi_dco2_line  = axs[0].plot(Var["lat"], ebm["2xCO2_tas"].mean(axis=1) - ebm["pi_tas"].mean(axis=1), color = 'black', lw = 2, linestyle = '-')
pi_dco2_norm_line = axs0.plot(Var["lat"], (ebm["2xCO2_tas"].mean(axis=1) - ebm["pi_tas"].mean(axis=1))/(ebm["2xCO2_tas_global"]-ebm["pi_tas_global"]), color = 'black', lw = 2, linestyle = '-')

# 2% INso
pi_dinso_line = axs[2].plot(Var["lat"], ebm["2%Inso_tas"].mean(axis=1) - ebm["pi_tas"].mean(axis=1), color = 'black', lw = 2, linestyle = ':')
pi_dinso_norm_line = axs2.plot(Var["lat"], (ebm["2%Inso_tas"].mean(axis=1) - ebm["pi_tas"].mean(axis=1))/(ebm["2%Inso_tas_global"]-ebm["pi_tas_global"]), color = 'black', lw = 2, linestyle = '-')

# seasonal cycle contour plots
cnt1 = axs[1].contourf(days, lat, 
                 
                 ebm["2xCO2_tas"] - ebm["pi_tas"],

                 vmin = vmin, vmax = vmax,
                 
                 norm=colors.CenteredNorm(),
                 
                 cmap=cmap)

cnt2 = axs[3].contourf(days, lat, 
                 
                 ebm["2%Inso_tas"] - ebm["pi_tas"],
                 
                 vmin = vmin, vmax = vmax,
                 
                 norm=colors.CenteredNorm(),
                 
                 cmap=cmap)

# colorbar
fig.colorbar(cnt1, loc='r', ticklabelsize=6)


fig.save(os.getcwd()+"/output/plots/f08.png", dpi = 400)
fig.save(os.getcwd()+"/output/plots/f08.pdf", dpi = 400)

#------------------------------------------------
# Print difference in global mean air temperature
#------------------------------------------------

# 2x CO2
print('##################')
print('Global mean Ta increase for CO2 doubling: ', round( ebm["2xCO2_tas_global"] - ebm["pi_tas_global"], 2))
print('##################\n')

print('##################')
print('Global mean Ta increase for +2% insolation: ', round( ebm["2%Inso_tas_global"] - ebm["pi_tas_global"], 2))  
print('##################\n')

#-----------------------------------------------------------
# Print Arctic and Antarctic warming relative to global mean
#-----------------------------------------------------------

# 2% + Insolation
#----------------

# Arctic 

inso2_arctic = np.average(ebm["2%Inso_tas"][30:].mean(axis=1) - ebm["pi_tas"][30:].mean(axis=1), weights=Var['sarea'][30:])

# Antarctic warming

inso2_antarctic = np.average(ebm["2%Inso_tas"][:6].mean(axis=1) - ebm["pi_tas"][:6].mean(axis=1), weights=Var['sarea'][:6])



# 2x CO2
#-------

# Arctic 

x2CO2_arctic = np.average(ebm["2xCO2_tas"][30:].mean(axis=1) - ebm["pi_tas"][30:].mean(axis=1), weights=Var['sarea'][30:])

# Antarctic warming

x2CO2_antarctic = np.average(ebm["2xCO2_tas"][:6].mean(axis=1) - ebm["pi_tas"][:6].mean(axis=1), weights=Var['sarea'][:6])


# Print Arctic and Antarctic warming - normalized to global mean warming
#-----------------------------------------------------------------------

# 2x CO2

print('##################')
print('2x CO2 Normalized Arctic warming: ', round( x2CO2_arctic / (ebm["2xCO2_tas_global"] - ebm["pi_tas_global"]), 2))
print('2x CO2 Normalized Antarcitc warming: ', round( x2CO2_antarctic / (ebm["2xCO2_tas_global"] - ebm["pi_tas_global"]), 2))
print('##################\n')


# 2% Inso

print('##################')
print('2% Inso Normalized Arctic warming: ', round( inso2_arctic / (ebm["2%Inso_tas_global"] - ebm["pi_tas_global"]), 2))
print('2% Inso Normalized Antarcitc warming: ', round( inso2_antarctic / (ebm["2%Inso_tas_global"] - ebm["pi_tas_global"]), 2))
print('##################\n')



