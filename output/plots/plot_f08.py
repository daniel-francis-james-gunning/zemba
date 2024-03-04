# -*- coding: utf-8 -*-
"""
plot figure 8 - results sections - 2xCO2 & +2% increase in solar insolation

@author: Daniel Gunning 
"""

import numpy as np
import proplot as pplt
import os
import pickle
import xarray as xr
import matplotlib.colors as colors

# paths
output_path  = os.path.dirname(os.getcwd())
script_path  = os.path.dirname(output_path)
input_path   = script_path + '/input'

os.chdir(script_path)
from utilities import *

#------------------------------
# load zemba pre-industrial sim
#------------------------------

with open(os.getcwd()+'\\output\\equilibrium\\pi_moist_res5.0.pkl', 'rb') as f:
    pi = pickle.load(f)['StateYear']
with open(os.getcwd()+'\\output\\equilibrium\\pi_moist_res5.0.pkl', 'rb') as f:
    Var = pickle.load(f)['Var']
    
#----------------------
# load zemba 2x CO2 sim
#----------------------

with open(os.getcwd()+'\\output\\equilibrium\\2xCO2_moist_res5.0.pkl', 'rb') as f:
    dco2 = pickle.load(f)['StateYear']
    
#--------------------------
# load zemba +2% insolation
#--------------------------

with open(os.getcwd()+'\\output\\equilibrium\\2%Inso_moist_res5.0.pkl', 'rb') as f:
    dinso = pickle.load(f)['StateYear']
    
    
# annual mean temperature
#------------------------

# pi
zemba_pi_tas             = pi["Ta"] 
zemba_pi_tas_global      = round(global_pymean(zemba_pi_tas.mean(axis=1), Var), 1)

# 2x CO2
zemba_2xCO2_tas          = dco2["Ta"] 
zemba_2xCO2_tas_global   = global_pymean(zemba_2xCO2_tas.mean(axis=1), Var)

# 2% Inso
zemba_2Inso_tas         = dinso["Ta"] 
zemba_2Inso_tas_global  = global_pymean(zemba_2Inso_tas.mean(axis=1), Var)


#---------
# plotting
#---------

# formating
#----------

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
fig, axs = pplt.subplots(shape, figsize = (8,4), sharey = False, sharex = False, hspace=1.8, grid = False)

# variables
x2CO2 = zemba_2xCO2_tas.mean(axis=1) - zemba_pi_tas.mean(axis=1)
Inso2 = zemba_2Inso_tas.mean(axis=1) - zemba_pi_tas.mean(axis=1)

# normalized variables
x2CO2_norm = (zemba_2xCO2_tas.mean(axis=1)  - zemba_pi_tas.mean(axis=1))/(zemba_2xCO2_tas_global-zemba_pi_tas_global)
Inso2_norm = (zemba_2Inso_tas.mean(axis=1) - zemba_pi_tas.mean(axis=1))/(zemba_2Inso_tas_global-zemba_pi_tas_global)

# format contour plots 
for i in [1,3]:       
    axs[i].format(ylim = (-90, 90), yformatter = 'deglat', ylocator = np.arange(-90, 120, 30), yminorlocator = np.arange(-90, 100, 10))
    axs[i].format(xlim = (1, 365), xlocator = np.arange(15, 365, 30), xminorlocator = [],
               xticklabels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

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
axs.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)
axs0.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)
axs2.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)

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
pi_dco2_line  = axs[0].plot(Var["lat"], zemba_2xCO2_tas.mean(axis=1) - zemba_pi_tas.mean(axis=1), color = 'black', lw = 2, linestyle = '-')
pi_dco2_norm_line = axs0.plot(Var["lat"], (zemba_2xCO2_tas.mean(axis=1) - zemba_pi_tas.mean(axis=1))/(zemba_2xCO2_tas_global-zemba_pi_tas_global), color = 'black', lw = 2, linestyle = '-')

# 2% Inso
pi_dinso_line = axs[2].plot(Var["lat"], zemba_2Inso_tas.mean(axis=1) - zemba_pi_tas.mean(axis=1), color = 'black', lw = 2, linestyle = ':')
pi_dinso_norm_line = axs2.plot(Var["lat"], (zemba_2Inso_tas.mean(axis=1) - zemba_pi_tas.mean(axis=1))/(zemba_2Inso_tas_global-zemba_pi_tas_global), color = 'black', lw = 2, linestyle = '-')

# seasonal cycle contour plots
cnt1 = axs[1].contourf(days, lat, zemba_2xCO2_tas - zemba_pi_tas, vmin = vmin, 
                       vmax = vmax, norm=colors.CenteredNorm(), cmap=cmap)

cnt2 = axs[3].contourf(days, lat, zemba_2Inso_tas - zemba_pi_tas, vmin = vmin, 
                       vmax = vmax, norm=colors.CenteredNorm(), cmap=cmap)

# colorbar
fig.colorbar(cnt1, loc='r', ticklabelsize=6)

fig.save(os.getcwd()+"/output/plots/f08.png", dpi= 300)
fig.save(os.getcwd()+"/output/plots/f08.pdf", dpi= 300)

#------------------------------------------------
# Print difference in global mean air temperature
#------------------------------------------------

# 2x CO2
print('##################')
print('Global mean Ta increase for CO2 doubling: ', round( zemba_2xCO2_tas_global - zemba_pi_tas_global, 2))
print('##################\n')

print('##################')
print('Global mean Ta increase for +2% insolation: ', round( zemba_2Inso_tas_global - zemba_pi_tas_global, 2))  
print('##################\n')

#-----------------------------------------------------------
# Print Arctic and Antarctic warming relative to global mean
#-----------------------------------------------------------

# 2% + Insolation
#----------------

# Arctic 

inso2_arctic = np.average(zemba_2Inso_tas[30:].mean(axis=1) - zemba_pi_tas[30:].mean(axis=1), weights=Var['sarea'][30:])

# Antarctic warming

inso2_antarctic = np.average(zemba_2Inso_tas[:6].mean(axis=1) - zemba_pi_tas[:6].mean(axis=1), weights=Var['sarea'][:6])



# 2x CO2
#-------

# Arctic 

x2CO2_arctic = np.average(zemba_2xCO2_tas[30:].mean(axis=1) - zemba_pi_tas[30:].mean(axis=1), weights=Var['sarea'][30:])

# Antarctic warming

x2CO2_antarctic = np.average(zemba_2xCO2_tas[:6].mean(axis=1) - zemba_pi_tas[:6].mean(axis=1), weights=Var['sarea'][:6])


# Print Arctic and Antarctic warming - normalized to global mean warming
#-----------------------------------------------------------------------

# 2x CO2

print('##################')
print('2x CO2 Normalized Arctic warming: ', round( x2CO2_arctic / (zemba_2xCO2_tas_global - zemba_pi_tas_global), 2))
print('2x CO2 Normalized Antarcitc warming: ', round( x2CO2_antarctic / (zemba_2xCO2_tas_global - zemba_pi_tas_global), 2))
print('##################\n')


# 2% Inso

print('##################')
print('2% Inso Normalized Arctic warming: ', round( inso2_arctic / (zemba_2Inso_tas_global - zemba_pi_tas_global), 2))
print('2% Inso Normalized Antarcitc warming: ', round( inso2_antarctic / (zemba_2Inso_tas_global - zemba_pi_tas_global), 2))
print('##################\n')