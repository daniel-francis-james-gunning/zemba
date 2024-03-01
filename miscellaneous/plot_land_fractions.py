# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

Plot land_fractions
"""

#-----------------------
# Import modules/scripts
#-----------------------

# modules
#--------
import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit
import os
import sys
import pickle
import proplot as pplt


# import scripts
#---------------

# change directory
cdir = os.getcwd()
pdir = os.path.dirname(os.getcwd())
os.chdir(pdir+"/energy_balance_model/scripts/")
from initialize_test2 import *


Var = get_constants(land_fraction  = 'ICE-6G_C_0kyr',
                    land_elevation = 'ICE-6G_C_0kyr',
                    ice_fraction   = 'ICE-6G_C_0kyr',
                    resolution = 1., nyrs= 3000.)

VarLGM = get_constants(land_fraction  = 'ICE-6G_C_0kyr',
                    land_elevation = 'ICE-6G_C_21kyr',
                    ice_fraction   = 'ICE-6G_C_21kyr',
                    resolution = 1., nyrs= 3000.)


# constants
#----------

ebm_color = "black"
ebm_lw    = 1
ebm_ls    = "-"

# formating
#----------

# figure and axes
fig, axs = pplt.subplots(nrows=3, figsize = (10,6), sharey = False, sharex = False, hspace=1.5, grid = False)

# fonts
axs.format(ticklabelsize=6, ticklabelweight='normal', 
           ylabelsize=7, ylabelweight='normal',
            xlabelsize=7, xlabelweight='normal', 
            titlesize=7, titleweight='normal',)
            
# x-axis
axs.format(xlim = (-90, 90),
           xlocator = np.arange(-90, 120, 30),
           xminorlocator = np.arange(-90, 100, 10),)
    
# format top plot
for i in np.arange(0, 1):
    
    # y-axis
    axs[i].format(ylim = (-0.1, 1.1), ylocator = np.arange(0, 1+0.2, 0.2), yminorlocator = [])
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)     

# format middle plot
for i in np.arange(1, 2):
    
    # y-axis
    axs[i].format(ylim = (-100, 3100), ylocator = np.arange(0, 3000+500, 500), yminorlocator = [])
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False) 
    
    
# format bottom plots
for i in np.arange(2, 3):
    
    # y-axis
    axs[i].format(ylim = (-0.1, 1.1), ylocator = np.arange(0, 1+0.2, 0.2), yminorlocator =[])
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].format(xlabel = "Latitude", xformatter='deglat')

   
# titles
axs[1].format(title = r'(A) Zonal Mean Land Elevation (m)', titleloc = "left")
axs[0].format(title = r'(B) Zonal Mean Land Fraction', titleloc = "left")
axs[2].format(title = r'(C) Zonal Mean Land Ice Fraction', titleloc = "left")


# plot land fraction
#-------------------

# lines
axs[0].plot(Var["lat"], Var["land_fraction"], color = ebm_color, lw = ebm_lw, ls = '-')

# plot land elevation
#--------------------

# lines
pi_elev = axs[1].plot(Var["lat"], Var["land_height"], color = ebm_color, lw = ebm_lw, ls = '-')

lgm_elev = axs[1].plot(Var["lat"], VarLGM["land_height"], color = 'blue9', lw = ebm_lw, ls = '-')

axs[1].legend(handles = [pi_elev, lgm_elev], 
              labels = ["PI", "LGM",], ncols = 2, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.8), prop={'size':6})

# plot ice fraction
#------------------

# lines
pi_if = axs[2].plot(Var["lat"], Var["ice_fraction"], color = ebm_color, lw = ebm_lw, ls = '-')
lgm_if = axs[2].plot(Var["lat"], VarLGM["ice_fraction"], color = 'blue9', lw = ebm_lw, ls = '-')

axs[2].legend(handles = [pi_if, lgm_if], 
              labels = ["PI", "LGM",], ncols = 2, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.8), prop={'size':6})

