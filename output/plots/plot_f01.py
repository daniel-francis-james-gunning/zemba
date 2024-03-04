# -*- coding: utf-8 -*-
"""
plot figure 1 - methods section - cloud cover, hadley cell contribution, ocean velocities

@author: Daniel Gunning 
"""

import numpy as np
import proplot as pplt
import os
import importlib

# paths
output_path  = os.path.dirname(os.getcwd()) + '/equilibrium/' 
script_path  = os.path.dirname(os.path.dirname(os.path.dirname(output_path)))
input_path   = script_path + '/input'

# load zemba files
os.chdir(script_path)
from initialize_zemba import *
from zemba import *

# load pre-industrial input
os.chdir(input_path)
file = importlib.import_module('input_pi_res1')
input_zemba = file.input_zemba
os.chdir(script_path)

# load model constants and state variables
#-----------------------------------------

# constants
Var = get_constants(input_zemba)

# horizontal ocean velocities
State = initialize_state(Var, input_zemba) # load initial model state
u1, u2, ww = horizontal_ocean_velocities(Var["olat"], Var["olatb"], Var["olatr"], Var["olatbr"], Var["dlatr"], State["idxcos"], np.array([80.]), np.array([-70.]), np.zeros((Var["olat"].size))+0.7, np.zeros((Var["olatb"].size))+0.7, Var["r_earth"], State["ww"], Var["odepth"])
u1_velocity = u1 *(60*60*24*365) * 1e-5 # horizontal velocity (in x10^5 m/yr)
u1_transport = u1*2*np.pi*Var["r_earth"]*0.7*np.cos(Var["olatbr"])*Var["odepth"][0] * 1e-6 # horizonal transport (in sverdrup)
ww_velocity = ww *(60*60*24*365) # vertical velocity (in m/yr)
ww_transport = ww * 2*np.pi*Var["r_earth"]*0.7*np.cos(Var["olatr"])*Var["r_earth"]*Var["dlatr"] * 1e-6 # vertical transport (in sverdrup)

# weighting function for hadley cell transport
wf = np.exp( (-np.sin(Var["latbr"])**2) / (0.3**2) )

# plotting
#---------

# initialize figure
fig, axs = pplt.subplots(figsize = (7,5), nrows = 2, ncols = 2, sharey = False, sharex = False, hspace=1.8, grid = False)

# fonts
axs.format(ticklabelsize=8, ticklabelweight='normal', 
            ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', 
            titlesize=8, titleweight='normal',)


# x-axis 
locatorx      = np.arange(-90, 120, 30)
minorlocatorx = np.arange(-90, 100, 10)
axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90),  xformatter='deglat')

# format top left plot
for i in np.arange(0,1):
    
    # y-axis
    axs[i].format(ylim = (0.3, 1), ylocator = np.arange(0.3,1+0.1, 0.1), yminorlocator = [],)
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
                  
    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)
                  
    
# format top right plot
for i in np.arange(1,2):
    
    # y-axis
    axs[i].format(ylim = (-0.05, 1.05), ylocator = np.arange(0,1+0.1, 0.2), yminorlocator = np.arange(0,1+0.1, 0.1),)

    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)
                  
# format bottom left plot
for i in np.arange(2,3):
    
    # y-axis
    axs[i].format(ylim = (-90, 20), ylocator = np.arange(-80,20+20, 20), yminorlocator = np.arange(-80,20+10, 10),) 
    
# format bottom right plot
for i in np.arange(3,4):
    
    # y-axis
    axs[i].format(ylim = (-4, 6), ylocator = np.arange(-4,6+2, 2), yminorlocator = np.arange(-4,6+1, 1),)
    axs[i].yaxis.set_label_coords(-0.07,0.5)
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
                                   
# label x-axis
axs[2:].format(xlabel='Latitude')

# titles 
axs[0].format(title='(a) Fractional Cloud Cover', titleloc = 'left')
axs[1].format(title='(b) Weighting function for Hadley Cell', titleloc = 'left')
axs[2].format(title='(c) Upwards ocean velocities (m/yr)', titleloc = 'left')
axs[3].format(title= r'(d) Meridional Ocean Transport', titleloc = 'left')

# plot cloud cover fractions
axs[0,0].plot(Var["lat"], input_zemba["ccl"], color = "red9", ls = '-', lw = 1, label = ["Land"])
axs[0,0].plot(Var["lat"], input_zemba["cco"], color = "blue", ls = '-.', lw = 1, label = ["Ocean"])
axs[0,0].legend(loc="uc", ncols=1, frame = False, prop = { "size": 8 })

# plot hadley cell weighting function
axs[0,1].plot(Var["latb"], wf, lw = 1, color = "black")

# plot vertical ocean velocities
axs[1,0].plot(Var["olat"], ww_velocity, lw = 1, color = "black", label = ["Velocity"])

# plot horizontal ocean velocities
axs[1,1].plot(Var["olatb"], u1_velocity, color = "black", lw = 1, label = ["Velocity"])
axs[1,1].format(ylabel = r'Northward velocities ($\times 10^{5}$ m/yr)')

# plot horizontal ocean transport
axs11 = axs[1,1].twinx()
axs11.format(ylim = (-20, 25), ylabel = "Transport (Sv)", ylocator = np.arange(-20, 25+5,5),
             yminorlocator=[], ticklabelsize=8, ticklabelweight='normal', 
             ylabelsize=8, ylabelweight='normal', xlabelsize=8, xlabelweight='normal', 
             titlesize=8, titleweight='normal',)
axs11.yaxis.label.set_color('gray7')        
axs11.tick_params(axis='y', colors='gray7')    
axs11.plot(Var["olatb"], u1_transport, color = "gray7", lw = 1, linestyle = ":")
axs[1,1].plot(np.empty(1), color = "black", lw = 1, linestyle = ":", label = ["Mass transport"]) # for legend
axs[1,1].legend(loc = "uc", ncols=1, frame = False, prop = { "size": 8 })

def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)  
align_yaxis(axs[1,1], 0, axs11, 0)


fig.save(os.getcwd()+"/output/plots/f01.png", dpi= 300)
fig.save(os.getcwd()+"/output/plots/f01.pdf", dpi= 300)

