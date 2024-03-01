# -*- coding: utf-8 -*-
"""
Plot Figure 3 (Pre-Industrial - Results - Hydrological)

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
ebm_precip = (pi['precip_flux'].mean(axis=1)/1000)*(60*60*24)*1000  # precipitation
# ebm_precip_hadley_off = (pi_hadley_off['precip_flux'].mean(axis=1)/1000)*(60*60*24)*1000  # precipitation
ebm_precip_global = global_mean2(ebm_precip, Var["lat"], Var["dlat"])

ebm_sf = (pi['snowfall_flux'].mean(axis=1)/1000)*(60*60*24)*1000  # precipitation
ebm_sf_global = global_mean2(ebm_sf, Var["lat"], Var["dlat"])

ebm_evap   = (pi['evap_flux'].mean(axis=1)/1000)*(60*60*24)*1000    # evaporation
ebm_evap_global = global_mean2(ebm_evap, Var["lat"], Var["dlat"])

# land
ebm_precip_land = (pi['precip_flux_land'].mean(axis=1)/1000)*(60*60*24)*1000  # precipitation
ebm_evap_land   = (pi['evap_flux_land'].mean(axis=1)/1000)*(60*60*24)*1000    # evaporationebm_evap_land_zones = spatial_average(ebm_evap_land, Var["lat"], np.diff(Var["lat"])[0], Var["land_fraction"]) # averaged for different zones...

# ocean
ebm_precip_ocean = (pi['precip_flux_ocean'].mean(axis=1)/1000)*(60*60*24)*1000  # precipitation
ebm_evap_ocean   = (pi['evap_flux_ocean'].mean(axis=1)/1000)*(60*60*24)*1000    # evaporation


#-----------------
# Add NorESM2 data
#-----------------

# load annual data
noresm2_lat      = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=0)
noresm2_precip   = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=2)
noresm2_sf       = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=3)
noresm2_evap     = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=4)

# interpolate data
noresm2_precip   = np.interp(Var['lat'], noresm2_lat, noresm2_precip)
noresm2_sf       = np.interp(Var['lat'], noresm2_lat, noresm2_sf)
noresm2_evap     = np.interp(Var['lat'], noresm2_lat, noresm2_evap)

# change from kg m-2 s-1 to mm day-1
noresm2_precip   = noresm2_precip * 1000 * (60*60*24) / 1000
noresm2_sf       = noresm2_sf * 1000 * (60*60*24) / 1000
noresm2_evap     = noresm2_evap * 1000 * (60*60*24) / 1000

# global_mean 
noresm2_precip_global   = global_mean2(noresm2_precip, Var["lat"], Var["dlat"])
noresm2_sf_global       = global_mean2(noresm2_sf, Var["lat"], Var["dlat"])
noresm2_evap_global     = global_mean2(noresm2_evap, Var["lat"], Var["dlat"])


#---------------------------
# ERA5 1940-1970 climatology
#---------------------------

# load annual data
era5_lat      = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=0)
era5_precip   = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=2)
era5_sf       = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=3)
era5_evap     = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=4)

# interpolate data
era5_precip   = np.interp(Var['lat'], era5_lat, era5_precip)
era5_sf       = np.interp(Var['lat'], era5_lat, era5_sf)
era5_evap     = -np.interp(Var['lat'], era5_lat, era5_evap)

# change from kg m-2 s-1 to mm day-1
era5_precip   = era5_precip * 1000 * (60*60*24) / 1000
era5_sf       = era5_sf * 1000 * (60*60*24) / 1000
era5_evap     = era5_evap * 1000 * (60*60*24) / 1000

# global_mean 
era5_precip_global   = global_mean2(era5_precip, Var["lat"], Var["dlat"])
era5_sf_global       = global_mean2(era5_sf, Var["lat"], Var["dlat"])
era5_evap_global     = global_mean2(era5_evap, Var["lat"], Var["dlat"])

#------------
# Plot figure 
#------------


# constants
#----------

ebm_color = "black"
ebm_lw    = 1.
ebm_ls    = "-"

noresm2_color = "blue9"
noresm2_lw    = 1.
noresm2_ls    = "-"

era5_color = "red9"
era5_lw    = 1.
era5_ls    = "-"

ebm_minus_noresm2_color = "blue9"
ebm_minus_noresm2_lw    = 1.
ebm_minus_noresm2_ls    = "-."

ebm_minus_era5_color = "red9"
ebm_minus_era5_lw    = 1.
ebm_minus_era5_ls    = ":"

legend_fs = 7.


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
locatorx      = np.arange(-90, 120, 30)
minorlocatorx = np.arange(-90, 100, 10)
axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))

# format top left subplot
for i in np.arange(0,1):
    
    # y-axis
    axs[i].format(ylim = (-0.5, 7), yminorlocator=[], ylocator = np.arange(0, 7+1, 1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)

# format top left subplot
for i in np.arange(1,2):
    
    # y-axis
    axs[i].format(ylim = (-0.5, 7), yminorlocator=[], ylocator = np.arange(0, 7+1, 1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False)
    
    
# format bottom subplots
for i in np.arange(2,3+1):
    
    # y-axis
    axs[i].format(ylim = (-2.5, 2.5), yminorlocator=[], ylocator = np.arange(-2, 2+1, 1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].format(xlabel = "Latitude", xformatter='deglat')    

# titles
axs[0].format(title = r'(a) Precipitation and Snowfall (mm/day)', titleloc = 'left')
axs[1].format(title = r'(b) Evaporation (mm/day)', titleloc = 'left')
axs[2].format(title = r'(b) Difference in Precipitation and Snowfall (mm/day)', titleloc = 'left')
axs[3].format(title = r'(d) Difference in Evaporation (mm/day)', titleloc = 'left')



# plot precipitation
#-------------------

# lines
ebm_precip_line = axs[0].plot(Var["lat"], ebm_precip, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls)
noresm2_precip_line = axs[0].plot(Var["lat"], noresm2_precip, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls)
era5_precip_line = axs[0].plot(Var["lat"], era5_precip, color = era5_color, lw = era5_lw, linestyle = era5_ls)

ebm_sf_line = axs[0].plot(Var["lat"], ebm_sf, color = ebm_color, lw = ebm_lw, linestyle = ":", label = ["EBCM"])
noresm2_sf_line = axs[0].plot(Var["lat"], noresm2_sf, color = noresm2_color, lw = noresm2_lw, linestyle = ":", label = ["noresm"])
era5_sf_line = axs[0].plot(Var["lat"], era5_sf, color = era5_color, lw = era5_lw, linestyle = ":", label = ["ERA5 (1940-1970)"])

# legend
axs[0].legend(handles = [ebm_precip_line, noresm2_precip_line, era5_precip_line],
              labels  = ["ZEMBA (Precipitation)", "NorESM2 (Precipitation)", "ERA5 (Precipitation)"], frameon = False, loc = "ul",
              bbox_to_anchor=(0.05, 0.9), ncols = 1, prop={'size':legend_fs})

# legend
axs[0].legend(handles = [ebm_sf_line, noresm2_sf_line, era5_sf_line],
              labels  = ["ZEMBA (Snowfall)", "NorESM2 (Snowfall)", "ERA5 (Snowfall)"], frameon = False, loc = "ur",
              bbox_to_anchor=(0.95, 0.9), ncols = 1, prop={'size':legend_fs})


# plot differences in precipitation
#----------------------------------


# lines
ebm_minus_noresm_precip = axs[2].plot(Var["lat"], ebm_precip - noresm2_precip, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = "-")
ebm_minus_era5_precip   = axs[2].plot(Var["lat"], ebm_precip - era5_precip, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = "-")

ebm_minus_noresm_sf = axs[2].plot(Var["lat"], ebm_sf - noresm2_sf, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ":")
ebm_minus_era5_sf   = axs[2].plot(Var["lat"], ebm_sf - era5_sf, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ":")

# legend
axs[2].legend(handles = [ebm_minus_noresm_precip, ebm_minus_era5_precip],
              labels  = ["ZEMBA − NorESM2 (Precipitation)", "ZEMBA − ERA5 (Precipitation) "], frameon = False,  loc = "ll",
               bbox_to_anchor=(0.01, 0.01), ncols = 1, prop={'size':legend_fs})

axs[2].legend(handles = [ebm_minus_noresm_sf, ebm_minus_era5_sf],
              labels  = ["ZEMBA − NorESM2 (Snowfall)", "ZEMBA − ERA5 (Snowfall)"], frameon = False,  loc = "ur",
               bbox_to_anchor=(0.98, 0.97), ncols = 1, prop={'size':legend_fs})

# plot evaporation
#------------------

# lines
ebm_evap_line = axs[1].plot(Var["lat"], ebm_evap, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
noresm2_evap_line = axs[1].plot(Var["lat"], noresm2_evap, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["noresm"])
era5_evap_line = axs[1].plot(Var["lat"], era5_evap, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])

# legend
axs[1].legend(handles = [ebm_evap_line, noresm2_evap_line, era5_evap_line],
              labels  = ["ZEMBA", "NorESM2", "ERA5"], frameon = False, loc = "uc",
              bbox_to_anchor=(0.3, 0.9), ncols = 1, prop={'size':legend_fs})

# plot differences in evaporation
#--------------------------------

# lines
ebm_minus_noresm_evap = axs[3].plot(Var["lat"], ebm_evap - noresm2_evap, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = "-")
ebm_minus_era5_evap = axs[3].plot(Var["lat"], ebm_evap - era5_evap, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = "-")

# legend
axs[3].legend(handles = [ebm_minus_noresm_evap, ebm_minus_era5_evap],
              labels  = ["ZEMBA − NorESM2", "ZEMBA − ERA5"], frameon = False,  loc = "lc",
               bbox_to_anchor=(0.5, 0.05), ncols = 1, prop={'size':legend_fs})

            
fig.save(os.getcwd()+"/output/plots/f03.png", dpi = 400)
fig.save(os.getcwd()+"/output/plots/f03.pdf", dpi = 400)




print('###########################')
print("Global Mean Precipitation....")
print('###########################')

print("pyEBM: " + str(round(ebm_precip_global, 2)))

print("NorESM2: " + str(round(noresm2_precip_global, 2)))

print("ERA5: " + str(round(era5_precip_global, 2))+'\n')

print('###########################')
print("Global Mean Evaporation....")
print('###########################')

print("pyEBM: " + str(round(ebm_evap_global, 2)))

print("NorESM2: " + str(round(noresm2_evap_global, 2)))

print("ERA5: " + str(round(era5_evap_global, 2))+'\n')

print('###########################')
print("Global Mean Snowfall....")
print('###########################')

print("pyEBM: " + str(round(ebm_sf_global, 2)))

print("NorESM2: " + str(round(noresm2_sf_global, 2)))

print("ERA5: " + str(round(era5_sf_global, 2))+'\n')


