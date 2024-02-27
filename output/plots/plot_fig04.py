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
noresm_annual = xr.open_dataset(os.getcwd() +  "/other_data/noresm2/monthly/noresm2_annual.nc")
noresm_annual = noresm_annual.interp(lat = Var["lat"]) # interpolate to EBM

# dictionary
noresm = {}

# annual_mean (across latitude and longitude)
noresm["rsut_spatial"]      = noresm_annual["rsut"] # TOA upwards
noresm["rsdt_spatial"]      = noresm_annual["rsdt"] # TOA donwards
noresm["rsds_spatial"]      = noresm_annual["rsds"] # BOA downwards
noresm["rsus_spatial"]      = noresm_annual["rsus"] # BOA upwards
noresm["alpha_toa_spatial"] = noresm["rsut_spatial"]/noresm["rsdt_spatial"] # TOA albedo
noresm["alpha_boa_spatial"] = noresm["rsus_spatial"]/noresm["rsds_spatial"] # BOA albedo

# annual and zonal mean
noresm["rsdt"] = noresm["rsdt_spatial"].mean(dim="lon", skipna=True) # TOA downards
noresm["rsut"] = noresm["rsut_spatial"].mean(dim="lon", skipna=True) # TOA upwards
noresm["rsds"] = noresm["rsds_spatial"].mean(dim="lon", skipna=True) # BOA donwards
noresm["rsus"] = noresm["rsus_spatial"].mean(dim="lon", skipna=True) # BOA upwards

# planetary albedo
noresm["alpha_toa"] = noresm["alpha_toa_spatial"].mean(dim="lon", skipna=True) # TOA albedo
noresm["alpha_toa_global"] = round(global_mean2(noresm["alpha_toa"].to_numpy(), Var["lat"], Var["dlat"]), 2)

# surface albedo
noresm["alpha_boa"] = noresm["alpha_boa_spatial"].mean(dim="lon", skipna=True) # BOA albedo
noresm["alpha_boa_global"] = round(global_mean2(noresm["alpha_boa"].to_numpy(), Var["lat"], Var["dlat"]), 2)

# planetary albedo -- land
noresm["alpha_toa_land"]  = (noresm["alpha_toa_spatial"]*noresm_annual["land_mask"]).mean(dim="lon", skipna=True) # TOA albedo over land

# surface albedo -- land
noresm["alpha_boa_land"] = (noresm["alpha_boa_spatial"]*noresm_annual["land_mask"]).mean(dim="lon", skipna=True) # BOA albedo over land

# planetary albedo -- ocean
noresm["alpha_toa_ocean"] = (noresm["alpha_toa_spatial"]*noresm_annual["ocean_mask"]).mean(dim="lon", skipna=True) # TOA albedo over ocean

# surface albedo -- ocean
noresm["alpha_boa_ocean"] = (noresm["alpha_boa_spatial"]*noresm_annual["ocean_mask"]).mean(dim="lon", skipna=True) # BOA albedo over ocean

#----------------------------
# CERES 2005-2015 climatology
# ---------------------------

# load data
ceres_data = xr.open_dataset(os.getcwd() + "/other_data/ceres/CERES_EBAF_Ed4.2_Subset_CLIM01-CLIM12 (1).nc")

# interpolate to NorESM2 grid.
ceres_data = ceres_data.interp(lon = noresm_annual.lon) # interpolate

# dictionary
ceres = {}

# annual mean
ceres["rsut_spatial"]      = np.sum(Var["days_in_months"].reshape(12,1,1)*ceres_data["toa_sw_all_clim"], axis = 0) / np.sum(Var["days_in_months"]) # TOA upwards
ceres["rsdt_spatial"]      = np.sum(Var["days_in_months"].reshape(12,1,1)*ceres_data["solar_clim"], axis = 0) / np.sum(Var["days_in_months"]) # TOA donwards
ceres["rsus_spatial"]      = np.sum(Var["days_in_months"].reshape(12,1,1)*ceres_data["sfc_sw_up_all_clim"], axis = 0) / np.sum(Var["days_in_months"]) # BOA upwards
ceres["rsds_spatial"]      = np.sum(Var["days_in_months"].reshape(12,1,1)*ceres_data["sfc_sw_down_all_clim"], axis = 0) / np.sum(Var["days_in_months"]) # BOA donwards
ceres["alpha_toa_spatial"] = ceres["rsut_spatial"]/ceres["rsdt_spatial"] # TOA albedo
ceres["alpha_boa_spatial"] = ceres["rsus_spatial"]/ceres["rsds_spatial"] # BOA albedo

# annual and zonal mean
ceres["rsdt"] = ceres["rsdt_spatial"].mean(dim="lon", skipna=True) # TOA downards
ceres["rsut"] = ceres["rsut_spatial"].mean(dim="lon", skipna=True) # TOA upwards
ceres["rsds"] = ceres["rsds_spatial"].mean(dim="lon", skipna=True) # BOA donwards
ceres["rsus"] = ceres["rsus_spatial"].mean(dim="lon", skipna=True) # BOA upwards

# planetary albedo
ceres["alpha_toa"] = ceres["alpha_toa_spatial"].mean(dim="lon", skipna=True) # TOA albedo

# surface albedo
ceres["alpha_boa"] = ceres["alpha_boa_spatial"].mean(dim="lon", skipna=True) # BOA albedo

# planetary albedo -- land
ceres["alpha_toa_land"] = (ceres["alpha_toa_spatial"]*noresm_annual["land_mask"]).mean(dim="lon", skipna=True) # TOA albedo over land

# surface albedo -- land
ceres["alpha_boa_land"] = (ceres["alpha_boa_spatial"]*noresm_annual["land_mask"]).mean(dim="lon", skipna=True) # BOA albedo over land

# planetary albedo -- ocean
ceres["alpha_toa_ocean"] = (ceres["alpha_toa_spatial"]*noresm_annual["ocean_mask"]).mean(dim="lon", skipna=True) # TOA albedo over ocean

# surface albedo -- ocean
ceres["alpha_boa_ocean"] = (ceres["alpha_boa_spatial"]*noresm_annual["ocean_mask"]).mean(dim="lon", skipna=True) # BOA albedo over ocean

#---------------------------
# ERA5 1940-1970 climatology
#---------------------------

# load data
era5_annual = xr.open_dataset(os.getcwd() +  "/other_data/era5/radiation_era5_1940_1970.nc")

# interpolate to NorESM2 grid.
era5_annual = era5_annual.interp(longitude = noresm_annual.lon, latitude = noresm_annual.lat) 

# era5
era5 = {}

# annual mean
era5["rsdt_spatial"] = era5_annual.mtdwswrf # TOA downards
era5["rsut_spatial"] = era5_annual.mtdwswrf - era5_annual.mtnswrf # TOA upwards
era5["rsds_spatial"] = era5_annual.msdwswrf # BOA donwards
era5["rsus_spatial"] = era5_annual.msdwswrf - era5_annual.msnswrf # BOA upwards
era5["alpha_toa_spatial"] = era5["rsut_spatial"]/era5["rsdt_spatial"] # TOA albedo
era5["alpha_boa_spatial"] = era5["rsus_spatial"]/era5["rsds_spatial"] # BOA albedo

# annual and zonal mean
era5["rsdt"] = era5["rsdt_spatial"].mean(dim="lon", skipna=True) # TOA downards
era5["rsut"] = era5["rsut_spatial"].mean(dim="lon", skipna=True) # TOA upwards
era5["rsds"] = era5["rsds_spatial"].mean(dim="lon", skipna=True) # BOA donwards
era5["rsus"] = era5["rsus_spatial"].mean(dim="lon", skipna=True) # BOA upwards

# planetary albedo
era5["alpha_toa"] = era5["alpha_toa_spatial"].mean(dim="lon", skipna=True) # TOA albedo
era5["alpha_toa_global"] = round(global_mean2(era5["alpha_toa"].to_numpy(), Var["lat"], Var["dlat"]), 2)

# surface albedo
era5["alpha_boa"] = era5["alpha_boa_spatial"].mean(dim="lon", skipna=True) # BOA albedo
era5["alpha_boa_global"] = round(global_mean2(era5["alpha_boa"].to_numpy(), Var["lat"], Var["dlat"]), 2)

# planetary albedo -- land
era5["alpha_toa_land"] = (era5["alpha_toa_spatial"]*noresm_annual["land_mask"]).mean(dim="lon", skipna=True) # TOA albedo over land

# surface albedo -- land
era5["alpha_boa_land"] = (era5["alpha_boa_spatial"]*noresm_annual["land_mask"]).mean(dim="lon", skipna=True) # BOA albedo over land

# planetary albedo -- ocean
era5["alpha_toa_ocean"] = (era5["alpha_toa_spatial"]*noresm_annual["ocean_mask"]).mean(dim="lon", skipna=True) # TOA albedo over ocean

# surface albedo -- ocean
era5["alpha_boa_ocean"] = (era5["alpha_boa_spatial"]*noresm_annual["ocean_mask"]).mean(dim="lon", skipna=True) # BOA albedo over ocean

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
noresm_toa_line = axs[0].plot(noresm_annual.lat, noresm["alpha_toa"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls)
era5_toa_line   = axs[0].plot(era5_annual.lat, era5["alpha_toa"], color = era5_color, lw = era5_lw, linestyle = era5_ls)

# legend
axs[0].legend(handles = [ebm_toa_line, noresm_toa_line, era5_toa_line],
              labels  = ["ZEMBA", "NorESM2", "ERA5"], frameon = False,
              loc = "c", bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})

# plot difference in planetary albedo
#------------------------------------

# lines
ebm_minus_noresm_toa_line = axs[2].plot(Var["lat"], ebm["alpha_toa"] - noresm["alpha_toa"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_toa_line   = axs[2].plot(Var["lat"], ebm["alpha_toa"] - era5["alpha_toa"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# legend
axs[2].legend(handles = [ebm_minus_noresm_toa_line, ebm_minus_era5_toa_line],
              labels  = ["ZEMBA - NorESM2", "ZEMBA - ERA5"],
              ncols = 2, loc = "uc", frameon = False, 
              bbox_to_anchor=(0.5, 0.2), prop={'size':legend_fs}) 



# plot surface albedo
#--------------------

# lines
ebm_boa_line    = axs[1].plot(Var["lat"], ebm["alpha_boa"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
noresm_boa_line = axs[1].plot(noresm_annual.lat, noresm["alpha_boa"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["noresm"])
era5_boa_line   = axs[1].plot(era5_annual.lat, era5["alpha_boa"], color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])

# legend
axs[1].legend(handles = [ebm_boa_line, noresm_boa_line, era5_boa_line],
              labels  = ["ZEMBA", "NorESM2", "ERA5"], frameon = False,
              loc = "c", bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})


# plot differences in surface albedo
#-------------------------------------
  
# lines
ebm_minus_noresm_boa_line = axs[3].plot(Var["lat"], ebm["alpha_boa"] - noresm["alpha_boa"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_boa_line   = axs[3].plot(Var["lat"], ebm["alpha_boa"] - era5["alpha_boa"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

axs[3].legend(handles = [ebm_minus_noresm_boa_line, ebm_minus_era5_boa_line],
              labels  = ["ZEMBA - NorESM2", "ZEMBA - ERA5"],
              ncols = 2, loc = "uc", frameon = False, 
              bbox_to_anchor=(0.5, 0.2), prop={'size': legend_fs}) 


fig.save(os.getcwd() +"/output/plots/f04.png", dpi = 400)
fig.save(os.getcwd() +"/output/plots/f04.pdf", dpi = 400)


# #--------------
# # Plot figure 3 -- v2
# #--------------

# # constants
# #----------

# ebm_color = "black"
# ebm_lw    = 1.5
# ebm_ls    = "-"

# noresm_color = "blue9"
# noresm_lw    = 1.5
# noresm_ls    = "-"

# era5_color = "red9"
# era5_lw    = 1.5
# era5_ls    = "-"

# ebm_minus_noresm_color = "blue9"
# ebm_minus_noresm_lw    = 1.5
# ebm_minus_noresm_ls    = "-."

# ebm_minus_era5_color = "red9"
# ebm_minus_era5_lw    = 1.5
# ebm_minus_era5_ls    = "-."

# # formating
# #----------

# # shape
# shape = [  
#         [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#         [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#         [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#         [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],]

# # figure and axes
# fig, axs = pplt.subplots(shape, figsize = (12,6), sharey = False, sharex = False)

# # secondary yaxis
# axs_twin = axs.twinx() 

# # fonts
# axs.format(ticklabelsize=6, ticklabelweight='normal', ylabelsize=10, ylabelweight='bold',
#             xlabelsize=7, xlabelweight='normal', titlesize=10, titleweight='bold', abc='A)', 
#             abcloc='ur', abcbbox=False, xlim = (-90, 90))

# axs_twin.format(ticklabelsize=6, ticklabelweight='normal', ylabelsize=7, ylabelweight='normal',
#             xlabelsize=7, xlabelweight='normal', titlesize=10, titleweight='bold', abc='A)', 
#             abcloc='ur', abcbbox=False, xlim = (-90, 90))


# # format bottom row
# locatorx      = np.arange(-90, 120, 30)
# minorlocatorx = np.arange(-90, 100, 10)
# axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))


# # format top row
# for i in np.arange(0,5+1):
    
#     axs[i].format(xlocator = locatorx, xlim = (-90, 90))
    
#     # set y-axis range
#     axs[i].format(ylim = (-0.2, 0.9), yminorlocator=np.arange(0.1, 0.9+0.1, 0.1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(0.2, 0.9+0.1, 0.1))
    
#     # change position of yaxis label
#     axs[i].yaxis.set_label_coords(-0.1,0.65)
    
# axs[3:].format(xlabel = "Latitude")

# # format bottom row
# for i in np.arange(3,5+1):
    
    
#     # set y-axis range
#     axs[i].format(ylim = (-0.4, 0.9), yminorlocator=np.arange(0.1, 0.9+0.1, 0.1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(0.1, 0.9+0.1, 0.1))


# # plot planetary albedo
# #----------------------

# # surface average
# ebm_toa_line    = axs[0].plot(Var["lat"], ebm["alpha_toa"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# noresm_toa_line = axs[0].plot(noresm_annual.lat, noresm["alpha_toa"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["noresm"])
# era5_toa_line   = axs[0].plot(era5_annual.lat, era5["alpha_toa"], color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[0].format(title = "Average", ylabel= 'Planetary Albedo')
# axs[0].legend(handles = [ebm_toa_line, noresm_toa_line, era5_toa_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "uc",
#               bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})

# # plot land temperature
# ebm_toa_land_line    = axs[1].plot(Var["lat"], ebm["alpha_toa_land"], color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# noresm_toa_land_line    = axs[1].plot(noresm_annual.lat, noresm["alpha_toa_land"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# era5_toa_land_line    = axs[1].plot(era5_annual.lat, era5["alpha_toa_land"], color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[1].format(title = "Land")
# axs[1].legend(handles = [ebm_toa_land_line, noresm_toa_land_line, era5_toa_land_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "uc",
#               bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})

# # plot ocean temperature
# ebm_toa_ocean_line    = axs[2].plot(Var["lat"], ebm["alpha_toa_ocean"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# noresm_toa_ocean_line    = axs[2].plot(noresm_annual.lat, noresm["alpha_toa_ocean"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# era5_toa_ocean_line    = axs[2].plot(era5_annual.lat, era5["alpha_toa_ocean"], color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[2].format(title = "Surface Ocean")
# axs[2].legend(handles = [ebm_toa_ocean_line, noresm_toa_ocean_line, era5_toa_ocean_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "uc",
#               bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})

# # plot differences in planetary albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(0, 5+1):
#     axs_twin[i].set_yticks(np.arange(-0.2, 0.2 + 0.1, 0.1))
#     axs_twin[i].format(ylim = (-0.3, 1.3), ylabel = 'Difference in Albedo', yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.1,0.15)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")

# # atmosphere
# ebm_minus_noresm_toa_line = axs_twin[0].plot(Var["lat"], ebm["alpha_toa"] - noresm["alpha_toa"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_toa_line = axs_twin[0].plot(Var["lat"], ebm["alpha_toa"] - era5["alpha_toa"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# axs[0].legend(handles = [ebm_minus_noresm_toa_line, ebm_minus_era5_toa_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False,  loc = "lc",
#                bbox_to_anchor=(0.5, 0.02), ncols = 1, prop={'size':legend_fs})

# # land
# ebm_minus_noresm_toa_land_line = axs_twin[1].plot(Var["lat"], ebm["alpha_toa_land"] - noresm["alpha_toa_land"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_toa_land_line = axs_twin[1].plot(Var["lat"], ebm["alpha_toa_land"] - era5["alpha_toa_land"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# axs[1].legend(handles = [ebm_minus_noresm_toa_land_line, ebm_minus_era5_toa_land_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False, loc = "lc",
#               bbox_to_anchor=(0.5, 0.02), ncols = 1, prop={'size':legend_fs})

# # ocean
# ebm_minus_noresm_toa_ocean_line = axs_twin[2].plot(Var["lat"], ebm["alpha_toa_ocean"] - noresm["alpha_toa_ocean"] , color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_toa_ocean_line = axs_twin[2].plot(Var["lat"], ebm["alpha_toa_ocean"]  - era5["alpha_toa_ocean"] , color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# axs[2].legend(handles = [ebm_minus_noresm_toa_ocean_line, ebm_minus_era5_toa_ocean_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False, loc = "lc",
#               bbox_to_anchor=(0.5, 0.02), ncols = 1, prop={'size':legend_fs})



# # plot surface albedo
# #--------------------

# # surface average
# ebm_boa_line    = axs[3].plot(Var["lat"], ebm["alpha_boa"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# noresm_boa_line = axs[3].plot(noresm_annual.lat, noresm["alpha_boa"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["noresm"])
# era5_boa_line   = axs[3].plot(era5_annual.lat, era5["alpha_boa"], color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[3].format(ylabel = 'Surface Albedo')
# axs[3].legend(handles = [ebm_boa_line, noresm_boa_line, era5_boa_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "uc",
#               bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})

# # plot land temperature
# ebm_boa_land_line    = axs[4].plot(Var["lat"], ebm["alpha_boa_land"], color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# noresm_boa_land_line    = axs[4].plot(noresm_annual.lat, noresm["alpha_boa_land"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# era5_boa_land_line    = axs[4].plot(era5_annual.lat, era5["alpha_boa_land"], color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[4].legend(handles = [ebm_boa_land_line, noresm_boa_land_line, era5_boa_land_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "uc",
#               bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})

# # plot ocean temperature
# ebm_boa_ocean_line    = axs[5].plot(Var["lat"], ebm["alpha_boa_ocean"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# noresm_boa_ocean_line    = axs[5].plot(noresm_annual.lat, noresm["alpha_boa_ocean"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# era5_boa_ocean_line    = axs[5].plot(era5_annual.lat, era5["alpha_boa_ocean"], color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[5].legend(handles = [ebm_boa_ocean_line, noresm_boa_ocean_line, era5_boa_ocean_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "uc",
#               bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})

# # plot differences in planetary albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(3, 5+1):
#     axs_twin[i].format(ylabel = 'Difference in Albedo', yminorlocator = [])
#     axs_twin[i].set_yticks(np.arange(-0.2, 0.2 + 0.1, 0.1))
#     axs_twin[i].format(ylim = (-0.3, 1.3), yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.1,0.15)


# # atmosphere
# ebm_minus_noresm_boa_line = axs_twin[3].plot(Var["lat"], ebm["alpha_boa"] - noresm["alpha_boa"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_boa_line = axs_twin[3].plot(Var["lat"], ebm["alpha_boa"] - era5["alpha_boa"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# axs[3].legend(handles = [ebm_minus_noresm_boa_line, ebm_minus_era5_boa_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False, loc = "lc",
#                bbox_to_anchor=(0.5, 0.02), ncols = 1, prop={'size':legend_fs})

# # land
# ebm_minus_noresm_boa_land_line = axs_twin[4].plot(Var["lat"], ebm["alpha_boa_land"] - noresm["alpha_boa_land"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_boa_land_line = axs_twin[4].plot(Var["lat"], ebm["alpha_boa_land"] - era5["alpha_boa_land"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# axs[4].legend(handles = [ebm_minus_noresm_boa_land_line, ebm_minus_era5_boa_land_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False, loc = "lc",
#               bbox_to_anchor=(0.5, 0.02), ncols = 1, prop={'size':legend_fs})

# # ocean
# ebm_minus_noresm_boa_ocean_line = axs_twin[5].plot(Var["lat"], ebm["alpha_boa_ocean"] - noresm["alpha_boa_ocean"] , color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_boa_ocean_line = axs_twin[5].plot(Var["lat"], ebm["alpha_boa_ocean"]  - era5["alpha_boa_ocean"] , color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# axs[5].legend(handles = [ebm_minus_noresm_boa_ocean_line, ebm_minus_era5_boa_ocean_line], loc = "lc",
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False,
#               bbox_to_anchor=(0.5, 0.02), ncols = 1, prop={'size':legend_fs})
            
# fig.save(cdir+"/figure3_v2.png", dpi = 400)
    
    
    
    
#------------------------------------------------------------------------------
# Print global mean air temperatures
#------------------------------------------------------------------------------


print('###########################')
print("Planetary Albedo....")
print('###########################')
print("ZEMBA: " + str(ebm["alpha_toa_global"]))
print("NorESM2: " + str(noresm["alpha_toa_global"]))
print("ERA5: " + str(era5["alpha_toa_global"]) +'\n')


print('###########################')
print("Surface Albedo....")
print('###########################')
print("ZEMBA: " + str(ebm["alpha_boa_global"]))
print("NorESM2: " + str(noresm["alpha_boa_global"]))
print("ERA5: " + str(era5["alpha_boa_global"])+'\n')



























'''
Older version of plots which include tables
'''




# #--------------
# # Plot figure 3 -- v2
# #--------------

# # constants
# #----------

# ebm_color = "black"
# ebm_lw    = 1
# ebm_ls    = "-"

# noresm_color = "blue9"
# noresm_lw    = 1
# noresm_ls    = "-"

# era5_color = "red9"
# era5_lw    = 1
# era5_ls    = "-"

# ebm_minus_noresm_color = "blue9"
# ebm_minus_noresm_lw    = 1
# ebm_minus_noresm_ls    = "-."

# ebm_minus_era5_color = "red9"
# ebm_minus_era5_lw    = 1
# ebm_minus_era5_ls    = "-."

# # formating
# #----------

# # shape
# shape = [  
#         [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#         [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#         [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#         [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],]

# # figure and axes
# fig, axs = pplt.subplots(shape, figsize = (10,6), sharey = False, sharex = False)

# # secondary yaxis
# axs_twin = axs.twinx() 

# # fonts
# axs.format(ticklabelsize=6, ticklabelweight='normal', ylabelsize=7, ylabelweight='normal',
#             xlabelsize=7, xlabelweight='normal', titlesize=10, titleweight='bold', abc='A)', 
#             abcloc='ur', abcbbox=False, xlim = (-90, 90))

# axs_twin.format(ticklabelsize=6, ticklabelweight='normal', ylabelsize=7, ylabelweight='normal',
#             xlabelsize=7, xlabelweight='normal', titlesize=10, titleweight='bold', abc='A)', 
#             abcloc='ur', abcbbox=False, xlim = (-90, 90))


# # format bottom row
# locatorx      = np.arange(-90, 120, 30)
# minorlocatorx = np.arange(-90, 100, 10)
# axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))


# # format top row
# for i in np.arange(0,5+1):
    
#     axs[i].format(xlocator = locatorx, xlim = (-90, 90))
    
#     # set y-axis range
#     axs[i].format(ylim = (-0.2, 0.9), yminorlocator=np.arange(0.1, 0.9+0.1, 0.1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(0.1, 0.9+0.1, 0.1))
    
#     # change position of yaxis label
#     axs[i].yaxis.set_label_coords(-0.15,0.65)
    
#     # remove xaxis labels for the top row
#     axs[i].grid(True)
#     axs[i].xaxis.set_ticklabels([])
#     axs[i].xaxis.set_ticks_position('none')
#     axs[i].xaxis.set_tick_params(labelbottom=False)
    
# # format bottom row
# for i in np.arange(3,5+1):
    
    
#     # set y-axis range
#     axs[i].format(ylim = (-0.3, 1), yminorlocator=np.arange(0.1, 1+0.1, 0.1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(0.1, 0.9+0.1, 0.1))
    



# # plot planetary albedo
# #----------------------

# # plot air temperature
# axs[0].plot(Var["lat"], ebm_alpha_toa, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[0].plot(noresm_annual.lat, noresm_alpha_toa, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["noresm"])
# axs[0].plot(era5_annual.lat, era5_alpha_toa, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[0:3].format(title = "Surface Average", ylabel= 'Planetary Albedo')

# # plot land temperature
# axs[1].plot(Var["lat"], ebm_alpha_toa_land, color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# axs[1].plot(noresm_annual.lat, noresm_alpha_toa_land, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# axs[1].plot(era5_annual.lat, era5_alpha_toa_land, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[1].format(title = "Land")

# # plot ocean temperature
# axs[2].plot(Var["lat"], ebm_alpha_toa_ocean, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# axs[2].plot(noresm_annual.lat, noresm_alpha_toa_ocean, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# axs[2].plot(era5_annual.lat, era5_alpha_toa_ocean, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[2].format(title = "Surface Ocean")

# # plot differences in planetary albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(0, 5+1):
#     axs_twin[i].set_yticks(np.arange(-0.2, 0.2 + 0.1, 0.1))
#     axs_twin[i].format(ylim = (-0.2, 1.2), ylabel = 'Difference in \n Planetary Albedo', yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.15,0.15)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")

# # atmosphere
# axs_twin[0].plot(Var["lat"], ebm_alpha_toa - noresm_alpha_toa, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[0].plot(Var["lat"], ebm_alpha_toa - era5_alpha_toa, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # land
# axs_twin[1].plot(Var["lat"], ebm_alpha_toa_land - noresm_alpha_toa_land, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[1].plot(Var["lat"], ebm_alpha_toa_land - era5_alpha_toa_land, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # ocean
# axs_twin[2].plot(Var["lat"], ebm_alpha_toa_ocean - noresm_alpha_toa_ocean , color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[2].plot(Var["lat"], ebm_alpha_toa_ocean  - era5_alpha_toa_ocean , color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# # plot tables for planetary albedo
# #---------------------------------

# # colors
# ebm_colour = 'warm gray'
# noresm_colour = 'blue5'
# era5_colour = 'red5'
# row_colors  = [ebm_colour, noresm_colour, era5_colour]
# cell_colors = [[ebm_colour]*6,[noresm_colour]*6,[era5_colour]*6,]

# # row names
# row_names  = ['MEBCM', 'NorESM2', "ERA5"]

# # column names
# column_names    = (r'90$^{\circ}$S - 60$^{\circ}$S', r'60$^{\circ}$S - 30$^{\circ}$S', 
#                     r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                     r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_alpha_toa_zones, noresm_alpha_toa_zones, era5_alpha_toa_zones]
# cell_text_2 = [ebm_alpha_toa_land_zones, noresm_alpha_toa_land_zones, era5_alpha_toa_land_zones]
# cell_text_3 = [ebm_alpha_toa_ocean_zones, noresm_alpha_toa_ocean_zones, era5_alpha_toa_ocean_zones]
# cell_text   = [cell_text_1, cell_text_2, cell_text_3]

# # plot and format tables
# for i in np.arange(0,2+1):
    
#     # plot table
#     table = axs[i].table(cellText=cell_text[i], rowLabels=row_names, colLabels = column_names, rowColours = row_colors, cellColours = cell_colors, cellLoc = 'center', loc = "bottom", bbox=[0., -0.5, 1, 0.5])
    
#     # change transparancy of colours
#     for cell in table._cells:
#         table._cells[cell].set_alpha(1.)
    
#     # change font size
#     table.set_fontsize(5)
    
#     # change font weight for headers
#     for (row, col), cell in table.get_celld().items():
#         if (row == 0):
#             cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            
#     for (row, col), cell in table.get_celld().items():
#         if ((row != 0) & (col != 0)):
#             cell.fontsize = 15
            

# # plot surfa albedo
# #----------------------

# # plot air temperature
# axs[3].plot(Var["lat"], ebm_alpha_boa, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[3].plot(noresm_annual.lat, noresm_alpha_boa, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["noresm"])
# axs[3].plot(era5_annual.lat, era5_alpha_boa, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])

# # plot land temperature
# axs[4].plot(Var["lat"], ebm_alpha_boa_land, color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# axs[4].plot(noresm_annual.lat, noresm_alpha_boa_land, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# axs[4].plot(era5_annual.lat, era5_alpha_boa_land, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])

# # plot ocean temperature
# axs[5].plot(Var["lat"], ebm_alpha_boa_ocean, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# axs[5].plot(noresm_annual.lat, noresm_alpha_boa_ocean, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# axs[5].plot(era5_annual.lat, era5_alpha_boa_ocean, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])

# # plot differences in surface albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(3, 5+1):
#     axs[i].format(ylabel = 'Surface Albedo')
#     axs_twin[i].format(ylabel = 'Difference in \n Surface Albedo', yminorlocator = [])
#     axs_twin[i].set_yticks(np.arange(-0.3, 0.3 + 0.1, 0.1))
#     axs_twin[i].format(ylim = (-0.3, 1.3), yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.15,0.15)
    
    
# # atmosphere
# axs_twin[3].plot(Var["lat"], ebm_alpha_boa - noresm_alpha_boa, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[3].plot(Var["lat"], ebm_alpha_boa - era5_alpha_boa, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # land
# axs_twin[4].plot(Var["lat"], ebm_alpha_boa_land - noresm_alpha_boa_land, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[4].plot(Var["lat"], ebm_alpha_boa_land - era5_alpha_boa_land, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # ocean
# axs_twin[5].plot(Var["lat"], ebm_alpha_boa_ocean - noresm_alpha_boa_ocean , color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[5].plot(Var["lat"], ebm_alpha_boa_ocean  - era5_alpha_boa_ocean , color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# # plot tables for planetary albedo
# #---------------------------------

# # colors
# ebm_colour = 'warm gray'
# noresm_colour = 'blue5'
# era5_colour = 'red5'
# row_colors  = [ebm_colour, noresm_colour, era5_colour]
# cell_colors = [[ebm_colour]*6,[noresm_colour]*6,[era5_colour]*6,]

# # row names
# row_names  = ['MEBCM', 'NorESM2', "ERA5"]

# # column names
# column_names    = (r'90$^{\circ}$S - 60$^{\circ}$S', r'60$^{\circ}$S - 30$^{\circ}$S', 
#                     r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                     r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_alpha_boa_zones, noresm_alpha_boa_zones, era5_alpha_boa_zones]
# cell_text_2 = [ebm_alpha_boa_land_zones, noresm_alpha_boa_land_zones, era5_alpha_boa_land_zones]
# cell_text_3 = [ebm_alpha_boa_ocean_zones, noresm_alpha_boa_ocean_zones, era5_alpha_boa_ocean_zones]
# cell_text   = [cell_text_1, cell_text_2, cell_text_3]

# # plot and format tables
# for i in np.arange(3,5+1):
    
#     # plot table
#     table = axs[i].table(cellText=cell_text[i-3], rowLabels=row_names, colLabels = column_names, rowColours = row_colors, cellColours = cell_colors, cellLoc = 'center', loc = "bottom", bbox=[0., -0.5, 1, 0.5])
    
#     # change transparancy of colours
#     for cell in table._cells:
#         table._cells[cell].set_alpha(1.)
    
#     # change font size
#     table.set_fontsize(5)
    
#     # change font weight for headers
#     for (row, col), cell in table.get_celld().items():
#         if (row == 0):
#             cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            
#     for (row, col), cell in table.get_celld().items():
#         if ((row != 0) & (col != 0)):
#             cell.fontsize = 15
            
# fig.save(cdir+"/figure3.png", dpi = 400)
    
    
    
    
    
    
    
    
    
    
# #--------------
# # Plot figure 3 -- v3
# #--------------

# # constants
# #----------

# ebm_color = "black"
# ebm_lw    = 1
# ebm_ls    = "-"

# noresm_color = "blue9"
# noresm_lw    = 1
# noresm_ls    = "-"

# era5_color = "red9"
# era5_lw    = 1
# era5_ls    = "-"

# ebm_minus_noresm_color = "blue9"
# ebm_minus_noresm_lw    = 1
# ebm_minus_noresm_ls    = "-."

# ebm_minus_era5_color = "red9"
# ebm_minus_era5_lw    = 1
# ebm_minus_era5_ls    = "-."

# # formating
# #----------

# # shape
# shape = [  
#         [1, 1, 1, 1, 2, 2, 2, 2],
#         [1, 1, 1, 1, 2, 2, 2, 2],
#         [1, 1, 1, 1, 2, 2, 2, 2],
#         [1, 1, 1, 1, 2, 2, 2, 2],]

# # figure and axes
# fig, axs = pplt.subplots(shape, figsize = (10,6), sharey = False, sharex = False)

# # secondary yaxis
# axs_twin = axs.twinx() 

# # fonts
# axs.format(ticklabelsize=6, ticklabelweight='normal', ylabelsize=7, ylabelweight='normal',
#             xlabelsize=7, xlabelweight='normal', titlesize=10, titleweight='bold', abc='A)', 
#             abcloc='ur', abcbbox=False, xlim = (-90, 90))

# axs_twin.format(ticklabelsize=6, ticklabelweight='normal', ylabelsize=7, ylabelweight='normal',
#             xlabelsize=7, xlabelweight='normal', titlesize=10, titleweight='bold', abc='A)', 
#             abcloc='ur', abcbbox=False, xlim = (-90, 90))


# # format bottom row
# locatorx      = np.arange(-90, 120, 30)
# minorlocatorx = np.arange(-90, 100, 10)
# axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))


# # format top row
# for i in np.arange(0,1+1):
    
#     axs[i].format(xlocator = locatorx, xlim = (-90, 90))
    
#     # set y-axis range
#     axs[i].format(ylim = (-0.2, 0.9), yminorlocator=np.arange(0.1, 0.9+0.1, 0.1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(0.1, 0.9+0.1, 0.1))
    
#     # change position of yaxis label
#     axs[i].yaxis.set_label_coords(-0.15,0.65)
    
#     # remove xaxis labels for the top row
#     axs[i].grid(True)
#     axs[i].xaxis.set_ticklabels([])
#     axs[i].xaxis.set_ticks_position('none')
#     axs[i].xaxis.set_tick_params(labelbottom=False)
    

    
# # plot planetary albedo
# #----------------------

# # plot air temperature
# axs[0].plot(Var["lat"], ebm_alpha_toa, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[0].plot(noresm_annual.lat, noresm_alpha_toa, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["noresm"])
# axs[0].plot(era5_annual.lat, era5_alpha_toa, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[0].format(title = "Planetary Albedo", ylabel= 'Albedo')



# # plot differences in planetary albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(0, 1+1):
#     axs_twin[i].set_yticks(np.arange(-0.2, 0.2 + 0.1, 0.1))
#     axs_twin[i].format(ylim = (-0.2, 1.2), ylabel = 'Difference in \n Albedo', yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.15,0.15)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")

# # atmosphere
# axs_twin[0].plot(Var["lat"], ebm_alpha_toa - noresm_alpha_toa, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[0].plot(Var["lat"], ebm_alpha_toa - era5_alpha_toa, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# # plot tables for planetary albedo
# #---------------------------------

# # colors
# ebm_colour = 'warm gray'
# noresm_colour = 'blue5'
# era5_colour = 'red5'
# row_colors  = [ebm_colour, noresm_colour, era5_colour]
# cell_colors = [[ebm_colour]*6,[noresm_colour]*6,[era5_colour]*6,]

# # row names
# row_names  = ['MEBCM', 'NorESM2', "ERA5"]

# # column names
# column_names    = (r'90$^{\circ}$S - 60$^{\circ}$S', r'60$^{\circ}$S - 30$^{\circ}$S', 
#                     r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                     r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_alpha_toa_zones, noresm_alpha_toa_zones, era5_alpha_toa_zones]
# cell_text   = [cell_text_1]

# # plot and format tables
# for i in np.arange(0,1):
    
#     # plot table
#     table = axs[i].table(cellText=cell_text[i], rowLabels=row_names, colLabels = column_names, rowColours = row_colors, cellColours = cell_colors, cellLoc = 'center', loc = "bottom", bbox=[0., -0.5, 1, 0.5])
    
#     # change transparancy of colours
#     for cell in table._cells:
#         table._cells[cell].set_alpha(1.)
    
#     # change font size
#     table.set_fontsize(5)
    
#     # change font weight for headers
#     for (row, col), cell in table.get_celld().items():
#         if (row == 0):
#             cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            
#     for (row, col), cell in table.get_celld().items():
#         if ((row != 0) & (col != 0)):
#             cell.fontsize = 15
            

# # plot surfa albedo
# #----------------------

# # plot air temperature
# axs[1].plot(Var["lat"], ebm_alpha_boa, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[1].plot(noresm_annual.lat, noresm_alpha_boa, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["noresm"])
# axs[1].plot(era5_annual.lat, era5_alpha_boa, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[1].format(title = "Surface Albedo")


# # plot differences in surface albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(1, 1):
#     axs[i].format(ylabel = 'Surface Albedo')
#     axs_twin[i].format(ylabel = 'Difference in \n Surface Albedo', yminorlocator = [])
#     axs_twin[i].set_yticks(np.arange(-0.3, 0.3 + 0.1, 0.1))
#     axs_twin[i].format(ylim = (-0.3, 1.3), yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.15,0.15)
    
    
# # atmosphere
# axs_twin[1].plot(Var["lat"], ebm_alpha_boa - noresm_alpha_boa, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[1].plot(Var["lat"], ebm_alpha_boa - era5_alpha_boa, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # plot tables for surface albedo
# #-------------------------------

# # colors
# ebm_colour = 'warm gray'
# noresm_colour = 'blue5'
# era5_colour = 'red5'
# row_colors  = [ebm_colour, noresm_colour, era5_colour]
# cell_colors = [[ebm_colour]*6,[noresm_colour]*6,[era5_colour]*6,]

# # row names
# row_names  = ['MEBCM', 'NorESM2', "ERA5"]

# # column names
# column_names    = (r'90$^{\circ}$S - 60$^{\circ}$S', r'60$^{\circ}$S - 30$^{\circ}$S', 
#                     r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                     r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_alpha_boa_zones, noresm_alpha_boa_zones, era5_alpha_boa_zones]
# cell_text   = [cell_text_1]

# # plot and format tables
# for i in np.arange(0,1):
    
#     # plot table
#     table = axs[i+1].table(cellText=cell_text[i], rowLabels=row_names, colLabels = column_names, rowColours = row_colors, cellColours = cell_colors, cellLoc = 'center', loc = "bottom", bbox=[0., -0.5, 1, 0.5])
    
#     # change transparancy of colours
#     for cell in table._cells:
#         table._cells[cell].set_alpha(1.)
    
#     # change font size
#     table.set_fontsize(5)
    
#     # change font weight for headers
#     for (row, col), cell in table.get_celld().items():
#         if (row == 0):
#             cell.set_text_props(fontproperties=FontProperties(weight='bold'))
            
#     for (row, col), cell in table.get_celld().items():
#         if ((row != 0) & (col != 0)):
#             cell.fontsize = 15

# fig.save(cdir+"/figure3_v2.png", dpi = 400)


















#----------------------------------
# Compare data for planetary albedo
#----------------------------------

# fig2, axs2 = pplt.subplots(ncols=3, nrows = 2, sharex = False, sharey = False, suptitle = "Planetary albedo")
# lons, lats = np.meshgrid(noresm2_annual.lon, noresm2_annual.lat)
# axs2[0,0].contourf(lons, lats, noresm2_alpha_toa, vmin=0, vmax=1, cmap='RdBu', colorbar='r')
# axs2[0,1].contourf(lons, lats, ceres_alpha_toa, vmin=0, vmax=1, cmap='RdBu', colorbar='r')
# axs2[0,2].contourf(lons, lats, era5_alpha_toa, vmin=0, vmax=1, cmap='RdBu', colorbar='r')
# axs2[1,1].contourf(lons, lats, noresm2_alpha_toa-ceres_alpha_toa, vmin=-0.3, vmax=0.3, cmap='RdBu', colorbar='r')
# axs2[1,2].contourf(lons, lats, noresm2_alpha_toa-era5_alpha_toa, vmin=-0.3, vmax=0.3, cmap='RdBu', colorbar='r')

# axs2[1,0].plot(noresm2_annual.lat, noresm2_alpha_toa, label = "NorESM2")
# axs2[1,0].plot(noresm2_annual.lat, ceres_alpha_toa, label = "CERES (2005-2015)")
# axs2[1,0].plot(noresm2_annual.lat, era5_alpha_toa, label = "ERA5 (1940-1970)")
# axs2.legend(frameon = False, loc = 'uc', ncols = 1)

# axs2[0,0].format(title = "NorESM2")
# axs2[0,1].format(title = "CERES (2005-2015)")
# axs2[0,2].format(title = "ERA5 (1940-1970)")
# axs2[1,0].format(title = "Zonal Mean")
# axs2[1,1].format(title = "NorESM2 - CERES (2005-2015)")
# axs2[1,2].format(title = "NorESM2 - ERA5 (1940-1970)")

#----------------------------------
# Compare data for surface albedo
#----------------------------------

# fig3, axs3 = pplt.subplots(ncols=3, nrows = 2, sharex = False, sharey = False, suptitle = "Surface albedo")
# lons, lats = np.meshgrid(noresm2_annual.lon, noresm2_annual.lat)
# axs3[0,0].contourf(lons, lats, noresm2_alpha_boa, vmin=0, vmax=1, cmap='RdBu', colorbar='r')
# axs3[0,1].contourf(lons, lats, ceres_alpha_boa, vmin=0, vmax=1, cmap='RdBu', colorbar='r')
# axs3[0,2].contourf(lons, lats, era5_alpha_boa, vmin=0, vmax=1, cmap='RdBu', colorbar='r')
# axs3[1,1].contourf(lons, lats, noresm2_alpha_boa-ceres_alpha_boa, vmin=-0.5, vmax=0.5, cmap='RdBu', colorbar='r')
# axs3[1,2].contourf(lons, lats, noresm2_alpha_boa-era5_alpha_boa, vmin=-0.5, vmax=0.5, cmap='RdBu', colorbar='r')

# axs3[1,0].plot(noresm2_annual.lat, noresm2_alpha_boa, label = "NorESM2")
# axs3[1,0].plot(noresm2_annual.lat, ceres_alpha_boa, label = "CERES (2005-2015)")
# axs3[1,0].plot(noresm2_annual.lat, era5_alpha_boa, label = "ERA5 (1940-1970)")
# axs3.legend(frameon = False, loc = 'uc', ncols = 1)

# axs3[0,0].format(title = "NorESM2")
# axs3[0,1].format(title = "CERES (2005-2015)")
# axs3[0,2].format(title = "ERA5 (1940-1970)")
# axs3[1,0].format(title = "Zonal Mean")
# axs3[1,1].format(title = "NorESM2 - CERES (2005-2015)")
# axs3[1,2].format(title = "NorESM2 - ERA5 (1940-1970)")


#-------------------------------------
# Calculate the global average albedos
#-------------------------------------

# Var = get_constants(topography = "PI", resolution = 1., nyrs= 3000.) 

# # surface
# noresm2_alpha_boa_global = np.round(global_mean(noresm2_alpha_boa.to_numpy(), Var),2)
# era5_alpha_boa_global = np.round(global_mean(era5_alpha_boa.to_numpy(), Var), 2)

# # planetary
# noresm2_alpha_toa_global = np.round(global_mean(noresm2_alpha_toa.to_numpy(), Var), 2)
# era5_alpha_toa_global = np.round(global_mean(era5_alpha_toa.to_numpy(), Var), 2)


