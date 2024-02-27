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
noresm2_annual = xr.open_dataset(os.getcwd() +  "/other_data/noresm2/noresm2_annual.nc")
noresm2_annual = noresm2_annual.interp(lat = Var["lat"]) # interpolate to EBM

# annual mean
#------------

noresm2_precip_spatial  = (noresm2_annual["pr"]/1000)*(60*60*24)*1000      # precipitation
noresm2_sf_spatial      = (noresm2_annual["prsn"]/1000)*(60*60*24)*1000    # snowfall
noresm2_evap_spatial    = (noresm2_annual["evspsbl"]/1000)*(60*60*24)*1000 # evaporation


# annual and zonal mean
#----------------------

# average
noresm2_precip  = noresm2_precip_spatial.mean(dim="lon", skipna=True)  # precipitation
noresm2_precip_global = global_mean2(noresm2_precip.to_numpy(), Var["lat"], Var["dlat"])


noresm2_sf  = noresm2_sf_spatial.mean(dim="lon", skipna=True)  # precipitation
noresm2_sf_global = global_mean2(noresm2_sf.to_numpy(), Var["lat"], Var["dlat"])

noresm2_evap    = noresm2_evap_spatial.mean(dim="lon", skipna=True)    # evaporation
noresm2_evap_global = global_mean2(noresm2_evap.to_numpy(), Var["lat"], Var["dlat"])

# land
noresm2_precip_land    = (noresm2_precip_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)  # precipitation
noresm2_evap_land      = (noresm2_evap_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)    # evaporation

# ocean
noresm2_precip_ocean    = (noresm2_precip_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)  # precipitation
noresm2_evap_ocean      = (noresm2_evap_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)    # evaporation


#---------------------------
# ERA5 1940-1970 climatology
#---------------------------

# load data
era5_annual = xr.open_dataset(os.getcwd() +  "/other_data/era5/hydrological_era5_1940_1970.nc")

# interpolate to NorESM2 grid.
era5_annual = era5_annual.interp(longitude = noresm2_annual.lon, latitude = noresm2_annual.lat) 

# annual mean
#------------

era5_precip_spatial   = (era5_annual.mtpr/1000)*(60*60*24)*1000  # precipitation
era5_sf_spatial   = (era5_annual.msr/1000)*(60*60*24)*1000  # precipitation
era5_evap_spatial     = -(era5_annual.mer/1000)*(60*60*24)*1000  # evaporation


# annual and zonal mean
#----------------------

# average
era5_precip  = era5_precip_spatial.mean(dim="lon", skipna=True)  # precipitation
era5_precip_zones = spatial_average(era5_precip, Var["lat"], np.diff(Var["lat"])[0], np.ones((Var["lat"].size))) # averaged for different zones...
era5_precip_global = global_mean2(era5_precip.to_numpy(), Var["lat"], Var["dlat"])

era5_sf  = era5_sf_spatial.mean(dim="lon", skipna=True)  # snowfall
era5_sf_zones = spatial_average(era5_sf, Var["lat"], np.diff(Var["lat"])[0], np.ones((Var["lat"].size))) # averaged for different zones...
era5_sf_global = global_mean2(era5_sf.to_numpy(), Var["lat"], Var["dlat"])

era5_evap    = era5_evap_spatial.mean(dim="lon", skipna=True)    # evaporation
era5_evap_zones = spatial_average(era5_evap, Var["lat"], np.diff(Var["lat"])[0], np.ones((Var["lat"].size))) # averaged for different zones...
era5_evap_global = global_mean2(era5_evap.to_numpy(), Var["lat"], Var["dlat"])

# land
era5_precip_land    = (era5_precip_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)  # precipitation
era5_precip_land_zones = spatial_average(era5_precip_land, Var["lat"], np.diff(Var["lat"])[0], INPUT["land_fraction"]) # averaged for different zones...


era5_evap_land      = (era5_evap_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)    # evaporation
era5_evap_land_zones = spatial_average(era5_evap_land, Var["lat"], np.diff(Var["lat"])[0], INPUT["land_fraction"]) # averaged for different zones...

# ocean

era5_precip_ocean    = (era5_precip_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)  # precipitation
era5_precip_ocean_zones = spatial_average(era5_precip_ocean, Var["lat"], np.diff(Var["lat"])[0], 1-INPUT["land_fraction"]) # averaged for different zones...


era5_evap_ocean      = (era5_evap_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)    # evaporation
era5_evap_ocean_zones = spatial_average(era5_evap_ocean, Var["lat"], np.diff(Var["lat"])[0], 1-INPUT["land_fraction"]) # averaged for different zones...


#--------------
# Plot figure 6 - v2
#--------------


# constants
#----------

ebm_color = "black"
ebm_lw    = 1
ebm_ls    = "-"

noresm2_color = "blue9"
noresm2_lw    = 1
noresm2_ls    = "-"

era5_color = "red9"
era5_lw    = 1
era5_ls    = "-"

ebm_minus_noresm2_color = "blue9"
ebm_minus_noresm2_lw    = 1
ebm_minus_noresm2_ls    = "-."

ebm_minus_era5_color = "red9"
ebm_minus_era5_lw    = 1
ebm_minus_era5_ls    = "-."

legend_fs = 6 
legend_fnt = "bold"


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
# fig, axs = pplt.subplots(shape, figsize = (12,5), sharey = False, sharex = False)

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
#     axs[i].format(ylim = (-4, 7.5), yminorlocator=np.arange(1, 7+1, 1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(1, 7+1, 1))
    
#     # change position of yaxis label
#     axs[i].yaxis.set_label_coords(-0.10,0.65)
    
    
# # format bottom row
# for i in np.arange(3,5+1):
    
    
#     axs[i].format(ylim = (-4, 5.5), yminorlocator=np.arange(0, 5+1, 1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(0, 5+1, 1))
    
    
# axs[3:].format(xlabel = "Latitude")
        



# # plot precipitation
# #-------------------

# # plot air temperature
# ebm_precip_line = axs[0].plot(Var["lat"], ebm_precip, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# noresm2_precip_line = axs[0].plot(noresm2_annual.lat, noresm2_precip, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["noresm"])
# era5_precip_line = axs[0].plot(era5_annual.lat, era5_precip, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[0].format(title = "Average", ylabel= 'Precipation (mm/day)')
# axs[0].legend(handles = [ebm_precip_line, noresm2_precip_line, era5_precip_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "ul",
#               bbox_to_anchor=(0.03, 0.97), ncols = 1, prop={'size':6})

# # plot land temperature
# ebm_precip_land_line = axs[1].plot(Var["lat"], ebm_precip_land, color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# noresm2_precip_land_line = axs[1].plot(noresm2_annual.lat, noresm2_precip_land, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["_nolabel"])
# era5_precip_land_line = axs[1].plot(era5_annual.lat, era5_precip_land, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[1].format(title = "Land")
# axs[1].legend(handles = [ebm_precip_land_line, noresm2_precip_land_line, era5_precip_land_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "ul",
#               bbox_to_anchor=(0.03, 0.97), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})

# # plot ocean temperature
# ebm_precip_ocean_line = axs[2].plot(Var["lat"], ebm_precip_ocean, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# noresm2_precip_ocean_line = axs[2].plot(noresm2_annual.lat, noresm2_precip_ocean, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["_nolabel"])
# era5_precip_ocean_line = axs[2].plot(era5_annual.lat, era5_precip_ocean, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[2].format(title = "Surface Ocean")
# axs[2].legend(handles = [ebm_precip_ocean_line, noresm2_precip_ocean_line, era5_precip_ocean_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "ul",
#               bbox_to_anchor=(0.03, 0.97), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})

# # plot differences in planetary albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(0, 5+1):
#     axs_twin[i].set_yticks(np.arange(-2, 2 + 1, 1))
#     axs_twin[i].format(ylim = (-3, 11), ylabel = 'Difference in \n Precipation (mm/day)', yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.1,0.15)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")

# # atmosphere
# ebm_minus_noresm2_precip_line = axs_twin[0].plot(Var["lat"], ebm_precip - noresm2_precip, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# ebm_minus_era5_precip_line = axs_twin[0].plot(Var["lat"], ebm_precip - era5_precip, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)
# axs[0].legend(handles = [ebm_minus_noresm2_precip_line, ebm_minus_era5_precip_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False,  loc = "ll",
#                bbox_to_anchor=(0.02, 0.02), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})
# # land
# ebm_minus_noresm2_precip_land_line = axs_twin[1].plot(Var["lat"], ebm_precip_land - noresm2_precip_land, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# ebm_minus_era5_precip_land_line = axs_twin[1].plot(Var["lat"], ebm_precip_land - era5_precip_land, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)
# axs[1].legend(handles = [ebm_minus_noresm2_precip_land_line, ebm_minus_era5_precip_land_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False,  loc = "ll",
#                bbox_to_anchor=(0.51, 0.02), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})
# # ocean
# ebm_minus_noresm2_precip_ocean_line = axs_twin[2].plot(Var["lat"], ebm_precip_ocean - noresm2_precip_ocean , color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# ebm_minus_era5_precip_ocean_line = axs_twin[2].plot(Var["lat"], ebm_precip_ocean  - era5_precip_ocean , color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)
# axs[2].legend(handles = [ebm_minus_noresm2_precip_ocean_line, ebm_minus_era5_precip_ocean_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False,  loc = "ll",
#                bbox_to_anchor=(0.02, 0.02), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})

# # plot surfa albedo
# #----------------------

# # plot air temperature
# ebm_evap_line = axs[3].plot(Var["lat"], ebm_evap, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# noresm2_evap_line = axs[3].plot(noresm2_annual.lat, noresm2_evap, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["noresm"])
# era5_evap_line = axs[3].plot(era5_annual.lat, era5_evap, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[3].legend(handles = [ebm_evap_line, noresm2_evap_line, era5_evap_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "ul",
#               bbox_to_anchor=(0.35, 0.7), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})
# # plot land temperature
# ebm_evap_land_line = axs[4].plot(Var["lat"], ebm_evap_land, color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# noresm2_evap_land_line = axs[4].plot(noresm2_annual.lat, noresm2_evap_land, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["_nolabel"])
# era5_evap_land_line = axs[4].plot(era5_annual.lat, era5_evap_land, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[4].legend(handles = [ebm_evap_land_line, noresm2_evap_land_line, era5_evap_land_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "ul",
#               bbox_to_anchor=(0.03, 0.97), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})
# # plot ocean temperature
# ebm_evap_ocean_line = axs[5].plot(Var["lat"], ebm_evap_ocean, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# noresm2_evap_ocean_line = axs[5].plot(noresm2_annual.lat, noresm2_evap_ocean, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["_nolabel"])
# era5_evap_ocean_line = axs[5].plot(era5_annual.lat, era5_evap_ocean, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[5].legend(handles = [ebm_evap_ocean_line, noresm2_evap_ocean_line, era5_evap_ocean_line],
#               labels  = ["PyEBM", "NorESM2", "ERA5 (1940 -1970)"], frameon = False, loc = "ul",
#               bbox_to_anchor=(0.35, 0.7), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})
# # plot differences in surface albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(3, 5+1):
#     axs_twin[i].format(ylabel = 'Difference in \n Precipation (mm/day)', yminorlocator = [])
#     axs_twin[i].set_yticks(np.arange(-2, 2 + 1, 1))
#     axs_twin[i].format(ylim = (-3, 11), yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.15,0.15)
# axs[3].format(ylabel = 'Evaporation (mm/day)')
    
    
# # atmosphere
# ebm_minus_noresm2_evap_line = axs_twin[3].plot(Var["lat"], ebm_evap - noresm2_evap, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# ebm_minus_era5_evap_line = axs_twin[3].plot(Var["lat"], ebm_evap - era5_evap, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)
# axs[3].legend(handles = [ebm_minus_noresm2_evap_line, ebm_minus_era5_evap_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False,  loc = "ll",
#                bbox_to_anchor=(0.01, 0.02), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})
# # land
# ebm_minus_noresm2_evap_land_line = axs_twin[4].plot(Var["lat"], ebm_evap_land - noresm2_evap_land, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# ebm_minus_era5_evap_land_line = axs_twin[4].plot(Var["lat"], ebm_evap_land - era5_evap_land, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)
# axs[4].legend(handles = [ebm_minus_noresm2_evap_land_line, ebm_minus_era5_evap_land_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False,  loc = "ll",
#                bbox_to_anchor=(0.01, 0.02), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})
# # ocean
# ebm_minus_noresm2_evap_ocean_line = axs_twin[5].plot(Var["lat"], ebm_evap_ocean - noresm2_evap_ocean , color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# ebm_minus_era5_evap_ocean_line = axs_twin[5].plot(Var["lat"], ebm_evap_ocean  - era5_evap_ocean , color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)
# axs[5].legend(handles = [ebm_minus_noresm2_evap_ocean_line, ebm_minus_era5_evap_ocean_line],
#               labels  = ["PyEBM -NorESM2", "PyEBM - ERA5 (1940 -1970)"], frameon = False,  loc = "ll",
#                bbox_to_anchor=(0.01, 0.02), ncols = 1, prop={'size':legend_fs, 'weight':legend_fnt})

            
# fig.save(cdir+"/figure6_v2.png", dpi = 400)




#--------------
# Plot figure 6 - v1
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
# ebm_precip_hadley_off_line = axs[0].plot(Var["lat"], ebm_precip_hadley_off, color = ebm_color, lw = ebm_lw, linestyle = ":", label = ["EBCM"])
noresm2_precip_line = axs[0].plot(noresm2_annual.lat, noresm2_precip, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls)
era5_precip_line = axs[0].plot(era5_annual.lat, era5_precip, color = era5_color, lw = era5_lw, linestyle = era5_ls)

ebm_sf_line = axs[0].plot(Var["lat"], ebm_sf, color = ebm_color, lw = ebm_lw, linestyle = ":", label = ["EBCM"])
noresm2_sf_line = axs[0].plot(noresm2_annual.lat, noresm2_sf, color = noresm2_color, lw = noresm2_lw, linestyle = ":", label = ["noresm"])
era5_sf_line = axs[0].plot(era5_annual.lat, era5_sf, color = era5_color, lw = era5_lw, linestyle = ":", label = ["ERA5 (1940-1970)"])

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
noresm2_evap_line = axs[1].plot(noresm2_annual.lat, noresm2_evap, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["noresm"])
era5_evap_line = axs[1].plot(era5_annual.lat, era5_evap, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])

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









'''
Old figures with tables
'''



# #--------------
# # Plot figure 6 - v2
# #--------------


# # constants
# #----------

# ebm_color = "black"
# ebm_lw    = 1
# ebm_ls    = "-"

# noresm2_color = "blue9"
# noresm2_lw    = 1
# noresm2_ls    = "-"

# era5_color = "red9"
# era5_lw    = 1
# era5_ls    = "-"

# ebm_minus_noresm2_color = "blue9"
# ebm_minus_noresm2_lw    = 1
# ebm_minus_noresm2_ls    = "-."

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
#     axs[i].format(ylim = (-4, 7.5), yminorlocator=np.arange(1, 7+1, 1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(1, 7+1, 1))
    
#     # change position of yaxis label
#     axs[i].yaxis.set_label_coords(-0.15,0.65)
    
#     # remove xaxis labels for the top row
#     axs[i].grid(True)
#     axs[i].xaxis.set_ticklabels([])
#     axs[i].xaxis.set_ticks_position('none')
#     axs[i].xaxis.set_tick_params(labelbottom=False)
    
# # format bottom row
# for i in np.arange(3,5+1):
    
    
#     axs[i].format(ylim = (-4, 5.5), yminorlocator=np.arange(1, 5+1, 1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(1, 5+1, 1))
        



# # plot precipitation
# #-------------------

# # plot air temperature
# axs[0].plot(Var["lat"], ebm_precip, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[0].plot(noresm2_annual.lat, noresm2_precip, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["noresm"])
# axs[0].plot(era5_annual.lat, era5_precip, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[0:3].format(title = "Average", ylabel= 'Precipation (mm/day)')

# # plot land temperature
# axs[1].plot(Var["lat"], ebm_precip_land, color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# axs[1].plot(noresm2_annual.lat, noresm2_precip_land, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["_nolabel"])
# axs[1].plot(era5_annual.lat, era5_precip_land, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[1].format(title = "Land")

# # plot ocean temperature
# axs[2].plot(Var["lat"], ebm_precip_ocean, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# axs[2].plot(noresm2_annual.lat, noresm2_precip_ocean, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["_nolabel"])
# axs[2].plot(era5_annual.lat, era5_precip_ocean, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])
# axs[2].format(title = "Surface Ocean")

# # plot differences in planetary albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(0, 5+1):
#     axs_twin[i].set_yticks(np.arange(-2, 2 + 1, 1))
#     axs_twin[i].format(ylim = (-3, 11), ylabel = 'Difference in \n Precipation (mm/day)', yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.15,0.15)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")

# # atmosphere
# axs_twin[0].plot(Var["lat"], ebm_precip - noresm2_precip, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# axs_twin[0].plot(Var["lat"], ebm_precip - era5_precip, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # land
# axs_twin[1].plot(Var["lat"], ebm_precip_land - noresm2_precip_land, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# axs_twin[1].plot(Var["lat"], ebm_precip_land - era5_precip_land, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # ocean
# axs_twin[2].plot(Var["lat"], ebm_precip_ocean - noresm2_precip_ocean , color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# axs_twin[2].plot(Var["lat"], ebm_precip_ocean  - era5_precip_ocean , color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# # plot tables for planetary albedo
# #---------------------------------

# # colors
# ebm_colour = 'warm gray'
# noresm2_colour = 'blue5'
# era5_colour = 'red5'
# row_colors  = [ebm_colour, noresm2_colour, era5_colour]
# cell_colors = [[ebm_colour]*6,[noresm2_colour]*6,[era5_colour]*6,]

# # row names
# row_names  = ['MEBCM', 'NorESM2', "ERA5"]

# # column names
# column_names    = (r'90$^{\circ}$S - 60$^{\circ}$S', r'60$^{\circ}$S - 30$^{\circ}$S', 
#                     r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                     r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_precip_zones, noresm2_precip_zones, era5_precip_zones]
# cell_text_2 = [ebm_precip_land_zones, noresm2_precip_land_zones, era5_precip_land_zones]
# cell_text_3 = [ebm_precip_ocean_zones, noresm2_precip_ocean_zones, era5_precip_ocean_zones]
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
# axs[3].plot(Var["lat"], ebm_evap, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[3].plot(noresm2_annual.lat, noresm2_evap, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["noresm"])
# axs[3].plot(era5_annual.lat, era5_evap, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])

# # plot land temperature
# axs[4].plot(Var["lat"], ebm_evap_land, color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# axs[4].plot(noresm2_annual.lat, noresm2_evap_land, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["_nolabel"])
# axs[4].plot(era5_annual.lat, era5_evap_land, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])

# # plot ocean temperature
# axs[5].plot(Var["lat"], ebm_evap_ocean, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# axs[5].plot(noresm2_annual.lat, noresm2_evap_ocean, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["_nolabel"])
# axs[5].plot(era5_annual.lat, era5_evap_ocean, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["_nolabel"])

# # plot differences in surface albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(3, 5+1):
#     axs[i].format(ylabel = 'Precipation (mm/day)')
#     axs_twin[i].format(ylabel = 'Difference in \n Precipation (mm/day)', yminorlocator = [])
#     axs_twin[i].set_yticks(np.arange(-2, 2 + 1, 1))
#     axs_twin[i].format(ylim = (-3, 11), yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.15,0.15)
    
    
# # atmosphere
# axs_twin[3].plot(Var["lat"], ebm_evap - noresm2_evap, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# axs_twin[3].plot(Var["lat"], ebm_evap - era5_evap, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # land
# axs_twin[4].plot(Var["lat"], ebm_evap_land - noresm2_evap_land, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# axs_twin[4].plot(Var["lat"], ebm_evap_land - era5_evap_land, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # ocean
# axs_twin[5].plot(Var["lat"], ebm_evap_ocean - noresm2_evap_ocean , color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# axs_twin[5].plot(Var["lat"], ebm_evap_ocean  - era5_evap_ocean , color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# # plot tables for planetary albedo
# #---------------------------------

# # colors
# ebm_colour = 'warm gray'
# noresm2_colour = 'blue5'
# era5_colour = 'red5'
# row_colors  = [ebm_colour, noresm2_colour, era5_colour]
# cell_colors = [[ebm_colour]*6,[noresm2_colour]*6,[era5_colour]*6,]

# # row names
# row_names  = ['MEBCM', 'NorESM2', "ERA5"]

# # column names
# column_names    = (r'90$^{\circ}$S - 60$^{\circ}$S', r'60$^{\circ}$S - 30$^{\circ}$S', 
#                     r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                     r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_evap_zones, noresm2_evap_zones, era5_evap_zones]
# cell_text_2 = [ebm_evap_land_zones, noresm2_evap_land_zones, era5_evap_land_zones]
# cell_text_3 = [ebm_evap_ocean_zones, noresm2_evap_ocean_zones, era5_evap_ocean_zones]
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
            
# fig.save(cdir+"/figure6.png", dpi = 400)




# #--------------
# # Plot figure 6 - v3
# #--------------


# # constants
# #----------

# ebm_color = "black"
# ebm_lw    = 1
# ebm_ls    = "-"

# noresm2_color = "blue9"
# noresm2_lw    = 1
# noresm2_ls    = "-"

# era5_color = "red9"
# era5_lw    = 1
# era5_ls    = "-"

# ebm_minus_noresm2_color = "blue9"
# ebm_minus_noresm2_lw    = 1
# ebm_minus_noresm2_ls    = "-."

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
# for i in np.arange(0,1):
    
#     axs[i].format(xlocator = locatorx, xlim = (-90, 90))
    
#     # set y-axis range
#     axs[i].format(ylim = (-4, 7.5), yminorlocator=np.arange(1, 7+1, 1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(1, 7+1, 1))
    
#     # change position of yaxis label
#     axs[i].yaxis.set_label_coords(-0.1,0.7)
    
#     # remove xaxis labels for the top row
#     axs[i].grid(True)
#     axs[i].xaxis.set_ticklabels([])
#     axs[i].xaxis.set_ticks_position('none')
#     axs[i].xaxis.set_tick_params(labelbottom=False)
    
# # format bottom row
# for i in np.arange(1,2):
    
    
#     axs[i].format(ylim = (-4, 5.5), yminorlocator=np.arange(1, 5+1, 1))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(1, 5+1, 1))
    
#     # change position of yaxis label
#     axs[i].yaxis.set_label_coords(-0.1,0.7)
    
#     # remove xaxis labels for the top row
#     axs[i].grid(True)
#     axs[i].xaxis.set_ticklabels([])
#     axs[i].xaxis.set_ticks_position('none')
#     axs[i].xaxis.set_tick_params(labelbottom=False)
        



# # plot precipitation
# #-------------------

# # plot precipitation
# axs[0].plot(Var["lat"], ebm_precip, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[0].plot(noresm2_annual.lat, noresm2_precip, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["noresm"])
# axs[0].plot(era5_annual.lat, era5_precip, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[0].format(title = "Precipitation", ylabel= 'Preciptation (mm/day)')


# # plot differences in precipitation
# #-------------------------------------

# # format secondary axis
# for i in np.arange(0, 1):
#     axs_twin[i].set_yticks(np.arange(-2, 2 + 1, 1))
#     axs_twin[i].format(ylim = (-3, 11), ylabel = 'Difference in \n Precipation (mm/day)', yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.1,0.15)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")

# # precipitation
# axs_twin[0].plot(Var["lat"], ebm_precip - noresm2_precip, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# axs_twin[0].plot(Var["lat"], ebm_precip - era5_precip, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# # plot tables for planetary albedo
# #---------------------------------

# # colors
# ebm_colour = 'warm gray'
# noresm2_colour = 'blue5'
# era5_colour = 'red5'
# row_colors  = [ebm_colour, noresm2_colour, era5_colour]
# cell_colors = [[ebm_colour]*6,[noresm2_colour]*6,[era5_colour]*6,]

# # row names
# row_names  = ['MEBCM', 'NorESM2', "ERA5"]

# # column names
# column_names    = (r'90$^{\circ}$S - 60$^{\circ}$S', r'60$^{\circ}$S - 30$^{\circ}$S', 
#                     r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                     r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_precip_zones, noresm2_precip_zones, era5_precip_zones]
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
            

# # plot evaporation
# #------------------

# # plot air temperature
# axs[1].plot(Var["lat"], ebm_evap, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[1].plot(noresm2_annual.lat, noresm2_evap, color = noresm2_color, lw = noresm2_lw, linestyle = noresm2_ls, label = ["noresm"])
# axs[1].plot(era5_annual.lat, era5_evap, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[1].format(title = "Evaporation", ylabel= 'Evaporation (mm/day)')

# # plot differences in surface albedo
# #-------------------------------------

# # format secondary axis
# for i in np.arange(1, 2):
#     axs[i].format(ylabel = 'Evaporation (mm/day)')
#     axs_twin[i].format(ylabel = 'Difference in \n Evaporation (mm/day)', yminorlocator = [])
#     axs_twin[i].set_yticks(np.arange(-2, 2 + 1, 1))
#     axs_twin[i].format(ylim = (-3, 11), yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.1,0.15)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")
    
    
# # atmosphere
# axs_twin[1].plot(Var["lat"], ebm_evap - noresm2_evap, color = ebm_minus_noresm2_color, lw = ebm_minus_noresm2_lw, linestyle = ebm_minus_noresm2_ls)
# axs_twin[1].plot(Var["lat"], ebm_evap - era5_evap, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# # plot tables for planetary albedo
# #---------------------------------

# # colors
# ebm_colour = 'warm gray'
# noresm2_colour = 'blue5'
# era5_colour = 'red5'
# row_colors  = [ebm_colour, noresm2_colour, era5_colour]
# cell_colors = [[ebm_colour]*6,[noresm2_colour]*6,[era5_colour]*6,]

# # row names
# row_names  = ['MEBCM', 'NorESM2', "ERA5"]

# # column names
# column_names    = (r'90$^{\circ}$S - 60$^{\circ}$S', r'60$^{\circ}$S - 30$^{\circ}$S', 
#                     r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                     r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_evap_zones, noresm2_evap_zones, era5_evap_zones]
# cell_text   = [cell_text_1]

# # plot and format tables
# for i in np.arange(1,2):
    
#     # plot table
#     table = axs[i].table(cellText=cell_text[i-1], rowLabels=row_names, colLabels = column_names, rowColours = row_colors, cellColours = cell_colors, cellLoc = 'center', loc = "bottom", bbox=[0., -0.5, 1, 0.5])
    
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
            
# fig.save(cdir+"/figure6_v2.png", dpi = 400)































#-------------
# Compare data
#-------------

# function
# def compare_hydrological(noresm2, noresm2, era5, era5, vmin, vmax, vmin1, vmax1, title):

#     fig, axs = pplt.subplots(ncols=2, nrows = 2, sharex = False, sharey = False, suptitle = title)
#     lons, lats = np.meshgrid(noresm2.lon, noresm2.lat)
#     axs[0,0].contourf(lons, lats, noresm2, vmin=vmin, vmax=vmax, cmap='BuRd', colorbar='r')
#     axs[0,1].contourf(lons, lats, era5, vmin=vmin, vmax=vmax, cmap='BuRd', colorbar='r')
#     axs[1,1].contourf(lons, lats, noresm2-era5, vmin=vmin1, vmax=vmax1, cmap='BuRd', colorbar='r')
    
#     axs[1,0].plot(noresm2_annual.lat, noresm2, label = "NorESM2")
#     axs[1,0].plot(noresm2_annual.lat, era5, label = "ERA5 (1940-1970)")
#     axs.legend(frameon = False, loc = 'uc', ncols = 1)
    
#     axs[0,0].format(title = "NorESM2")
#     axs[0,1].format(title = "ERA5 (1940-1970)")
#     axs[1,0].format(title = "zones Mean")
#     axs[1,1].format(title = "NorESM2 - ERA5 (1940-1970)")
    
#     return fig, axs

# precipitation
# fig2, ax2 = compare_hydrological(noresm2_precip, noresm2_precip, era5_precip, era5_precip, vmin = 0, vmax = 15, vmin1 = -5, vmax1 = 5, title="Total Precipitation Flux")

# evaporation
# fig3, ax3 = compare_hydrological(noresm2_evap, noresm2_evap, era5_evap, era5_evap, vmin = -1, vmax = 10, vmin1 = -5, vmax1 = 5, title="Evaporation Flux")

