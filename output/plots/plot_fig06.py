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

# land
ebm_rsdtnet_land = pi['rsdtnet_land'].mean(axis=1)  # TOA net shortwave
ebm_rlut_land    = pi['rlut_land'].mean(axis=1)     # TOA outgoing longwave
ebm_rtnet_land   = pi['rtnet_land'].mean(axis=1)    # TOA net total radiation
ebm_rsdsnet_land = pi['rsdsnet_land'].mean(axis=1)  # BOA net shortwave
ebm_rlus_land    = -pi['rldsnet_land'].mean(axis=1) # BOA net outgoing longwave
ebm_rsnet_land   = pi['rsnet_land'].mean(axis=1)    # BOA net total radiation

# ocean
ebm_rsdtnet_ocean = pi['rsdtnet_ocean'].mean(axis=1)  # TOA net shortwave
ebm_rlut_ocean    = pi['rlut_ocean'].mean(axis=1)     # TOA outgoing longwave
ebm_rtnet_ocean   = pi['rtnet_ocean'].mean(axis=1)    # TOA net total radiation
ebm_rsdsnet_ocean = pi['rsdsnet_ocean'].mean(axis=1)  # BOA net shortwave
ebm_rlus_ocean    = -pi['rldsnet_ocean'].mean(axis=1) # BOA net outgoing longwave
ebm_rsnet_ocean   = pi['rsnet_ocean'].mean(axis=1)    # BOA net total radiation

#-----------------
# Add NorESM2 data
#-----------------

# load annual data
noresm2_annual = xr.open_dataset(os.getcwd() +  "/other_data/noresm2/noresm2_annual.nc")
noresm2_annual = noresm2_annual.interp(lat = Var["lat"]) 

# annual mean
#------------

noresm2_rsdtnet_spatial   = noresm2_annual["rsdt"] - noresm2_annual["rsut"]  # TOA net shortwave
noresm2_rlut_spatial      = noresm2_annual["rlut"]                           # TOA outgoing longwave
noresm2_rtnet_spatial     = noresm2_rsdtnet_spatial - noresm2_rlut_spatial                   # TOA net total radiation
noresm2_rsdsnet_spatial   = noresm2_annual["rsds"] - noresm2_annual["rsus"]  # BOA net shortwave
noresm2_rlus_spatial      = noresm2_annual["rlus"] - noresm2_annual["rlds"]  # BOA net outgoing longwave
noresm2_rsnet_spatial     = noresm2_rsdsnet_spatial - noresm2_rlus_spatial                   # BOA net total radiation

# annual and zonal mean
#----------------------

# average
noresm2_rsdtnet   = noresm2_rsdtnet_spatial.mean(dim="lon", skipna=True)  # TOA net shortwave
noresm2_rlut      = noresm2_rlut_spatial.mean(dim="lon", skipna=True)     # TOA outgoing longwave
noresm2_rtnet     = noresm2_rtnet_spatial.mean(dim="lon", skipna=True)    # TOA net total radiation
noresm2_rsdsnet   = noresm2_rsdsnet_spatial.mean(dim="lon", skipna=True)  # BOA net shortwave
noresm2_rlus      = noresm2_rlus_spatial.mean(dim="lon", skipna=True)     # BOA net outgoing longwave
noresm2_rsnet     = noresm2_rsnet_spatial.mean(dim="lon", skipna=True)    # BOA net total radiation

noresm2_rtnet_global   = global_mean2(noresm2_rtnet.to_numpy(), noresm2_rtnet.lat.to_numpy(), np.diff(noresm2_rtnet.lat.to_numpy())[0])    # global mean BOA net total radiation
noresm2_rsnet_global   = global_mean2(noresm2_rsnet.to_numpy(), noresm2_rsnet.lat.to_numpy(), np.diff(noresm2_rsnet.lat.to_numpy())[0])    # global mean BOA net total radiation


# land
noresm2_rsdtnet_land   = (noresm2_rsdtnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)  # TOA net shortwave
noresm2_rlut_land      = (noresm2_rlut_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)     # TOA outgoing longwave
noresm2_rtnet_land     = (noresm2_rtnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)    # TOA net total radiation
noresm2_rsdsnet_land   = (noresm2_rsdsnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)  # BOA net shortwave
noresm2_rlus_land      = (noresm2_rlus_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)     # BOA net outgoing longwave
noresm2_rsnet_land     = (noresm2_rsnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)    # BOA net total radiation

# ocean
noresm2_rsdtnet_ocean   = (noresm2_rsdtnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)  # TOA net shortwave
noresm2_rlut_ocean      = (noresm2_rlut_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)     # TOA outgoing longwave
noresm2_rtnet_ocean    = (noresm2_rtnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)     # TOA net total radiation
noresm2_rsdsnet_ocean  = (noresm2_rsdsnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)   # BOA net shortwave
noresm2_rlus_ocean     = (noresm2_rlus_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)      # BOA net outgoing longwave
noresm2_rsnet_ocean     = (noresm2_rsnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)    # BOA net total radiation


#----------------------------
# CERES 2005-2015 climatology
# ---------------------------

# load data
ceres = xr.open_dataset(os.getcwd() +  "/other_data/ceres/CERES_EBAF_Ed4.2_Subset_CLIM01-CLIM12 (1).nc")

# interpolate to NorESM2 grid.
ceres = ceres.interp(lon = noresm2_annual.lon) # interpolate

# annual mean
#------------

ceres_rsdtnet_spatial   = np.sum(Var["days_in_months"].reshape(12,1,1)*(ceres["solar_clim"] - ceres["toa_sw_all_clim"]), axis = 0) / np.sum(Var["days_in_months"])               # TOA net shortwave
ceres_rlut_spatial      = np.sum(Var["days_in_months"].reshape(12,1,1)*(ceres["toa_lw_all_clim"]), axis = 0) / np.sum(Var["days_in_months"])                                     # TOA outgoing longwave
ceres_rtnet_spatial     = ceres_rsdtnet_spatial - ceres_rlut_spatial                                                                                                                             # TOA net total radiation
ceres_rsdsnet_spatial   = np.sum(Var["days_in_months"].reshape(12,1,1)*(ceres["sfc_sw_down_all_clim"] - ceres["sfc_sw_up_all_clim"]), axis = 0) / np.sum(Var["days_in_months"])  # TOA net shortwave
ceres_rlus_spatial      = np.sum(Var["days_in_months"].reshape(12,1,1)*(ceres["sfc_lw_up_all_clim"] - ceres["sfc_lw_down_all_clim"]), axis = 0) / np.sum(Var["days_in_months"])  # TOA net shortwave
ceres_rsnet_spatial     = ceres_rsdsnet_spatial - ceres_rlus_spatial                                                                                                                             # BOA net total radiation

# annual and zonal mean
#----------------------

# average
ceres_rsdtnet   = ceres_rsdtnet_spatial.mean(dim="lon", skipna=True)  # TOA net shortwave
ceres_rlut      = ceres_rlut_spatial.mean(dim="lon", skipna=True)     # TOA outgoing longwave
ceres_rtnet     = ceres_rtnet_spatial.mean(dim="lon", skipna=True)    # TOA net total radiation
ceres_rsdsnet   = ceres_rsdsnet_spatial.mean(dim="lon", skipna=True)  # BOA net shortwave
ceres_rlus      = ceres_rlus_spatial.mean(dim="lon", skipna=True)     # BOA net outgoing longwave
ceres_rsnet     = ceres_rsnet_spatial.mean(dim="lon", skipna=True)     # BOA net total radiation

# land
ceres_rsdtnet_land   = (ceres_rsdtnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)  # TOA net shortwave
ceres_rlut_land      = (ceres_rlut_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)     # TOA outgoing longwave
ceres_rtnet_land     = (ceres_rtnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)    # TOA net total radiation
ceres_rsdsnet_land   = (ceres_rsdsnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)  # BOA net shortwave
ceres_rlus_land      = (ceres_rlus_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)     # BOA net outgoing longwave
ceres_rsnet_land     = (ceres_rsnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)    # BOA net total radiation

# ocean
ceres_rsdtnet_ocean   = (ceres_rsdtnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)  # TOA net shortwave
ceres_rlut_ocean      = (ceres_rlut_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)     # TOA outgoing longwave
ceres_rtnet_ocean     = (ceres_rtnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)    # TOA net total radiation
ceres_rsdsnet_ocean   = (ceres_rsdsnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)  # BOA net shortwave
ceres_rlus_ocean      = (ceres_rlus_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)     # BOA net outgoing longwave
ceres_rsnet_ocean     = (ceres_rsnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)    # BOA net total radiation


#---------------------------
# ERA5 1940-1970 climatology
#---------------------------

# load data
era5 = xr.open_dataset(os.getcwd() +  "/other_data/era5/radiation_era5_1940_1970.nc")

# interpolate to NorESM2 grid.
era5 = era5.interp(longitude = noresm2_annual.lon, latitude = noresm2_annual.lat) 

# annual mean
#------------

era5_rsdtnet_spatial   = era5.mtnswrf              # TOA net shortwave
era5_rlut_spatial      = -era5.mtnlwrf             # TOA outgoing longwave
era5_rtnet_spatial     = era5_rsdtnet_spatial - era5_rlut_spatial  # TOA net total radiation
era5_rsdsnet_spatial   = era5.msnswrf              # BOA net shortwave
era5_rlus_spatial      = -era5.msnlwrf             # BOA net outgoing longwave
era5_rsnet_spatial     = era5_rsdsnet_spatial - era5_rlus_spatial  # BOA net total radiation

# annual and zonal mean
#----------------------

# average
era5_rsdtnet   = era5_rsdtnet_spatial.mean(dim="lon", skipna=True)  # TOA net shortwave
era5_rlut      = era5_rlut_spatial.mean(dim="lon", skipna=True)     # TOA outgoing longwave
era5_rtnet     = era5_rtnet_spatial.mean(dim="lon", skipna=True)    # TOA net total radiation
era5_rsdsnet   = era5_rsdsnet_spatial.mean(dim="lon", skipna=True)  # BOA net shortwave
era5_rlus      = era5_rlus_spatial.mean(dim="lon", skipna=True)     # BOA net outgoing longwave
era5_rsnet     = era5_rsnet_spatial.mean(dim="lon", skipna=True)     # BOA net total radiation

era5_rtnet_global   = global_mean2(era5_rtnet.to_numpy(), era5_rtnet.lat.to_numpy(), np.diff(era5_rtnet.lat.to_numpy())[0])    # global mean BOA net total radiation
era5_rsnet_global   = global_mean2(era5_rsnet.to_numpy(), era5_rsnet.lat.to_numpy(), np.diff(era5_rsnet.lat.to_numpy())[0])    # global mean BOA net total radiation


# land
era5_rsdtnet_land   = (era5_rsdtnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)  # TOA net shortwave
era5_rlut_land      = (era5_rlut_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)     # TOA outgoing longwave
era5_rtnet_land     = (era5_rtnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)    # TOA net total radiation
era5_rsdsnet_land   = (era5_rsdsnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)  # BOA net shortwave
era5_rlus_land      = (era5_rlus_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)     # BOA net outgoing longwave
era5_rsnet_land     = (era5_rsnet_spatial*noresm2_annual["land_mask"]).mean(dim="lon", skipna=True)    # BOA net total radiation

# ocean
era5_rsdtnet_ocean   = (era5_rsdtnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)  # TOA net shortwave
era5_rlut_ocean      = (era5_rlut_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)     # TOA outgoing longwave
era5_rtnet_ocean     = (era5_rtnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)    # TOA net total radiation
era5_rsdsnet_ocean   = (era5_rsdsnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)  # BOA net shortwave
era5_rlus_ocean      = (era5_rlus_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)     # BOA net outgoing longwave
era5_rsnet_ocean     = (era5_rsnet_spatial*noresm2_annual["ocean_mask"]).mean(dim="lon", skipna=True)    # BOA net total radiation




#--------------
# Plot figure 5 -- v1
#--------------

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
noresm_boa_olw = axs[1].plot(Var["lat"], noresm2_rlus, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
noresm_boa_net = axs[1].plot(Var["lat"], noresm2_rsnet, color = noresm_color, lw = noresm_lw, ls = net_ls)    # Net radiation at boa

era5_boa_asr = axs[1].plot(Var["lat"], era5_rsdsnet, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at boa
era5_boa_olw = axs[1].plot(Var["lat"], era5_rlus, color = era5_color, lw = era5_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
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





# #--------------
# # Plot figure 5 -- v2
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

# asr_ls = "-"
# olw_ls = ":"
# net_ls = "-."

# # legend font size
# legend_fs = 6

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
# fig, axs = pplt.subplots(shape, figsize = (12,7), sharey = False, sharex = False)


# # fonts
# axs.format(ticklabelsize=6, ticklabelweight='normal', ylabelsize=10, ylabelweight='bold',
#             xlabelsize=7, xlabelweight='normal', titlesize=10, titleweight='bold', abc='A)', 
#             abcloc='ur', abcbbox=False, xlim = (-90, 90))


# # format bottom row
# locatorx      = np.arange(-90, 120, 30)
# minorlocatorx = np.arange(-90, 100, 10)
# axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))


# # format top row
# for i in np.arange(0,3):
    
#     axs[i].format(xlocator = locatorx, xlim = (-90, 90))
    
#     # set y-axis range
#     axs[i].format(ylim = (-150, 350), yminorlocator=np.arange(-150, 350+50, 50))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(-150, 350+50, 50))

    

# for i in np.arange(3,6):
    
#     # set y-axis range
#     axs[i].format(ylim = (-50, 250), yminorlocator=np.arange(-50, 250+50, 50))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(-50, 250+50, 50))


# axs[3:6].format(xlabel = "Latitude")

    
# # plot TOA radiative fluxes
# #--------------------------

# # average
 
# ebm_toa_asr = axs[0].plot(Var["lat"], ebm_rsdtnet, color = ebm_color, lw = ebm_lw, ls = asr_ls) # Net shortwave at TOA
# ebm_toa_olw = axs[0].plot(Var["lat"], ebm_rlut, color = ebm_color, lw = ebm_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# ebm_toa_net = axs[0].plot(Var["lat"], ebm_rtnet, color = ebm_color, lw = ebm_lw, ls = net_ls)   # Net radiation at TOA

# noresm_toa_asr = axs[0].plot(Var["lat"], noresm2_rsdtnet, color = noresm_color, lw = noresm_lw, ls = asr_ls)  # Net shortwave at TOA
# noresm_toa_olw = axs[0].plot(Var["lat"], noresm2_rlut, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# noresm_toa_net = axs[0].plot(Var["lat"], noresm2_rtnet, color = noresm_color, lw = noresm_lw, ls = net_ls)    # Net radiation at TOA

# era5_toa_asr = axs[0].plot(Var["lat"], era5_rsdtnet, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at TOA
# era5_toa_olw = axs[0].plot(Var["lat"], era5_rlut, color = era5_color, lw = era5_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# era5_toa_net = axs[0].plot(Var["lat"], era5_rtnet, color = era5_color, lw = era5_lw, ls = net_ls)    # Net radiation at TOA

# # title and label
# axs[0].format(title = "Average", ylabel= 'TOA Radiative Flux (W m$^{-2}$)')

# # legend
# axs[0].legend(handles = [ebm_toa_asr, noresm_toa_asr, era5_toa_asr],
#               labels  = ["ASR-PyEBM", "ASR-NorESM2", "ASR-ERA5"], frameon = False,
#               loc = "ul", bbox_to_anchor=(0.1, 0.97), ncols = 1, prop={'size':legend_fs})  

# axs[0].legend(handles = [ebm_toa_olw, noresm_toa_olw, era5_toa_olw],
#               labels  = ["OLR-PyEBM", "OLR-NorESM2", "OLR-ERA5"], frameon = False,
#               loc = "c", bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})  

# axs[0].legend(handles = [ebm_toa_net, noresm_toa_net, era5_toa_net],
#               labels  = ["Net-PyEBM", "Net-NorESM2", "Net-ERA5"], frameon = False,
#               loc = "lc", bbox_to_anchor=(0.5, 0.1), ncols = 1, prop={'size':legend_fs})  

# # land
 
# ebm_toa_asr_land = axs[1].plot(Var["lat"], ebm_rsdtnet_land, color = ebm_color, lw = ebm_lw, ls = asr_ls) # Net shortwave at TOA
# ebm_toa_olw_land = axs[1].plot(Var["lat"], ebm_rlut_land, color = ebm_color, lw = ebm_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# ebm_toa_net_land = axs[1].plot(Var["lat"], ebm_rtnet_land, color = ebm_color, lw = ebm_lw, ls = net_ls)   # Net radiation at TOA

# noresm_toa_asr_land = axs[1].plot(Var["lat"], noresm2_rsdtnet_land, color = noresm_color, lw = noresm_lw, ls = asr_ls)  # Net shortwave at TOA
# noresm_toa_olw_land = axs[1].plot(Var["lat"], noresm2_rlut_land, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# noresm_toa_net_land = axs[1].plot(Var["lat"], noresm2_rtnet_land, color = noresm_color, lw = noresm_lw, ls = net_ls)    # Net radiation at TOA

# era5_toa_asr_land = axs[1].plot(Var["lat"], era5_rsdtnet_land, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at TOA
# era5_toa_olw_land = axs[1].plot(Var["lat"], era5_rlut_land, color = era5_color, lw = era5_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# era5_toa_net_land = axs[1].plot(Var["lat"], era5_rtnet_land, color = era5_color, lw = era5_lw, ls = net_ls)    # Net radiation at TOA

# # title and label
# axs[1].format(title = "Land",)

# # legend
# axs[1].legend(handles = [ebm_toa_asr_land, noresm_toa_asr_land, era5_toa_asr_land],
#               labels  = ["ASR-PyEBM", "ASR-NorESM2", "ASR-ERA5"], frameon = False,
#               loc = "ul", bbox_to_anchor=(0.1, 0.97), ncols = 1, prop={'size':legend_fs})  

# axs[1].legend(handles = [ebm_toa_olw_land, noresm_toa_olw_land, era5_toa_olw_land],
#               labels  = ["OLR-PyEBM", "OLR-NorESM2", "OLR-ERA5"], frameon = False,
#               loc = "c", bbox_to_anchor=(0.5, 0.6), ncols = 1, prop={'size':legend_fs})  

# axs[1].legend(handles = [ebm_toa_net_land, noresm_toa_net_land, era5_toa_net_land],
#               labels  = ["Net-PyEBM", "Net-NorESM2", "Net-ERA5"], frameon = False,
#               loc = "lc", bbox_to_anchor=(0.5, 0.1), ncols = 1, prop={'size':legend_fs})


# # ocean
 
# ebm_toa_asr_ocean = axs[2].plot(Var["lat"], ebm_rsdtnet_ocean, color = ebm_color, lw = ebm_lw, ls = asr_ls) # Net shortwave at TOA
# ebm_toa_olw_ocean = axs[2].plot(Var["lat"], ebm_rlut_ocean, color = ebm_color, lw = ebm_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# ebm_toa_net_ocean = axs[2].plot(Var["lat"], ebm_rtnet_ocean, color = ebm_color, lw = ebm_lw, ls = net_ls)   # Net radiation at TOA

# noresm_toa_asr_ocean = axs[2].plot(Var["lat"], noresm2_rsdtnet_ocean, color = noresm_color, lw = noresm_lw, ls = asr_ls)  # Net shortwave at TOA
# noresm_toa_olw_ocean = axs[2].plot(Var["lat"], noresm2_rlut_ocean, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# noresm_toa_net_ocean = axs[2].plot(Var["lat"], noresm2_rtnet_ocean, color = noresm_color, lw = noresm_lw, ls = net_ls)    # Net radiation at TOA

# era5_toa_asr_ocean = axs[2].plot(Var["lat"], era5_rsdtnet_ocean, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at TOA
# era5_toa_olw_ocean = axs[2].plot(Var["lat"], era5_rlut_ocean, color = era5_color, lw = era5_lw, ls = olw_ls)     # Outgoing longwave radiation at TOA
# era5_toa_net_ocean = axs[2].plot(Var["lat"], era5_rtnet_ocean, color = era5_color, lw = era5_lw, ls = net_ls)    # Net radiation at TOA

# # title and label
# axs[2].format(title = "Ocean",)

# # legend
# axs[2].legend(handles = [ebm_toa_asr_ocean, noresm_toa_asr_ocean, era5_toa_asr_ocean],
#               labels  = ["ASR-PyEBM", "ASR-NorESM2", "ASR-ERA5"], frameon = False,
#               loc = "ul", bbox_to_anchor=(0.1, 0.97), ncols = 1, prop={'size':legend_fs})  

# axs[2].legend(handles = [ebm_toa_olw_ocean, noresm_toa_olw_ocean, era5_toa_olw_ocean],
#               labels  = ["OLR-PyEBM", "OLR-NorESM2", "OLR-ERA5"], frameon = False,
#               loc = "c", bbox_to_anchor=(0.5, 0.7), ncols = 1, prop={'size':legend_fs})  

# axs[2].legend(handles = [ebm_toa_net_ocean, noresm_toa_net_ocean, era5_toa_net_ocean],
#               labels  = ["Net-PyEBM", "Net-NorESM2", "Net-ERA5"], frameon = False,
#               loc = "lc", bbox_to_anchor=(0.5, 0.1), ncols = 1, prop={'size':legend_fs})




# # plot SFC radiative fluxes
# #--------------------------

# ebm_boa_asr = axs[3].plot(Var["lat"], ebm_rsdsnet, color = ebm_color, lw = ebm_lw, ls = asr_ls) # Net shortwave at boa
# ebm_boa_olw = axs[3].plot(Var["lat"], ebm_rlus, color = ebm_color, lw = ebm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# ebm_boa_net = axs[3].plot(Var["lat"], ebm_rsnet, color = ebm_color, lw = ebm_lw, ls = net_ls)   # Net radiation at boa

# noresm_boa_asr = axs[3].plot(Var["lat"], noresm2_rsdsnet, color = noresm_color, lw = noresm_lw, ls = asr_ls)  # Net shortwave at boa
# noresm_boa_olw = axs[3].plot(Var["lat"], noresm2_rlus, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# noresm_boa_net = axs[3].plot(Var["lat"], noresm2_rsnet, color = noresm_color, lw = noresm_lw, ls = net_ls)    # Net radiation at boa

# era5_boa_asr = axs[3].plot(Var["lat"], era5_rsdsnet, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at boa
# era5_boa_olw = axs[3].plot(Var["lat"], era5_rlus, color = era5_color, lw = era5_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# era5_boa_net = axs[3].plot(Var["lat"], era5_rsnet, color = era5_color, lw = era5_lw, ls = net_ls)    # Net radiation at boa

# axs[3].format(ylabel= 'Surfae Radiative Flux (W m$^{-2}$)')

# # legend
# axs[3].legend(handles = [ebm_boa_asr, noresm_boa_asr, era5_boa_asr],
#               labels  = ["ASR-PyEBM", "ASR-NorESM2", "ASR-ERA5"], frameon = False,
#               loc = "ul", bbox_to_anchor=(0.1, 0.97), ncols = 1, prop={'size':legend_fs})  

# axs[3].legend(handles = [ebm_boa_olw, noresm_boa_olw, era5_boa_olw],
#               labels  = ["OLR-PyEBM", "OLR-NorESM2", "OLR-ERA5"], frameon = False,
#               loc = "c", bbox_to_anchor=(0.5, 0.2), ncols = 1, prop={'size':legend_fs})  

# axs[3].legend(handles = [ebm_boa_net, noresm_boa_net, era5_boa_net],
#               labels  = ["Net-PyEBM", "Net-NorESM2", "Net-ERA5"], frameon = False,
#               loc = "lc", bbox_to_anchor=(0.5, 0.45), ncols = 1, prop={'size':legend_fs})  


# # land
 
# ebm_boa_asr_land = axs[4].plot(Var["lat"], ebm_rsdsnet_land, color = ebm_color, lw = ebm_lw, ls = asr_ls) # Net shortwave at boa
# ebm_boa_olw_land = axs[4].plot(Var["lat"], ebm_rlus_land, color = ebm_color, lw = ebm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# ebm_boa_net_land = axs[4].plot(Var["lat"], ebm_rsnet_land, color = ebm_color, lw = ebm_lw, ls = net_ls)   # Net radiation at boa

# noresm_boa_asr_land = axs[4].plot(Var["lat"], noresm2_rsdsnet_land, color = noresm_color, lw = noresm_lw, ls = asr_ls)  # Net shortwave at boa
# noresm_boa_olw_land = axs[4].plot(Var["lat"], noresm2_rlus_land, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# noresm_boa_net_land = axs[4].plot(Var["lat"], noresm2_rsnet_land, color = noresm_color, lw = noresm_lw, ls = net_ls)    # Net radiation at boa

# era5_boa_asr_land = axs[4].plot(Var["lat"], era5_rsdsnet_land, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at boa
# era5_boa_olw_land = axs[4].plot(Var["lat"], era5_rlus_land, color = era5_color, lw = era5_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# era5_boa_net_land = axs[4].plot(Var["lat"], era5_rsnet_land, color = era5_color, lw = era5_lw, ls = net_ls)    # Net radiation at boa


# # legend
# axs[4].legend(handles = [ebm_boa_asr_land, noresm_boa_asr_land, era5_boa_asr_land],
#               labels  = ["ASR-PyEBM", "ASR-NorESM2", "ASR-ERA5"], frameon = False,
#               loc = "ul", bbox_to_anchor=(0.15, 0.97), ncols = 1, prop={'size':legend_fs})  

# axs[4].legend(handles = [ebm_boa_olw_land, noresm_boa_olw_land, era5_boa_olw_land],
#               labels  = ["OLR-PyEBM", "OLR-NorESM2", "OLR-ERA5"], frameon = False,
#               loc = "c", bbox_to_anchor=(0.5, 0.2), ncols = 1, prop={'size':legend_fs})  

# axs[4].legend(handles = [ebm_boa_net_land, noresm_boa_net_land, era5_boa_net_land],
#               labels  = ["Net-PyEBM", "Net-NorESM2", "Net-ERA5"], frameon = False,
#               loc = "lc", bbox_to_anchor=(0.5, 0.4), ncols = 1, prop={'size':legend_fs})


# # ocean
 
# ebm_boa_asr_ocean = axs[5].plot(Var["lat"], ebm_rsdsnet_ocean, color = ebm_color, lw = ebm_lw, ls = asr_ls) # Net shortwave at boa
# ebm_boa_olw_ocean = axs[5].plot(Var["lat"], ebm_rlus_ocean, color = ebm_color, lw = ebm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# ebm_boa_net_ocean = axs[5].plot(Var["lat"], ebm_rsnet_ocean, color = ebm_color, lw = ebm_lw, ls = net_ls)   # Net radiation at boa

# noresm_boa_asr_ocean = axs[5].plot(Var["lat"], noresm2_rsdsnet_ocean, color = noresm_color, lw = noresm_lw, ls = asr_ls)  # Net shortwave at boa
# noresm_boa_olw_ocean = axs[5].plot(Var["lat"], noresm2_rlus_ocean, color = noresm_color, lw = noresm_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# noresm_boa_net_ocean = axs[5].plot(Var["lat"], noresm2_rsnet_ocean, color = noresm_color, lw = noresm_lw, ls = net_ls)    # Net radiation at boa

# era5_boa_asr_ocean = axs[5].plot(Var["lat"], era5_rsdsnet_ocean, color = era5_color, lw = era5_lw, ls = asr_ls)  # Net shortwave at boa
# era5_boa_olw_ocean = axs[5].plot(Var["lat"], era5_rlus_ocean, color = era5_color, lw = era5_lw, ls = olw_ls)     # Outgoing longwave radiation at boa
# era5_boa_net_ocean = axs[5].plot(Var["lat"], era5_rsnet_ocean, color = era5_color, lw = era5_lw, ls = net_ls)    # Net radiation at boa

# # legend
# axs[5].legend(handles = [ebm_boa_asr_ocean, noresm_boa_asr_ocean, era5_boa_asr_ocean],
#               labels  = ["ASR-PyEBM", "ASR-NorESM2", "ASR-ERA5"], frameon = False,
#               loc = "ul", bbox_to_anchor=(0.1, 0.97), ncols = 1, prop={'size':legend_fs})  

# axs[5].legend(handles = [ebm_boa_olw_ocean, noresm_boa_olw_ocean, era5_boa_olw_ocean],
#               labels  = ["OLR-PyEBM", "OLR-NorESM2", "OLR-ERA5"], frameon = False,
#               loc = "c", bbox_to_anchor=(0.5, 0.2), ncols = 1, prop={'size':legend_fs})  

# axs[5].legend(handles = [ebm_boa_net_ocean, noresm_boa_net_ocean, era5_boa_net_ocean],
#               labels  = ["Net-PyEBM", "Net-NorESM2", "Net-ERA5"], frameon = False,
#               loc = "lc", bbox_to_anchor=(0.5, 0.5), ncols = 1, prop={'size':legend_fs})
 


# fig.save(cdir+"/figure5_v2.png", dpi = 400)



print('###########################')
print("Global Mean Net Radiation at Surface....")
print('###########################')

print("ZEMBA: " + str(round(ebm_rsnet_global, 2)))

print("NorESM2: " + str(round(noresm2_rsnet_global, 2)))

print("ERA5: " + str(round(era5_rsnet_global, 2))+'\n')

print('###########################')
print("Global Mean Net Radiation at TOA....")
print('###########################')

print("pyEBM: " + str(round(ebm_rtnet_global, 2)))

print("NorESM2: " + str(round(noresm2_rtnet_global, 2)))

print("ERA5: " + str(round(era5_rtnet_global, 2))+'\n')
















# #--------------
# # Plot figure 3
# #--------------

# # initialize figure
# fig, axs = pplt.subplots(figsize=(11, 7.75), nrows=2, ncols=3, sharey=False, sharex=False)

# # fonts
# axs.format(ticklabelsize=8, ticklabelweight='normal',
#            ylabelsize=8, ylabelweight='normal',
#            xlabelsize=8, xlabelweight='normal',
#            titlesize=10, titleweight='bold',
#            abc='A)', abcloc='ur', abcbbox=False,)


# # x-axis
# locatorx = np.arange(-90, 120, 30)
# minorlocatorx = np.arange(-90, 100, 10)
# axs.format(xminorlocator=minorlocatorx, xlocator=locatorx, xlim=(-90, 90))


# # plot EBM
# #---------

# # average
# axs[0,0].plot(Var["lat"], ebm_rsdtnet, color = "red9", lw = 2, label = "Net Shortwave (EBCM)") # Net shortwave at TOA
# axs[0,0].plot(Var["lat"], ebm_rlut, color = "blue9", lw = 2, label = "Outgoing Longwave (EBCM)") # Outgoing longwave radiation at TOA
# axs[0,0].plot(Var["lat"], ebm_rtnet, color = "black", lw = 2, label = "Net Radiation (EBCM)") # Net radiation at TOA
# axs[1,0].plot(Var["lat"], ebm_rsdsnet, color = "red9", lw = 2, label = "_nolabel") # Net shortwave at TOA
# axs[1,0].plot(Var["lat"], ebm_rlus, color = "blue9", lw = 2, label = "_nolabelM)") # Outgoing longwave radiation at TOA
# axs[1,0].plot(Var["lat"], ebm_rsnet, color = "black", lw = 2, label = "_nolabel") # Net radiation at TOA

# # land
# axs[0,1].plot(Var["lat"], ebm_rsdtnet_land, color = "red9", lw = 2, label = "_nolabel") # Net shortwave at TOA
# axs[0,1].plot(Var["lat"], ebm_rlut_land, color = "blue9", lw = 2, label = "_nolabel") # Outgoing longwave radiation at TOA
# axs[0,1].plot(Var["lat"], ebm_rtnet_land, color = "black", lw = 2, label = "_nolabel") # Net radiation at TOA
# axs[1,1].plot(Var["lat"], ebm_rsdsnet_land, color = "red9", lw = 2, label = "_nolabel") # Net shortwave at TOA
# axs[1,1].plot(Var["lat"], ebm_rlus_land, color = "blue9", lw = 2, label = "_nolabel") # Outgoing longwave radiation at TOA
# axs[1,1].plot(Var["lat"], ebm_rsnet_land, color = "black", lw = 2, label = "_nolabel") # Net radiation at TOA

# # ocean
# axs[0,2].plot(Var["lat"], ebm_rsdtnet_ocean, color = "red9", lw = 2, label = "_nolabel") # Net shortwave at TOA
# axs[0,2].plot(Var["lat"], ebm_rlut_ocean, color = "blue9", lw = 2, label = "_nolabel") # Outgoing longwave radiation at TOA
# axs[0,2].plot(Var["lat"], ebm_rtnet_ocean, color = "black", lw = 2, label = "_nolabel") # Net radiation at TOA
# axs[1,2].plot(Var["lat"], ebm_rsdsnet_ocean, color = "red9", lw = 2, label = "_nolabel") # Net shortwave at TOA
# axs[1,2].plot(Var["lat"], ebm_rlus_ocean, color = "blue9", lw = 2, label = "_nolabel") # Outgoing longwave radiation at TOA
# axs[1,2].plot(Var["lat"], ebm_rsnet_ocean, color = "black", lw = 2, label = "_nolabel") # Net radiation at TOA

# # plot NorESM2
# #-------------

# # average
# axs[0,0].plot(noresm2_annual.lat, noresm2_rsdtnet, color = "red9", lw = 1, linestyle = ":", label = "Net Shortwave (NorESM2)", alpha = 0.5) # Net shortwave at TOA
# axs[0,0].plot(noresm2_annual.lat, noresm2_rlut, color = "blue9", lw = 1, linestyle = ":", label = "Outgoing Longwave (NorESM2)", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[0,0].plot(noresm2_annual.lat, noresm2_rtnet, color = "black", lw = 1, linestyle = ":", label = "Net Radiation (NorESM2)", alpha = 0.5) # Net radiation at TOA
# axs[1,0].plot(noresm2_annual.lat, noresm2_rsdsnet, color = "red9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[1,0].plot(noresm2_annual.lat, noresm2_rlus, color = "blue9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[1,0].plot(noresm2_annual.lat, noresm2_rsnet, color = "black", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net radiation at TOA

# # land
# axs[0,1].plot(noresm2_annual.lat, noresm2_rsdtnet_land, color = "red9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[0,1].plot(noresm2_annual.lat, noresm2_rlut_land, color = "blue9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[0,1].plot(noresm2_annual.lat, noresm2_rtnet_land, color = "black", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net radiation at TOA
# axs[1,1].plot(noresm2_annual.lat, noresm2_rsdsnet_land, color = "red9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[1,1].plot(noresm2_annual.lat, noresm2_rlus_land, color = "blue9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[1,1].plot(noresm2_annual.lat, noresm2_rsnet_land, color = "black", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net radiation at TOA

# # ocean
# axs[0,2].plot(noresm2_annual.lat, noresm2_rsdtnet_ocean, color = "red9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[0,2].plot(noresm2_annual.lat, noresm2_rlut_ocean, color = "blue9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[0,2].plot(noresm2_annual.lat, noresm2_rtnet_ocean, color = "black", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net radiation at TOA
# axs[1,2].plot(noresm2_annual.lat, noresm2_rsdsnet_ocean, color = "red9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[1,2].plot(noresm2_annual.lat, noresm2_rlus_ocean, color = "blue9", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[1,2].plot(noresm2_annual.lat, noresm2_rsnet_ocean, color = "black", lw = 1, linestyle = ":", label = "_nolabel", alpha = 0.5) # Net radiation at TOA


# # plot ERA5
# #----------

# # average
# axs[0,0].plot(noresm2_annual.lat, era5_rsdtnet, color = "orange5", lw = 1, linestyle = "--", label = "Net Shortwave (ERA5 1940-1970)", alpha = 0.5) # Net shortwave at TOA
# axs[0,0].plot(noresm2_annual.lat, era5_rlut, color = "teal3", lw = 1, linestyle = "--", label = "Outgoing Longwave (ERA5 1940-1970)", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[0,0].plot(noresm2_annual.lat, era5_rtnet, color = "gray5", lw = 1, linestyle = "--", label = "Net Radiation (ERA5 1940-1970)", alpha = 0.5) # Net radiation at TOA
# axs[1,0].plot(noresm2_annual.lat, era5_rsdsnet, color = "orange5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[1,0].plot(noresm2_annual.lat, era5_rlus, color = "teal3", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[1,0].plot(noresm2_annual.lat, era5_rsnet, color = "gray5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net radiation at TOA

# # land
# axs[0,1].plot(noresm2_annual.lat, era5_rsdtnet_land, color = "orange5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[0,1].plot(noresm2_annual.lat, era5_rlut_land, color = "teal3", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[0,1].plot(noresm2_annual.lat, era5_rtnet_land, color = "gray5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net radiation at TOA
# axs[1,1].plot(noresm2_annual.lat, era5_rsdsnet_land, color = "orange5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[1,1].plot(noresm2_annual.lat, era5_rlus_land, color = "teal3", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[1,1].plot(noresm2_annual.lat, era5_rsnet_land, color = "gray5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net radiation at TOA


# # ocean
# axs[0,2].plot(noresm2_annual.lat, era5_rsdtnet_ocean, color = "orange5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[0,2].plot(noresm2_annual.lat, era5_rlut_ocean, color = "teal3", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[0,2].plot(noresm2_annual.lat, era5_rtnet_ocean, color = "gray5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net radiation at TOA
# axs[1,2].plot(noresm2_annual.lat, era5_rsdsnet_ocean, color = "orange5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net shortwave at TOA
# axs[1,2].plot(noresm2_annual.lat, era5_rlus_ocean, color = "teal3", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Outgoing longwave radiation at TOA
# axs[1,2].plot(noresm2_annual.lat, era5_rsnet_ocean, color = "gray5", lw = 1, linestyle = "--", label = "_nolabel", alpha = 0.5) # Net radiation at TOA

# # more formatting
# axs[0,:].format(xticklabels =[])
# axs[1:,1:].format(yticklabels =[])
# axs[0:,1:].format(yticklabels =[])
# axs[1,:].format(xlabel = "Latitude")
# axs[0,0].format(ylabel = r'TOA Radiative Fluxes ($W m^{-2}$)')
# axs[1,0].format(ylabel = r'Surface Radiative Fluxes ($W m^{-2}$)')
# axs[0,:].format(ylim = (-150, 350))
# axs[1,:].format(ylim = (-50, 250))
# axs[0,0].format(title = "Average")
# axs[0,1].format(title = "Land")
# axs[0,2].format(title = "Ocean")




# # legend
# fig.legend(loc = 'b', frameon = False, ncols = 3)

# # save figure
# fig.save(cdir+"/figure5.png", dpi = 400)




#-------------
# Compare data
#-------------

# function
# def compare_radiation(noresm2, noresm2_zonal, ceres, ceres_zonal, era5, era5_zonal, vmin, vmax, vmin1, vmax1, title):

#     fig, axs = pplt.subplots(ncols=3, nrows = 2, sharex = False, sharey = False, suptitle = title)
#     lons, lats = np.meshgrid(noresm2.lon, noresm2.lat)
#     axs[0,0].contourf(lons, lats, noresm2, vmin=vmin, vmax=vmax, cmap='BuRd', colorbar='r')
#     axs[0,1].contourf(lons, lats, ceres, vmin=vmin, vmax=vmax, cmap='BuRd', colorbar='r')
#     axs[0,2].contourf(lons, lats, era5, vmin=vmin, vmax=vmax, cmap='BuRd', colorbar='r')
#     axs[1,1].contourf(lons, lats, noresm2-ceres, vmin=vmin1, vmax=vmax1, cmap='BuRd', colorbar='r')
#     axs[1,2].contourf(lons, lats, noresm2-era5, vmin=vmin1, vmax=vmax1, cmap='BuRd', colorbar='r')
    
#     axs[1,0].plot(noresm2_annual.lat, noresm2_zonal, label = "NorESM2")
#     axs[1,0].plot(noresm2_annual.lat, ceres_zonal, label = "CERES (2005-2015)")
#     axs[1,0].plot(noresm2_annual.lat, era5_zonal, label = "ERA5 (1940-1970)")
#     axs.legend(frameon = False, loc = 'uc', ncols = 1)
    
#     axs[0,0].format(title = "NorESM2")
#     axs[0,1].format(title = "CERES (2005-2015)")
#     axs[0,2].format(title = "ERA5 (1940-1970)")
#     axs[1,0].format(title = "Zonal Mean")
#     axs[1,1].format(title = "NorESM2 - CERES (2005-2015)")
#     axs[1,2].format(title = "NorESM2 - ERA5 (1940-1970)")
    
#     return fig, axs

# net shortwave radiation at TOA
# fig2, ax2 = compare_radiation(noresm2_rsdtnet, noresm2_rsdtnet_zonal, ceres_rsdtnet, ceres_rsdtnet_zonal, era5_rsdtnet, era5_rsdtnet_zonal, vmin = 0, vmax = 400, vmin1 = -40, vmax1 = 40, title="Net Shortwave Radiation at TOA")

# outgoing longwave radiation at TOA
# fig3, ax3 = compare_radiation(noresm2_rlut, noresm2_rlut_zonal, ceres_rlut, ceres_rlut_zonal, era5_rlut, era5_rlut_zonal, vmin = 0, vmax = 300, vmin1 = -40, vmax1 = 40,  title="Outgoing Longwave Radiation at TOA")

# net total radiation at TOA
# fig4, ax4 = compare_radiation(noresm2_rtnet, noresm2_rtnet_zonal, ceres_rtnet, ceres_rtnet_zonal, era5_rtnet, era5_rtnet_zonal, vmin = -200, vmax = 100, vmin1 = -40, vmax1 = 40,  title="Net Total Radiation at TOA")

# net shortwave radiation at surface
# fig5, ax5 = compare_radiation(noresm2_rsdsnet, noresm2_rsdsnet_zonal, ceres_rsdsnet, ceres_rsdsnet_zonal, era5_rsdsnet, era5_rsdsnet_zonal, vmin = 0, vmax = 300, vmin1 = -40, vmax1 = 40, title="Net Shortwave Radiation at surface")

# outgoing longwave radiation at TOA
# fig6, ax6 = compare_radiation(noresm2_rlus, noresm2_rlus_zonal, ceres_rlus, ceres_rlus_zonal, era5_rlus, era5_rlus_zonal, vmin = 0, vmax = 150, vmin1 = -40, vmax1 = 40,  title="Outgoing Longwave Radiation at surface")

# net total radiation at TOA
# fig7, ax7 = compare_radiation(noresm2_rsnet, noresm2_rsnet_zonal, ceres_rsnet, ceres_rsnet_zonal, era5_rsnet, era5_rsnet_zonal, vmin = -50, vmax = 250, vmin1 = -40, vmax1 = 40,  title="Net Total Radiation at surface")









