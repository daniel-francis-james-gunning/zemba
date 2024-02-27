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
#----------

# load annual data
noresm_data = xr.open_dataset(os.getcwd()+"/other_data/noresm2/noresm2_annual.nc")
noresm_interp = noresm_data.interp(lat = Var["lat"]) # interpolate to EBM

# load monthly data
noresm_monthly = xr.open_dataset(os.getcwd()+ "/other_data/noresm2/noresm2_monthly.nc")
noresm_monthly_interp = noresm_monthly.interp(lat = Var["lat"]) # interpolate to EBM

# annual mean temperature
#------------------------

# atmosphere
noresm["tas_annual"] = (noresm_data["tas"]).mean(dim = "lon", skipna = True) - Var["K"]
noresm["tas_annual_interp"] = noresm_interp["tas"].mean(dim = "lon", skipna = True) - Var["K"]
noresm["tas_global_mean"]  = global_mean2(noresm["tas_annual"].to_numpy(), noresm_data.lat.to_numpy(), np.diff(noresm_data.lat.to_numpy())[0])

# land
noresm["tl_annual_interp"] = (noresm_interp["ts"]*noresm_interp["land_mask"]).mean(dim = "lon", skipna = True)  - Var["K"] 

# ocean
noresm["tos_annual_interp"] = (noresm_interp["ts"]*noresm_interp["ocean_mask"]).mean(dim = "lon", skipna = True)  - Var["K"] 

# june-july-august
#-----------------

# atmosphere
noresm["tas_jja_interp"] =       (
                        
                        (noresm_monthly_interp["tas"][5,:,:]).mean(dim = "lon", skipna = True).to_numpy() * 30
                        +
                        (noresm_monthly_interp["tas"][6,:,:]).mean(dim = "lon", skipna = True).to_numpy() * 31.
                        +
                        (noresm_monthly_interp["tas"][7,:,:]).mean(dim = "lon", skipna = True).to_numpy() * 31
                        
                        ) / (30. + 31. + 31.) - Var["K"]

# land
noresm["tl_jja_interp"]  =       ( 
                        
                        (noresm_monthly_interp["ts"][5,:,:]*noresm_interp["land_mask"]).mean(dim = "lon", skipna = True).to_numpy() * 30
                        +
                        (noresm_monthly_interp["ts"][6,:,:]*noresm_interp["land_mask"]).mean(dim = "lon", skipna = True).to_numpy() * 31.
                        +
                        (noresm_monthly_interp["ts"][7,:,:]*noresm_interp["land_mask"]).mean(dim = "lon", skipna = True).to_numpy() * 31
                        
                        ) / (30. + 31. + 31.) - Var["K"]

# ocean
noresm["tos_jja_interp"]  =       ( 
                        
                        (noresm_monthly_interp["ts"][5,:,:]*noresm_interp["ocean_mask"]).mean(dim = "lon", skipna = True).to_numpy() * 30
                        +
                        (noresm_monthly_interp["ts"][6,:,:]*noresm_interp["ocean_mask"]).mean(dim = "lon", skipna = True).to_numpy() * 31.
                        +
                        (noresm_monthly_interp["ts"][7,:,:]*noresm_interp["ocean_mask"]).mean(dim = "lon", skipna = True).to_numpy() * 31
                        
                        ) / (30. + 31. + 31.) - Var["K"]


# december-january-february
#--------------------------

# atmosphere
noresm["tas_djf_interp"] =       ( # atmosphere
                
                       (noresm_monthly_interp["tas"][-1,:,:]).mean(dim = "lon", skipna = True).to_numpy() * 31.
                       +
                       (noresm_monthly_interp["tas"][0,:,:]).mean(dim = "lon", skipna = True).to_numpy()  * 31.         
                       +
                       (noresm_monthly_interp["tas"][1,:,:]).mean(dim = "lon", skipna = True).to_numpy()  * 28.
            
                       ) / (31. + 31. + 28.) - Var["K"]

# land
noresm["tl_djf_interp"]  =       ( # land
                
                       (noresm_monthly_interp["ts"][-1,:,:]*noresm_interp["land_mask"]).mean(dim = "lon", skipna = True).to_numpy() * 31.
                       +
                       (noresm_monthly_interp["ts"][0,:,:]*noresm_interp["land_mask"]).mean(dim = "lon", skipna = True).to_numpy()  * 31.         
                       +
                       (noresm_monthly_interp["ts"][1,:,:]*noresm_interp["land_mask"]).mean(dim = "lon", skipna = True).to_numpy()  * 28.
            
                       ) / (31. + 31. + 28.) - Var["K"]

# ocean
noresm["tos_djf_interp"] =       ( # surface ocean
                
                       (noresm_monthly_interp["ts"][-1,:,:]*noresm_interp["ocean_mask"]).mean(dim = "lon", skipna = True).to_numpy() * 31.
                       +
                       (noresm_monthly_interp["ts"][0,:,:]*noresm_interp["ocean_mask"]).mean(dim = "lon", skipna = True).to_numpy()  * 31.         
                       +
                       (noresm_monthly_interp["ts"][1,:,:]*noresm_interp["ocean_mask"]).mean(dim = "lon", skipna = True).to_numpy()  * 28.
            
                       ) / (31. + 31. + 28.) - Var["K"]


#----------
# ERA5 data
#----------

# load data
#----------

# load annual data
era5_data = xr.open_dataset(os.getcwd() + "/other_data/era5/temp_era5_1940_1970.nc")
era5_interp = era5_data.interp(latitude = Var["lat"]) # interpolate to NorESM

# load monthly data
era5_monthly = xr.open_dataset(os.getcwd() + "/other_data/era5/temp_era5_1940_1970_monthly.nc")
era5_monthly_interp = era5_monthly.interp(latitude = Var["lat"]) # interpolate to NorESM

# annual mean temperature
#------------------------

# atmosphere
era5["tas_annual"] = (era5_data["t2m"]).mean(dim = "longitude", skipna = True) - Var["K"]
era5["tas_annual_interp"] = (era5_interp["t2m"]).mean(dim = "longitude", skipna = True) - Var["K"] 
era5["tas_global_mean"]  = round(global_mean2(era5["tas_annual"].to_numpy(), era5_data.latitude.to_numpy(), np.diff(era5_data.latitude.to_numpy())[0]), 2)

# june-july-august 
#-----------------

# atmosphere
era5["tas_jja_interp"] =       (
                        
                        (era5_monthly_interp["t2m"][5,:,:]).mean(dim = "longitude", skipna = True).to_numpy() * 30
                        +
                        (era5_monthly_interp["t2m"][6,:,:]).mean(dim = "longitude", skipna = True).to_numpy() * 31.
                        +
                        (era5_monthly_interp["t2m"][7,:,:]).mean(dim = "longitude", skipna = True).to_numpy() * 31
                        
                        ) / (30. + 31. + 31.) - Var["K"]

# december-january-february
#--------------------------

# atmosphere
era5["tas_djf_interp"]  =        ( 
                
                       (era5_monthly_interp["t2m"][-1,:,:]).mean(dim = "longitude", skipna = True).to_numpy() * 31.
                       +
                       (era5_monthly_interp["t2m"][0,:,:]).mean(dim = "longitude", skipna = True).to_numpy()  * 31.         
                       +
                       (era5_monthly_interp["t2m"][1,:,:]).mean(dim = "longitude", skipna = True).to_numpy()  * 28.
            
                       ) / (31. + 31. + 28.) - Var["K"]


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
era5_annual_line   = axs[0].plot(Var["lat"], era5["tas_annual_interp"], color = era5_color, lw = era5_lw, linestyle = era5_ls)
noresm_annual_line = axs[0].plot(Var["lat"], noresm["tas_annual_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls)

# legends
axs[0].legend(handles = [ebm_annual_line, noresm_annual_line, era5_annual_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})

# plot differences
ebm_minus_noresm_annual_line = axs[3].plot(Var["lat"], ebm["tas_annual"] - noresm["tas_annual_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_annual_line   = axs[3].plot(Var["lat"], ebm["tas_annual"] - era5["tas_annual_interp"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# difference legends
axs[3].legend(handles = [ebm_minus_noresm_annual_line, ebm_minus_era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})


# plot december - january - february
#-----------------------------------


# lines
ebm_djf_line    = axs[1].plot(Var["lat"], ebm["tas_djf"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
noresm_djf_line = axs[1].plot(Var["lat"], noresm["tas_djf_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
era5_djf_line   = axs[1].plot(Var["lat"], era5["tas_djf_interp"], color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# legend
axs[1].legend(handles = [ebm_djf_line, noresm_djf_line, era5_djf_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})

# plot difference
ebm_minus_noresm_djf_line = axs[4].plot(Var["lat"], ebm["tas_djf"] - noresm["tas_djf_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_djf_line   = axs[4].plot(Var["lat"], ebm["tas_djf"] - era5["tas_djf_interp"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# differences legend
axs[4].legend(handles = [ebm_minus_noresm_djf_line, ebm_minus_era5_djf_line, era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})


# plot june - july - august
#--------------------------


# lines
ebm_jja_line    = axs[2].plot(Var["lat"], ebm["tas_jja"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
noresm_jja_line = axs[2].plot(Var["lat"], noresm["tas_jja_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
era5_jja_line   = axs[2].plot(Var["lat"], era5["tas_jja_interp"], color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# legend
axs[2].legend(handles = [ebm_jja_line, noresm_jja_line, era5_jja_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})


# plot difference
ebm_minus_noresm_jja_line = axs[5].plot(Var["lat"], ebm["tas_jja"] - noresm["tas_jja_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
ebm_minus_era5_jja_line   = axs[5].plot(Var["lat"], ebm["tas_jja"] - era5["tas_jja_interp"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# differences legend
leg = axs[5].legend(handles = [ebm_minus_noresm_jja_line, ebm_minus_era5_jja_line, era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})


fig.save(os.getcwd()+"/output/plots/f02.png", dpi = 400)
fig.save(os.getcwd()+"/output/plots/f02.pdf", dpi = 400)



# #------------------------------------------------------------------------------
# # Plot -- Version 2
# #------------------------------------------------------------------------------

# # colors and line widths
# #-----------------------

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

# # legend font size
# legend_fs = 8

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
#         [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
        
#         [7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9],
#         [7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9],
#         [7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9],
#         [7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9]]

# # figure and axes
# fig, axs = pplt.subplots(shape, figsize = (15,10), sharey = False, sharex = False)

# # secondary yaxis
# axs_twin = axs.twinx() 

# # fonts
# axs.format(ticklabelsize=6, ticklabelweight='normal', ylabelsize=10, ylabelweight='bold',
#             xlabelsize=7, xlabelweight='normal', titlesize=10, titleweight='bold', abc='A)', 
#             abcloc='ur', abcbbox=False, xlim = (-90, 90))

# # fonts -- secondary axis
# axs_twin.format(ylabelsize=7, ylabelweight='normal', ticklabelsize=6)

# # ticks and labels
# locatorx      = np.arange(-90, 120, 30)
# minorlocatorx = np.arange(-90, 100, 10)
# for i in np.arange(0, 8+1):
    
    
#     axs[i].format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))
#     axs[i].format(ylim = (-80, 30), yminorlocator=np.arange(-40, 30, 5),)
#     axs[i].set_yticks(np.arange(-40, 30+10, 10))
#     axs[i].yaxis.set_label_coords(-0.1,0.65)
    
#     # secondary y-axis
#     axs_twin[i].set_yticks(np.arange(-8, 8+2, 4))
#     axs_twin[i].format(ylim = (-10, 50), ylabel = r'Difference in' + "\n"+ r' Temperature ($^{\circ}$C)', yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.1,0.13)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")
    
    
# axs[0].format(ylabel = 'Surface Air Temperature ($^{\circ}$C)')
# axs[0+3].format(ylabel = 'Surface Land Temperature ($^{\circ}$C)')
# axs[0+6].format(ylabel = 'Surface Ocean Temperature ($^{\circ}$C)')


# axs[6:9].format(xlabel = 'Latitude', )

# # plot annual mean temperatures
# #------------------------------

# # air temperature lines
# ebm_tas_annual_line    = axs[0].plot(Var["lat"], ebm["tas_annual"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls)
# noresm_tas_annual_line  = axs[0].plot(Var["lat"], noresm["tas_annual_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls)
# era5_tas_annual_line    = axs[0].plot(Var["lat"], era5["tas_annual_interp"], color = era5_color, lw = era5_lw, linestyle = era5_ls)

# # land temperature lines
# ebm_tl_annual_line    = axs[0+3].plot(Var["lat"], ebm["tl_annual"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls)
# noresm_tl_annual_line  = axs[0+3].plot(Var["lat"], noresm["tl_annual_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls)
# # era5_tl_annual_line    = axs[0+3].plot(era5_annual.lat, era5_tl_annual, color = era5_color, lw = era5_lw, linestyle = era5_ls)

# # ocean temperature lines
# ebm_tos_annual_line    = axs[0+6].plot(Var["lat"], ebm["tos_annual"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls)
# noresm_tos_annual_line  = axs[0+6].plot(Var["lat"], noresm["tos_annual_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls)
# # era5_tos_annual_line    = axs[0+6].plot(era5_annual.lat, era5_tos_annual, color = era5_color, lw = era5_lw, linestyle = era5_ls)

# # titles
# axs[0].format(title = 'Annual')

# # legends
# axs[0].legend(handles = [ebm_tas_annual_line, noresm_tas_annual_line, era5_tas_annual_line], labels = ["PyEBM", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})
# axs[0+3].legend(handles = [ebm_tl_annual_line, noresm_tl_annual_line], labels = ["PyEBM", "NorESM2"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})
# axs[0+6].legend(handles = [ebm_tos_annual_line, noresm_tos_annual_line], labels = ["PyEBM", "NorESM2"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})

# # plot differences

# ebm_minus_noresm_tas_annual_line = axs_twin[0].plot(Var["lat"], ebm["tas_annual"] - noresm["tas_annual_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_tas_annual_line   = axs_twin[0].plot(Var["lat"], ebm["tas_annual"] - era5["tas_annual_interp"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# ebm_minus_noresm_tl_annual_line = axs_twin[0+3].plot(Var["lat"], ebm["tl_annual"] - noresm["tl_annual_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# # ebm_minus_era5_tl_annual_line   = axs_twin[0+3].plot(Var["lat"], ebm_tl_annual - era5_tl_annual, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# ebm_minus_noresm_tos_annual_line = axs_twin[0+6].plot(Var["lat"], ebm["tos_annual"] - noresm["tos_annual_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# # ebm_minus_era5_tos_annual_line   = axs_twin[0+6].plot(Var["lat"], ebm_tos_annual - era5_tos_annual, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)


# # difference legends
# axs[0].legend(handles = [ebm_minus_noresm_tas_annual_line], labels = ["PyEBM - NorESM2"], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})
# axs[0+3].legend(handles = [ebm_minus_noresm_tl_annual_line], labels = ["PyEBM - NorESM2"], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})
# axs[0+6].legend(handles = [ebm_minus_noresm_tos_annual_line], labels = ["PyEBM - NorESM2"], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})


# # plot december - january - february
# #-----------------------------------


# # air temperature lines
# ebm_tas_djf_line    = axs[1].plot(Var["lat"], ebm["tas_djf"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
# noresm_tas_djf_line = axs[1].plot(Var["lat"], noresm["tas_djf_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
# era5_tas_djf_line   = axs[1].plot(Var["lat"], era5["tas_djf_interp"], color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# # land temperature lines
# ebm_tl_djf_line      = axs[1+3].plot(Var["lat"], ebm["tl_djf"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
# noresm_tl_djf_line   = axs[1+3].plot(Var["lat"], noresm["tl_djf_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
# # era5_tl_djf_line     = axs[1+3].plot(Var["lat"], era5_tl_djf, color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# # ocean temperature lines
# ebm_tos_djf_line = axs[1+6].plot(Var["lat"], ebm["tos_djf"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
# noresm_tos_djf_line  = axs[1+6].plot(Var["lat"], noresm["tos_djf_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
# # era5_tos_djf_line    = axs[1+6].plot(Var["lat"], era5_tos_djf, color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)


# # titles
# axs[1].format(title = 'DJF')

# # legend
# axs[1].legend(handles = [ebm_tas_djf_line, noresm_tas_djf_line, era5_tas_djf_line], labels = ["PyEBM", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})
# axs[1+3].legend(handles = [ebm_tl_djf_line, noresm_tl_djf_line], labels = ["PyEBM", "NorESM2"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})
# axs[1+6].legend(handles = [ebm_tos_djf_line, noresm_tos_djf_line], labels = ["PyEBM", "NorESM2"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})

# # plot difference
# ebm_minus_noresm_tas_djf_line = axs_twin[1].plot(Var["lat"], ebm["tas_djf"] - noresm["tas_djf_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_tas_djf_line   = axs_twin[1].plot(Var["lat"], ebm["tas_djf"] - era5["tas_djf_interp"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# ebm_minus_noresm_tl_djf_line = axs_twin[1+3].plot(Var["lat"], ebm["tl_djf"] - noresm["tl_djf_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# # ebm_minus_era5_tl_djf_line   = axs_twin[1+3].plot(Var["lat"], ebm_tl_djf - era5_tl_djf, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# ebm_minus_noresm_tos_djf_line = axs_twin[1+6].plot(Var["lat"], ebm["tos_djf"] - noresm["tos_djf_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# # ebm_minus_era5_tos_djf_line   = axs_twin[1+6].plot(Var["lat"], ebm_tos_djf - era5_tos_djf, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # differences legend
# axs[1].legend(handles = [ebm_minus_noresm_tas_djf_line, ebm_minus_era5_tas_djf_line], labels = ["PyEBM - NorESM2", "PyEBM - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})
# axs[1+3].legend(handles = [ebm_minus_noresm_tl_djf_line], labels = ["PyEBM - NorESM2",], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})
# axs[1+6].legend(handles = [ebm_minus_noresm_tos_djf_line], labels = ["PyEBM - NorESM2"], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})


# # plot june - july - august
# #--------------------------


# # air temperature lines
# ebm_tas_jja_line    = axs[2].plot(Var["lat"], ebm["tas_jja"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
# noresm_tas_jja_line = axs[2].plot(Var["lat"], noresm["tas_jja_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
# era5_tas_jja_line   = axs[2].plot(Var["lat"], era5["tas_jja_interp"], color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# # land temperature lines
# ebm_tl_jja_line      = axs[2+3].plot(Var["lat"], ebm["tl_jja"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
# noresm_tl_jja_line   = axs[2+3].plot(Var["lat"], noresm["tl_jja_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
# # era5_tl_jja_line     = axs[1+3].plot(Var["lat"], era5_tl_jja, color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# # ocean temperature lines
# ebm_tos_jja_line = axs[2+6].plot(Var["lat"], ebm["tos_jja"], color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, alpha = 1)
# noresm_tos_jja_line  = axs[2+6].plot(Var["lat"], noresm["tos_jja_interp"], color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
# # era5_tos_jja_line    = axs[1+6].plot(Var["lat"], era5_tos_jja, color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)

# axs[2].format(title = 'JJA')

# # legend
# axs[2].legend(handles = [ebm_tas_jja_line, noresm_tas_jja_line, era5_tas_jja_line], labels = ["PyEBM", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})
# axs[2+3].legend(handles = [ebm_tl_jja_line, noresm_tl_jja_line], labels = ["PyEBM", "NorESM2"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})
# axs[2+6].legend(handles = [ebm_tos_jja_line, noresm_tos_jja_line], labels = ["PyEBM", "NorESM2"], ncols = 1, loc = "uc", frameon = False, bbox_to_anchor=(0.5, 0.85), prop={'size':legend_fs})

# # plot difference
# ebm_minus_noresm_tas_jja_line = axs_twin[2].plot(Var["lat"], ebm["tas_jja"] - noresm["tas_jja_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# ebm_minus_era5_tas_jja_line   = axs_twin[2].plot(Var["lat"], ebm["tas_jja"] - era5["tas_jja_interp"], color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# ebm_minus_noresm_tl_jja_line = axs_twin[2+3].plot(Var["lat"], ebm["tl_jja"] - noresm["tl_jja_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# # ebm_minus_era5_tl_jja_line   = axs_twin[1+3].plot(Var["lat"], ebm_tl_jja - era5_tl_jja, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# ebm_minus_noresm_tos_jja_line = axs_twin[2+6].plot(Var["lat"], ebm["tos_jja"] - noresm["tos_jja_interp"], color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# # ebm_minus_era5_tos_jja_line   = axs_twin[1+6].plot(Var["lat"], ebm_tos_jja - era5_tos_jja, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # differences legend
# axs[2].legend(handles = [ebm_minus_noresm_tas_jja_line, ebm_minus_era5_tas_jja_line], labels = ["PyEBM - NorESM2", "PyEBM - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})
# axs[2+3].legend(handles = [ebm_minus_noresm_tl_jja_line], labels = ["PyEBM - NorESM2",], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})
# axs[2+6].legend(handles = [ebm_minus_noresm_tos_jja_line], labels = ["PyEBM - NorESM2"], ncols = 1, loc = "lc", frameon = False, bbox_to_anchor=(0.5, 0.25), prop={'size':legend_fs})


# fig.save(cdir+"/figure2_v2.png", dpi = 400)



#------------------------------------------------------------------------------
# Print global mean air temperatures
#------------------------------------------------------------------------------

print('###########################')
print('Global Mean Air Temperature')
print('###########################')   
print("ZEMBA: " + str(ebm["tas_global_mean"]))
print("NorESM2: " + str(noresm["tas_global_mean"]))
print("ERA5: " + str(era5["tas_global_mean"]) + '\n')










'''
Previous version -- included tables
'''


# #------------------------------------------------------------------------------
# # Plot -- Version 1
# #------------------------------------------------------------------------------


# # constants
# lc         = "black"
# lc_noresm2 = "blue9"
# lc_era5    = "red9"
# lw         = 1.5
# lw_noresm2 = 1
# lw_era5    = 1
# linestyle_noresm2 = ":"
# linestyle_era5 = "--"

# # initialize figure
# fig, axs = pplt.subplots(figsize = (10,6), nrows = 3, ncols = 3, sharey = False, sharex = False)

# # fonts
# axs.format(ticklabelsize=8, ticklabelweight='normal',
#            ylabelsize=8, ylabelweight='normal',
#            xlabelsize=8, xlabelweight='normal',
#            titlesize=10, titleweight='bold',
#            abc='A)', abcloc='ur', abcbbox=False,)


# # x-axis 
# locatorx      = np.arange(-90, 120, 30)
# minorlocatorx = np.arange(-90, 100, 10)
# axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))
# axs[2,:].format(xlabel = "Latitude")
# axs[0:2,:].format(xticklabels =[])
# axs[:,1:3].format(yticklabels =[])

# # y-axis
# axs.format(ylim = (-60, 35))

# # EBM
# #----

# # air temperature
# axs[0,0].plot(Var["lat"], ebm_tas_annual, color = lc, lw = lw, label = ["EBCM"])
# axs[1,0].plot(Var["lat"], ebm_tas_jja, color = lc, lw = lw, label = ["_nolabel"])
# axs[2,0].plot(Var["lat"], ebm_tas_djf, color = lc, lw = lw, label = ["_nolabel"])
# axs[0,0].format(title = "Atmosphere", ylabel= 'Annual Mean \n' +r' Temperature ($^{\circ}$C)')
# axs[1,0].format(ylabel= 'JJA Mean \n' + r'Temperature ($^{\circ}$C)')
# axs[2,0].format(ylabel= 'DJF Mean \n' + r'Temperature ($^{\circ}$C)')

# # land temperature
# axs[0,1].plot(Var["lat"], ebm_tl_annual, color = lc, lw = lw, label = ["_nolabel"])
# axs[1,1].plot(Var["lat"], ebm_tl_jja, color = lc, lw = lw, label = ["_nolabel"])
# axs[2,1].plot(Var["lat"], ebm_tl_djf, color = lc, lw = lw, label = ["_nolabel"])
# axs[0,1].format(title = "Land")

# # air temperature
# axs[0,2].plot(Var["lat"], ebm_tos_annual, color = lc, lw = lw, label = ["_nolabel"])
# axs[1,2].plot(Var["lat"], ebm_tos_jja, color = lc, lw = lw, label = ["_nolabel"])
# axs[2,2].plot(Var["lat"], ebm_tos_djf, color = lc, lw = lw, label = ["_nolabel"])
# axs[0,2].format(title = "Surface Ocean")

# # NorESM
# #-------

# # plot air temperature
# axs[0,0].plot(noresm_annual.lat, noresm_tas_annual, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["NorESM2"])
# axs[1,0].plot(noresm_annual.lat, noresm_tas_jja, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["_nolabel"])
# axs[2,0].plot(noresm_annual.lat, noresm_tas_djf, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["_nolabel"])


# # plot land temperature
# axs[0,1].plot(noresm_annual.lat, noresm_tl_annual, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["_nolabel"])
# axs[1,1].plot(noresm_annual.lat, noresm_tl_jja, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["_nolabel"])
# axs[2,1].plot(noresm_annual.lat, noresm_tl_djf, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["_nolabel"])


# # plot surface ocean temperature
# axs[0,2].plot(noresm_annual.lat, noresm_tos_annual, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["_nolabel"])
# axs[1,2].plot(noresm_annual.lat, noresm_tos_jja, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["_nolabel"])
# axs[2,2].plot(noresm_annual.lat, noresm_tos_djf, color = lc_noresm2, lw = lw_noresm2, linestyle = linestyle_noresm2, label = ["_nolabel"])

# # ERA5
# #-----

# # plot air temperature
# axs[0,0].plot(era5_annual.latitude, era5_tas_annual, color = lc_era5, lw = lw_noresm2, linestyle = linestyle_era5, label = ["ERA5 (1940-1970)"])
# axs[1,0].plot(era5_annual.latitude, era5_tas_jja, color = lc_era5, lw = lw_noresm2, linestyle = linestyle_era5, label = ["_nolabel"])
# axs[2,0].plot(era5_annual.latitude, era5_tas_djf, color = lc_era5, lw = lw_noresm2, linestyle = linestyle_era5, label = ["_nolabel"])


# fig.legend(loc="b", frame = False)




# #------------------------------------------------------------------------------
# # Plot -- Version 2
# #------------------------------------------------------------------------------

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
# fig, axs = pplt.subplots(shape, figsize = (10,5), sharey = False, sharex = False)

# # secondary yaxis
# axs_twin = axs[0:3].twinx() 

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
# for i in np.arange(3,5+1):
#     axs[i].format(xminorlocator = minorlocatorx, xlocator = locatorx, xlabel = "Latitude", xlim = (-90, 90))


# # format top row
# for i in np.arange(0,2+1):
    
#     axs[i].format(xlocator = locatorx, xlim = (-90, 90))
    
#     # set y-axis range
#     axs[i].format(ylim = (-80, 30), yminorlocator=np.arange(-40, 30, 5))
    
#     # set y-axis tick marks
#     axs[i].set_yticks(np.arange(-40, 30+10, 10))
    
#     # change position of yaxis label
#     axs[i].yaxis.set_label_coords(-0.15,0.65)
    
#     # remove xaxis labels for the top row
#     axs[i].grid(True)
#     axs[i].xaxis.set_ticklabels([])
#     axs[i].xaxis.set_ticks_position('none')
#     axs[i].xaxis.set_tick_params(labelbottom=False)


# # plot annual mean temperatures
# #------------------------------

# # plot air temperature
# axs[0].plot(Var["lat"], ebm_tas_annual, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["EBCM"])
# axs[0].plot(noresm_annual.lat, noresm_tas_annual, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["NorESM2"])
# axs[0].plot(era5_annual.lat, era5_tas_annual, color = era5_color, lw = era5_lw, linestyle = era5_ls, label = ["ERA5 (1940-1970)"])
# axs[0:3].format(title = "Atmosphere", ylabel= 'Annual Mean \n' +r' Temperature ($^{\circ}$C)')

# # plot land temperature
# axs[1].plot(Var["lat"], ebm_tl_annual, color = ebm_color, lw = ebm_lw, label = ["_nolabel"])
# axs[1].plot(noresm_annual.lat, noresm_tl_annual, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# axs[1].format(title = "Land")

# # plot ocean temperature
# axs[2].plot(Var["lat"], ebm_tos_annual, color = ebm_color, lw = ebm_lw, linestyle = ebm_ls, label = ["_nolabel"])
# axs[2].plot(noresm_annual.lat, noresm_tos_annual, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, label = ["_nolabel"])
# axs[2].format(title = "Surface Ocean")

# # plot differences in annual mean temperature
# #--------------------------------------------

# # format secondary axis
# for i in np.arange(0, 2+1):
#     axs_twin[i].set_yticks(np.arange(-8, 8+2, 4))
#     axs_twin[i].format(ylim = (-10, 50), ylabel = r'Difference in Annual' + "\n"+ r' Mean Temperature ($^{\circ}$C)', yminorlocator = [])
#     axs_twin[i].yaxis.set_label_coords(1.12,0.13)
#     axs_twin[i].grid(axis = 'y', linestyle = (0, (1, 4)), lw = 0.25, alpha = 0.25)
#     axs_twin[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.5, linestyle =":")
# # atmosphere
# axs_twin[0].plot(Var["lat"], ebm_tas_annual - noresm_tas_annual, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)
# axs_twin[0].plot(Var["lat"], ebm_tas_annual - era5_tas_annual, color = ebm_minus_era5_color, lw = ebm_minus_era5_lw, linestyle = ebm_minus_era5_ls)

# # land
# axs_twin[1].plot(Var["lat"], ebm_tl_annual - noresm_tl_annual, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)

# # ocean
# axs_twin[2].plot(Var["lat"], ebm_tos_annual - noresm_tos_annual, color = ebm_minus_noresm_color, lw = ebm_minus_noresm_lw, linestyle = ebm_minus_noresm_ls)


# # plot tables
# #------------

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
#                    r'30$^{\circ}$S - 0$^{\circ}$E',  r'0$^{\circ}$E - 30$^{\circ}$N',  r'30$^{\circ}$N - 60$^{\circ}$N', 
#                    r'60$^{\circ}$N - 90$^{\circ}$N')

# # table data
# cell_text_1 = [ebm_tas_annual_zones, noresm_tas_annual_zones, era5_tas_annual_zonal]
# cell_text_2 = [ebm_tl_annual_zones, noresm_tl_annual_zones, np.zeros((6))]
# cell_text_3 = [ebm_tos_annual_zones, noresm_tos_annual_zones, np.zeros((6))]
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
            

    
# # plot djf and jja temperatures
# #------------------------------

# # format bottom row
# for i in np.arange(3, 5+1):
#     axs[i].format(ylim = (-60, 35), ylabel = r'Seasonal Temperature ($^{\circ}$C)',
#                   yminorlocator = [], ylocator = np.arange(-60, 30+1, 10))

# # atmosphere
# ebm_tas_plot_line = axs[3].plot(Var["lat"], ebm_tas_jja, color = ebm_color, lw = ebm_lw, linestyle = "--", alpha = 1, marker = "^", markerfacecolor = "gray", markeredgecolor = "black", markeredgewidth=0.2, markersize = 2)

# noresm_tas_plot_line = axs[3].plot(Var["lat"], noresm_tas_jja, color = noresm_color, lw = noresm_lw, linestyle = "--", alpha = 1, marker = "^", markerfacecolor = noresm_color, markersize = 2, markeredgecolor = "black", markeredgewidth=0.2)

# era5_tas_plot_line = axs[3].plot(Var["lat"], era5_tas_jja, color = era5_color, lw = era5_lw, linestyle = "--", alpha = 1, marker = "^", markerfacecolor = era5_color, markersize = 2, markeredgecolor = "black", markeredgewidth=0.2)


# ebm_tas_plot_line_djf = axs[3].plot(Var["lat"], ebm_tas_djf, color = ebm_color, lw = ebm_lw, linestyle = ":", alpha = 0.5)
# # axs[3].plot(Var["lat"], ebm_tas_djf, color = ebm_color, lw = 0, linestyle = ":", marker = "o", markerfacecolor = ebm_color, markersize = 2, markeredgecolor = "black", markeredgewidth=0.2)

# axs[3].plot(Var["lat"], noresm_tas_djf, color = noresm_color, lw = noresm_lw, linestyle =":", alpha = 0.5)
# # axs[3].plot(Var["lat"], noresm_tas_djf, color = noresm_color, lw = 0, linestyle =":", marker = "o", markerfacecolor = noresm_color, markersize = 2, markeredgecolor = "black", markeredgewidth=0.2)

# axs[3].plot(Var["lat"], era5_tas_djf, color = era5_color, lw = era5_lw, linestyle = ":", alpha = 0.5)
# # axs[3].plot(Var["lat"], era5_tas_djf, color = era5_color, lw = 0, linestyle = ":", marker = "o", markerfacecolor = era5_color, markersize = 2, markeredgecolor = "black", markeredgewidth=0.2)


# # land
# axs[4].plot(Var["lat"], ebm_tl_jja, color = ebm_color, lw = ebm_lw, linestyle = "--", alpha = 1, marker = "^", markerfacecolor = "gray", markersize = 2, markeredgewidth=0.2, markeredgecolor = "black")

# axs[4].plot(Var["lat"], noresm_tl_jja, color = noresm_color, lw = noresm_lw, linestyle = "--", alpha = 1, marker = "^", markerfacecolor = noresm_color, markersize = 2, markeredgewidth=0.2, markeredgecolor = "black")

# axs[4].plot(Var["lat"], ebm_tl_djf, color = ebm_color, lw = ebm_lw, linestyle =":", alpha = 0.5)

# axs[4].plot(Var["lat"], noresm_tl_djf, color = noresm_color, lw = noresm_lw, linestyle =":", alpha = 0.5)


# # ocean
# axs[5].plot(Var["lat"], ebm_tos_jja, color = ebm_color, lw = ebm_lw, linestyle = "--", alpha = 1, marker = "^", markerfacecolor = "gray", markersize = 2, markeredgewidth=0.2, markeredgecolor = "black")

# axs[5].plot(Var["lat"], noresm_tos_jja, color = noresm_color, lw = noresm_lw, linestyle = ebm_ls, alpha=1, marker = "^", markerfacecolor = noresm_color, markeredgewidth=0.2, markersize = 2, markeredgecolor = "black")

# axs[5].plot(Var["lat"], ebm_tos_djf, color = ebm_color, lw = ebm_lw, linestyle =":", alpha = 0.5)

# axs[5].plot(Var["lat"], noresm_tos_djf, color = noresm_color, lw = noresm_lw, linestyle =":", alpha = 0.5)

# for i in np.arange(3, 5+1):
#     axs[i].legend([ebm_tas_plot_line, ebm_tas_plot_line_djf], ['JJA', 'DJF'], frameon = False, loc = "lower center", ncols = 2, fontsize=1)


# fig.save(cdir+"/figure2.png", dpi = 400)
