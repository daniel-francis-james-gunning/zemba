# -*- coding: utf-8 -*-
"""
Plot Figure 5 (Pre-Industrial - Results) Snow Cover

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

    

# snow coverage 
ebm_snc_monthly = calculate_monthlies(pi["snow_fraction_land"], Var["lat"].size)
ebm_snc_nh      = np.nansum(ebm_snc_monthly[Var["idxnh"].astype("int")]*Var["larea"][Var["idxnh"].astype("int")].reshape(18, 1), axis=0)
ebm_snc_sh      = np.nansum(ebm_snc_monthly[Var["idxsh"].astype("int")]*Var["larea"][Var["idxsh"].astype("int")].reshape(18, 1), axis=0)

ebm_snc_nh_global = np.average(ebm_snc_nh, weights = Var["days_in_months"])
ebm_snc_sh_global = np.average(ebm_snc_sh, weights = Var["days_in_months"])

# sea ice coverage
ebm_sic_nh = calculate_monthlies(pi["si_area_nh"], 1)
ebm_sic_sh = calculate_monthlies(pi["si_area_sh"], 1)

ebm_sic_nh_global = np.average(ebm_sic_nh[0,:], weights = Var["days_in_months"])
ebm_sic_sh_global = np.average(ebm_sic_sh[0,:], weights = Var["days_in_months"])
 
#-----------------
# Add NorESM2 data
#-----------------

# load monthly data
noresm_monthly = xr.open_dataset(os.getcwd() +  "/other_data/noresm2/monthly/noresm2_monthly.nc")

# monthly mean snow cover
noresm_snc_spatial  = noresm_monthly["snc"]/100   # snow fraction over land
noresm_area_spatial = noresm_monthly["areacella"] # cell area
noresm_snc_nh = (noresm_snc_spatial[:,90:,:] * noresm_area_spatial[90:,:]).sum(dim = "lon", skipna = True).sum(dim = "lat", skipna = True)
noresm_snc_sh = (noresm_snc_spatial[:,0:90,:] * noresm_area_spatial[0:90,:]).sum(dim = "lon", skipna = True).sum(dim = "lat", skipna = True)

noresm_snc_nh_global = np.average(noresm_snc_nh, weights = Var["days_in_months"])
noresm_snc_sh_global = np.average(noresm_snc_sh, weights = Var["days_in_months"])

# monthly mean sea ice cover
noresm_sic_nh = noresm_monthly["siarean"].to_numpy()*1e12
noresm_sic_sh = noresm_monthly["siareas"].to_numpy()*1e12

noresm_sic_nh_global = np.average(noresm_sic_nh, weights = Var["days_in_months"])
noresm_sic_sh_global = np.average(noresm_sic_sh, weights = Var["days_in_months"])



#-----------------
# Add ERA5 data
#-----------------

# load monthly data
era5_seaice_monthly = xr.open_dataset(os.getcwd() +  "/other_data/era5/snow_era5_1940_1970_monthly.nc")
era5_seaice_monthly = era5_seaice_monthly.interp(latitude = noresm_monthly.lat, longitude = noresm_monthly.lon)

era5_snow_monthly = xr.open_dataset(os.getcwd() +  "/other_data/era5/snow_cover_era5_1950_1970_monthly.nc")
era5_snow_monthly = era5_snow_monthly.interp(latitude = noresm_monthly.lat, longitude = noresm_monthly.lon)

# monthly mean snow cover
era5_snow_spatial = era5_snow_monthly["snowc"]/100
era5_snc_nh = (era5_snow_spatial[:,90:,:] * noresm_area_spatial[90:,:]).sum(dim = "lon", skipna = True).sum(dim = "lat", skipna = True)
era5_snc_sh = (era5_snow_spatial[:,0:90,:] * noresm_area_spatial[0:90,:]).sum(dim = "lon", skipna = True).sum(dim = "lat", skipna = True)

era5_snc_nh_global = np.average(era5_snc_nh, weights = Var["days_in_months"])
era5_snc_sh_global = np.average(era5_snc_sh, weights = Var["days_in_months"])


# monthly mean sea ice cover
era5_sic_spatial  = era5_seaice_monthly["siconc"]   # snow fraction over land
era5_sic_nh = (era5_sic_spatial[:,90:,:] * noresm_area_spatial[90:,:]).sum(dim = "lon", skipna = True).sum(dim = "lat", skipna = True)
era5_sic_sh = (era5_sic_spatial[:,0:90,:] * noresm_area_spatial[0:90,:]).sum(dim = "lon", skipna = True).sum(dim = "lat", skipna = True)

era5_sic_nh_global = np.average(era5_sic_nh, weights = Var["days_in_months"])
era5_sic_sh_global = np.average(era5_sic_sh, weights = Var["days_in_months"])


#---------------------------
# Plot Figure 4 -- version 1
#---------------------------

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


# legend font size
legend_fs = 7
legend_fnt = "bold"


# formating
#----------

# shape
shape = [  
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],]

# figure and axes
fig, axs = pplt.subplots(shape, figsize = (8,3), sharey = False, sharex = False, grid = False)


# fonts
axs.format(ticklabelsize=7, ticklabelweight='normal', 
           ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', 
            titlesize=8, titleweight='normal',)

# x-axis
axs.format(xlocator = np.arange(1, 12+1), xminorlocator = [], xticklabels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
           xlabel = 'Month')


# format top subplots
for i in np.arange(0,1+1):
    
    # y-axis
    axs[i].format(ylim = (0, 5), yminorlocator=np.arange(0, 5+1, 1), ylocator = np.arange(0, 5+1, 1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    

# titles
axs[0].format(title = r'(a) PI areal extent (x10$^{13} \: m^{2}$) in snow coverage over land', titleloc = 'left')
axs[1].format(title = r'(b) PI areal extent (x10$^{13} \: m^{2}$) in sea ice cover', titleloc = 'left')



# snow coverage
#--------------

# northern hemisphere
ebm_snc_nh_line = axs[0].plot(np.arange(1, 12+1), ebm_snc_nh/1e13, color = ebm_color, ls = ebm_ls)
noresm_snc_nh_line   = axs[0].plot(np.arange(1, 12+1), noresm_snc_nh/1e13, color = noresm_color, ls = noresm_ls)
era5_snc_nh_line   = axs[0].plot(np.arange(1, 12+1), era5_snc_nh/1e13, color = era5_color, ls = era5_ls)

# southern hemisphere
ebm_snc_sh_line = axs[0].plot(np.arange(1, 12+1), ebm_snc_sh/1e13, color = ebm_color, ls = ":")
noresm_snc_sh_line = axs[0].plot(np.arange(1, 12+1), noresm_snc_sh/1e13, color = noresm_color, ls = ":")
era5_snc_sh_line   = axs[0].plot(np.arange(1, 12+1), era5_snc_sh/1e13, color = era5_color, ls = ":")

# sea ice coverage
#-----------------

# northern hemsphere
ebm_sic_nh_line = axs[1].plot(np.arange(1, 12+1), ebm_sic_nh[0,:]/1e13, color = ebm_color, ls = ebm_ls)
noresm_sic_nh_line = axs[1].plot(np.arange(1, 12+1), noresm_sic_nh/1e13, color = noresm_color, ls = noresm_ls)
era5_sic_nh_line = axs[1].plot(np.arange(1, 12+1), era5_sic_nh/1e13, color = era5_color, ls = era5_ls)

# southern hemisphere
ebm_sic_sh_line = axs[1].plot(np.arange(1, 12+1), ebm_sic_sh[0,:]/1e13, color = ebm_color, ls = ":")
noresm_sic_sh_line = axs[1].plot(np.arange(1, 12+1), noresm_sic_sh/1e13, color = noresm_color, ls = ":")
era5_sic_sh_line = axs[1].plot(np.arange(1, 12+1), era5_sic_sh/1e13, color = era5_color, ls = ":")

# legend
#-------
axs[0].legend(handles = [ebm_snc_nh_line, noresm_snc_nh_line, era5_snc_nh_line], 
              labels = ["ZEMBA (NH)", "NorESM2 (NH)", "ERA5 (NH)"], frameon = False, ncols = 1, 
              loc = "ur", bbox_to_anchor=(0.5, 0.9), prop={'size':legend_fs})

axs[0].legend(handles = [ebm_snc_sh_line, noresm_snc_sh_line, era5_snc_sh_line], 
              labels = ["ZEMBA (SH)", "NorESM2 (SH)", "ERA5 (SH)"], frameon = False, ncols = 1, 
              loc = "ll", bbox_to_anchor=(0.1, 0.05), prop={'size':legend_fs})


axs[1].legend(handles = [ebm_sic_nh_line, noresm_sic_nh_line, era5_sic_nh_line], 
              labels = ["ZEMBA (NH)", "NorESM2 (NH)", "ERA5 (NH)"], frameon = False, ncols = 1, 
              loc = "cl", bbox_to_anchor=(0.1, 0.5), prop={'size':legend_fs})

axs[1].legend(handles = [ebm_sic_sh_line, noresm_sic_sh_line, era5_sic_sh_line], 
              labels = ["ZEMBA (SH)", "NorESM2 (SH)", "ERA5 (SH)"], frameon = False, ncols = 1, 
              loc = "cr", bbox_to_anchor=(0.85, 0.5), prop={'size':legend_fs})

# save
#-----

fig.save(os.getcwd()+"/output/plots/f05.png", dpi = 400)
fig.save(os.getcwd()+"/output/plots/f05.pdf", dpi = 400)



print('###########################')
print("Annual Mean Snow Cover")
print('###########################')

print("ZEMBA (NH)): " + str(round(ebm_snc_nh_global/1e13, 2)))
print("ZEMBA (SH)): " + str(round(ebm_snc_sh_global/1e13, 2)))

print("NorESM2 (NH)): " + str(round(noresm_snc_nh_global/1e13, 2)))
print("NorESM2 (SH)): " + str(round(noresm_snc_sh_global/1e13, 2)))

print("ERA5 (NH)): " + str(round(era5_snc_nh_global/1e13, 2)))
print("ERA5 (SH)): " + str(round(era5_snc_sh_global/1e13, 2))+'\n')

print('###########################')
print("Annual Mean Sea Ice Cover")
print('###########################')

print("ZEMBA (NH)): " + str(round(ebm_sic_nh_global/1e13, 2)))
print("ZEMBA (SH)): " + str(round(ebm_sic_sh_global/1e13, 2)))

print("NorESM2 (NH)): " + str(round(noresm_sic_nh_global/1e13, 2)))
print("NorESM2 (SH)): " + str(round(noresm_sic_sh_global/1e13, 2)))

print("ERA5 (NH)): " + str(round(era5_sic_nh_global/1e13, 2)))
print("ERA5 (SH)): " + str(round(era5_sic_sh_global/1e13, 2))+'\n')







