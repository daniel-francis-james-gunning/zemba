# -*- coding: utf-8 -*-
"""
plot figure 5 - results section - pre-industrial snow cover and sea ice cover

@author: Daniel Gunning 
"""

import numpy as np
import proplot as pplt
import os
import pickle
import xarray as xr

# paths
output_path  = os.path.dirname(os.getcwd())
script_path  = os.path.dirname(output_path)
input_path   = script_path + '/input'

os.chdir(script_path)
from utilities import *

#------------------------------
# load zemba pre-industrial sim
#------------------------------

with open('output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
    pi_data = pickle.load(f)
    
pi    = pi_data['StateYear']
Var   = pi_data['Var']
INPUT = pi_data['Input']
    
# snow cover (land)
#------------------
 
zemba_snc_monthly = calculate_monthlies(pi["snow_fraction_land"], Var["lat"].size)
zemba_snc_nh      = np.nansum(zemba_snc_monthly[Var["idxnh"].astype("int")]*Var["larea"][Var["idxnh"].astype("int")].reshape(18, 1), axis=0)
zemba_snc_sh      = np.nansum(zemba_snc_monthly[Var["idxsh"].astype("int")]*Var["larea"][Var["idxsh"].astype("int")].reshape(18, 1), axis=0)

# global
zemba_snc_nh_global = np.average(zemba_snc_nh, weights = Var["days_in_months"])
zemba_snc_sh_global = np.average(zemba_snc_sh, weights = Var["days_in_months"])

# sea ice cover
#--------------

zemba_sic_nh = calculate_monthlies(pi["si_area_nh"], 1)
zemba_sic_sh = calculate_monthlies(pi["si_area_sh"], 1)

# global
zemba_sic_nh_global = np.average(zemba_sic_nh[0,:], weights = Var["days_in_months"])
zemba_sic_sh_global = np.average(zemba_sic_sh[0,:], weights = Var["days_in_months"])
 
#-----------------------------------
# load pre-industrial NorESM2 output
#-----------------------------------

# load annual data
noresm2 = np.loadtxt(script_path+'/other_data/noresm2/noresm2_snc_monthly.txt', skiprows=5, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
noresm2_sic_nh = noresm2[0,:]*1e12
noresm2_sic_sh = noresm2[1,:]*1e12
noresm2_snc_nh = noresm2[2,:]*1e12
noresm2_snc_sh = noresm2[3,:]*1e12

# global mean
noresm2_sic_nh_global = np.average(noresm2_sic_nh, weights = Var["days_in_months"])
noresm2_sic_sh_global = np.average(noresm2_sic_sh, weights = Var["days_in_months"])
noresm2_snc_nh_global = np.average(noresm2_snc_nh, weights = Var["days_in_months"])
noresm2_snc_sh_global = np.average(noresm2_snc_sh, weights = Var["days_in_months"])

#---------------------------
# ERA5 1940-1970 climatology
#---------------------------

# load annual data
era5 = np.loadtxt(script_path+'/other_data/era5/era5_snc_monthly.txt', skiprows=5, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])
era5_sic_nh = era5[0,:]*1e12
era5_sic_sh = era5[1,:]*1e12
era5_snc_nh = era5[2,:]*1e12
era5_snc_sh = era5[3,:]*1e12 

# global mean
era5_sic_nh_global = np.average(era5_sic_nh, weights = Var["days_in_months"])
era5_sic_sh_global = np.average(era5_sic_sh, weights = Var["days_in_months"])
era5_snc_nh_global = np.average(era5_snc_nh, weights = Var["days_in_months"])
era5_snc_sh_global = np.average(era5_snc_sh, weights = Var["days_in_months"])

#---------
# plotting
#---------

# constants
#----------

zemba_color = "black"
zemba_lw    = 1.
zemba_ls    = "-"

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

shape = [  
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],]

# figure and axes
fig, axs = pplt.subplots(shape, figsize = (8,3), sharey = False, sharex = False, grid = False)


# fonts
axs.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)

# x-axis
axs.format(xlocator = np.arange(1, 12+1), xminorlocator = [], xticklabels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"],
           xlabel = 'Month')

# format subplots
for i in np.arange(0,1+1):
    
    # y-axis
    axs[i].format(ylim = (0, 5), yminorlocator=np.arange(0, 5+1, 1), ylocator = np.arange(0, 5+1, 1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
# titles
axs[0].format(title = r'(a) PI areal extent (x10$^{13} \: m^{2}$) in snow coverage over land', titleloc = 'left')
axs[1].format(title = r'(b) PI areal extent (x10$^{13} \: m^{2}$) in sea ice cover', titleloc = 'left')


# plot snow cover
#----------------

# northern hemisphere
zemba_snc_nh_line    = axs[0].plot(np.arange(1, 12+1), zemba_snc_nh/1e13, color = zemba_color, ls = zemba_ls)
noresm_snc_nh_line = axs[0].plot(np.arange(1, 12+1), noresm2_snc_nh/1e13, color = noresm_color, ls = noresm_ls)
era5_snc_nh_line   = axs[0].plot(np.arange(1, 12+1), era5_snc_nh/1e13, color = era5_color, ls = era5_ls)

# southern hemisphere
zemba_snc_sh_line    = axs[0].plot(np.arange(1, 12+1), zemba_snc_sh/1e13, color = zemba_color, ls = ":")
noresm_snc_sh_line = axs[0].plot(np.arange(1, 12+1), noresm2_snc_sh/1e13, color = noresm_color, ls = ":")
era5_snc_sh_line   = axs[0].plot(np.arange(1, 12+1), era5_snc_sh/1e13, color = era5_color, ls = ":")

# sea ice cover
#--------------

# northern hemsphere
zemba_sic_nh_line    = axs[1].plot(np.arange(1, 12+1), zemba_sic_nh[0,:]/1e13, color = zemba_color, ls = zemba_ls)
noresm_sic_nh_line = axs[1].plot(np.arange(1, 12+1), noresm2_sic_nh/1e13, color = noresm_color, ls = noresm_ls)
era5_sic_nh_line   = axs[1].plot(np.arange(1, 12+1), era5_sic_nh/1e13, color = era5_color, ls = era5_ls)

# southern hemisphere
zemba_sic_sh_line    = axs[1].plot(np.arange(1, 12+1), zemba_sic_sh[0,:]/1e13, color = zemba_color, ls = ":")
noresm_sic_sh_line = axs[1].plot(np.arange(1, 12+1), noresm2_sic_sh/1e13, color = noresm_color, ls = ":")
era5_sic_sh_line   = axs[1].plot(np.arange(1, 12+1), era5_sic_sh/1e13, color = era5_color, ls = ":")

# legends
#--------

axs[0].legend(handles = [zemba_snc_nh_line, noresm_snc_nh_line, era5_snc_nh_line], 
              labels = ["ZEMBA (NH)", "NorESM2 (NH)", "ERA5 (NH)"], frameon = False, ncols = 1, 
              loc = "ur", bbox_to_anchor=(0.5, 0.9), prop={'size':legend_fs})

axs[0].legend(handles = [zemba_snc_sh_line, noresm_snc_sh_line, era5_snc_sh_line], 
              labels = ["ZEMBA (SH)", "NorESM2 (SH)", "ERA5 (SH)"], frameon = False, ncols = 1, 
              loc = "ll", bbox_to_anchor=(0.1, 0.05), prop={'size':legend_fs})


axs[1].legend(handles = [zemba_sic_nh_line, noresm_sic_nh_line, era5_sic_nh_line], 
              labels = ["ZEMBA (NH)", "NorESM2 (NH)", "ERA5 (NH)"], frameon = False, ncols = 1, 
              loc = "cl", bbox_to_anchor=(0.1, 0.5), prop={'size':legend_fs})

axs[1].legend(handles = [zemba_sic_sh_line, noresm_sic_sh_line, era5_sic_sh_line], 
              labels = ["ZEMBA (SH)", "NorESM2 (SH)", "ERA5 (SH)"], frameon = False, ncols = 1, 
              loc = "cr", bbox_to_anchor=(0.85, 0.5), prop={'size':legend_fs})

# save
#-----

fig.save(os.getcwd()+"/output/plots/f05.png", dpi= 300)
fig.save(os.getcwd()+"/output/plots/f05.pdf", dpi= 300)



print('###########################')
print("Annual Mean Snow Cover")
print('###########################')

print("ZEMBA (NH)): " + str(round(zemba_snc_nh_global/1e13, 2)))
print("ZEMBA (SH)): " + str(round(zemba_snc_sh_global/1e13, 2)))

print("NorESM2 (NH)): " + str(round(noresm2_snc_nh_global/1e13, 2)))
print("NorESM2 (SH)): " + str(round(noresm2_snc_sh_global/1e13, 2)))

print("ERA5 (NH)): " + str(round(era5_snc_nh_global/1e13, 2)))
print("ERA5 (SH)): " + str(round(era5_snc_sh_global/1e13, 2))+'\n')

print('###########################')
print("Annual Mean Sea Ice Cover")
print('###########################')

print("ZEMBA (NH)): " + str(round(zemba_sic_nh_global/1e13, 2)))
print("ZEMBA (SH)): " + str(round(zemba_sic_sh_global/1e13, 2)))

print("NorESM2 (NH)): " + str(round(noresm2_sic_nh_global/1e13, 2)))
print("NorESM2 (SH)): " + str(round(noresm2_sic_sh_global/1e13, 2)))

print("ERA5 (NH)): " + str(round(era5_sic_nh_global/1e13, 2)))
print("ERA5 (SH)): " + str(round(era5_sic_sh_global/1e13, 2))+'\n')







