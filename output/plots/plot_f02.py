# -*- coding: utf-8 -*-
"""
plot figure 2 - results section - pre-industrial air temperature

@author: Daniel Gunning 
"""

import numpy as np
import proplot as pplt
import os
import pickle
import xarray as xr

# path
output_path  = os.path.dirname(os.getcwd())
script_path  = os.path.dirname(output_path)
input_path   = script_path + '/input'

os.chdir(script_path)
from utilities import *

#------------------------------
# load zemba pre-industrial sim
#------------------------------

with open('output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
    pi_sim = pickle.load(f)
    
pi    = pi_sim['StateYear']
Var   = pi_sim['Var']
INPUT = pi_sim['Input']
     
# annual-mean surface air temperature
#------------------------------------

zemba_tas_annual = pi["Tax"].mean(axis=1) - Var["K"]
zemba_tas_global = round(global_pymean(zemba_tas_annual, Var), 2)

# june-july-august mean surface air temperature
#----------------------------------------------

zemba_tas_jja = pi["Tax"][:,151:242+1].mean(axis=1) - Var["K"] 

# december-january-february mean surface air temperature
#-------------------------------------------------------

zemba_tas_djf = ((np.append(pi["Tax"][:,0:58+1], pi["Tax"][:,334:], axis = 1)).mean(axis=1)) - Var["K"] 

#-----------------------------------
# load pre-industrial NorESM2 output
#-----------------------------------

# load data
noresm2_lat             = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=0)
noresm2_tas_annual      = np.loadtxt(script_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=1)
noresm2_tas_monthly_nc  = np.loadtxt(script_path+'/other_data/noresm2/noresm2_t2m_monthly.txt', skiprows=5, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])

# interpolate data
noresm2_tas_annual  = np.interp(Var['lat'], noresm2_lat, noresm2_tas_annual) - Var['K']
noresm2_tas_monthly = np.zeros((Var['lat'].size, 12)) 
for i in np.arange(0,12):
    noresm2_tas_monthly[:,i]=np.interp(Var['lat'], noresm2_lat, noresm2_tas_monthly_nc[:,i])- Var['K']
    
# global_mean 
noresm2_tas_global = global_mean2(noresm2_tas_annual, Var["lat"], Var["dlat"])

# jja mean
noresm2_tas_jja = ( ((noresm2_tas_monthly[:,5]*30.) +
                    (noresm2_tas_monthly[:,6]*31.) +
                    (noresm2_tas_monthly[:,7]*31.))
            
                   /   
                   (30.+31.+31.) ) 

# djf mean
noresm2_tas_djf = ( ((noresm2_tas_monthly[:,11]*31.) +
                     (noresm2_tas_monthly[:,0]*31.) +
                     (noresm2_tas_monthly[:,1]*28.))
                   /
                   (31.+31.+28.) ) 

#-------------------------
# load era5 1940-1970 data
#-------------------------

# load data
era5_lat             = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=0)
era5_tas_annual      = np.loadtxt(script_path+'/other_data/era5/era5_annual.txt', skiprows=5, usecols=1)
era5_tas_monthly_nc  = np.loadtxt(script_path+'/other_data/era5/era5_t2m_monthly.txt', skiprows=5, usecols=[1,2,3,4,5,6,7,8,9,10,11,12])

# interpolate data
era5_tas_annual  = np.interp(Var['lat'], era5_lat, era5_tas_annual) - Var['K']
era5_tas_monthly = np.zeros((Var['lat'].size, 12)) 
for i in np.arange(0,12):
    era5_tas_monthly[:,i]=np.interp(Var['lat'], era5_lat, era5_tas_monthly_nc[:,i]) - Var['K']
    
# global_mean 
era5_tas_global   = global_mean2(era5_tas_annual, Var["lat"], Var["dlat"])

# jja mean
era5_tas_jja = ( ((era5_tas_monthly[:,5]*30.) +
                 (era5_tas_monthly[:,6]*31.) +
                 (era5_tas_monthly[:,7]*31.))
                /
                (30.+31.+31.) ) 

# djf mean
era5_tas_djf = ( ((era5_tas_monthly[:,11]*31.) +
                 (era5_tas_monthly[:,0]*31.) +
                 (era5_tas_monthly[:,1]*28.))
                    /
                (31.+31.+28.) ) 

#---------
# plotting
#---------

# colors and line widths
#-----------------------

zemba_color = "black"
zemba_lw    = 1.
zemba_ls    = "-"

noresm_color = "blue9"
noresm_lw    = 1.
noresm_ls    = "-"

era5_color = "red9"
era5_lw    = 1.
era5_ls    = "-"

zemba_minus_noresm_color = "blue9"
zemba_minus_noresm_lw    = 1.
zemba_minus_noresm_ls    = "-."

zemba_minus_era5_color = "red9"
zemba_minus_era5_lw    = 1.
zemba_minus_era5_ls    = ":"

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
axs.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)

# x-axis
axs.format(xlim = (-90, 90), xlocator = np.arange(-90, 120, 30), xminorlocator = np.arange(-90, 100, 10),)

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
zemba_annual_line    = axs[0].plot(Var["lat"], zemba_tas_annual, color = zemba_color, lw = zemba_lw, linestyle = zemba_ls)
era5_annual_line   = axs[0].plot(Var["lat"], era5_tas_annual, color = era5_color, lw = era5_lw, linestyle = era5_ls)
noresm_annual_line = axs[0].plot(Var["lat"], noresm2_tas_annual, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls)

# legends
axs[0].legend(handles = [zemba_annual_line, noresm_annual_line, era5_annual_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})

# plot differences
zemba_minus_noresm_annual_line = axs[3].plot(Var["lat"], zemba_tas_annual - noresm2_tas_annual, color = zemba_minus_noresm_color, lw = zemba_minus_noresm_lw, linestyle = zemba_minus_noresm_ls)
zemba_minus_era5_annual_line   = axs[3].plot(Var["lat"], zemba_tas_annual - era5_tas_annual, color = zemba_minus_era5_color, lw = zemba_minus_era5_lw, linestyle = zemba_minus_era5_ls)

# difference legends
axs[3].legend(handles = [zemba_minus_noresm_annual_line, zemba_minus_era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})

# plot december - january - february
#-----------------------------------

# lines
zemba_djf_line    = axs[1].plot(Var["lat"], zemba_tas_djf, color = zemba_color, lw = zemba_lw, linestyle = zemba_ls, alpha = 1)
noresm_djf_line = axs[1].plot(Var["lat"], noresm2_tas_djf, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
era5_djf_line   = axs[1].plot(Var["lat"], era5_tas_djf, color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)
axs[1].legend(handles = [zemba_djf_line, noresm_djf_line, era5_djf_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})

# plot difference
zemba_minus_noresm_djf_line = axs[4].plot(Var["lat"], zemba_tas_djf - noresm2_tas_djf, color = zemba_minus_noresm_color, lw = zemba_minus_noresm_lw, linestyle = zemba_minus_noresm_ls)
zemba_minus_era5_djf_line   = axs[4].plot(Var["lat"], zemba_tas_djf - era5_tas_djf, color = zemba_minus_era5_color, lw = zemba_minus_era5_lw, linestyle = zemba_minus_era5_ls)
axs[4].legend(handles = [zemba_minus_noresm_djf_line, zemba_minus_era5_djf_line, era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})

# plot june - july - august
#--------------------------

# lines
zemba_jja_line    = axs[2].plot(Var["lat"], zemba_tas_jja, color = zemba_color, lw = zemba_lw, linestyle = zemba_ls, alpha = 1)
noresm_jja_line = axs[2].plot(Var["lat"], noresm2_tas_jja, color = noresm_color, lw = noresm_lw, linestyle = noresm_ls, alpha = 1)
era5_jja_line   = axs[2].plot(Var["lat"], era5_tas_jja, color = era5_color, lw = era5_lw, linestyle = era5_ls, alpha = 1)
axs[2].legend(handles = [zemba_jja_line, noresm_jja_line, era5_jja_line], labels = ["ZEMBA", "NorESM2", "ERA5 (1940-1970)"], ncols = 1, loc = "c", frameon = False, prop={'size':legend_fs})

# plot difference
zemba_minus_noresm_jja_line = axs[5].plot(Var["lat"], zemba_tas_jja - noresm2_tas_jja, color = zemba_minus_noresm_color, lw = zemba_minus_noresm_lw, linestyle = zemba_minus_noresm_ls)
zemba_minus_era5_jja_line   = axs[5].plot(Var["lat"], zemba_tas_jja - era5_tas_jja, color = zemba_minus_era5_color, lw = zemba_minus_era5_lw, linestyle = zemba_minus_era5_ls)
leg = axs[5].legend(handles = [zemba_minus_noresm_jja_line, zemba_minus_era5_jja_line, era5_annual_line], labels = ["ZEMBA - NorESM2", "ZEMBA - ERA5 (1940-1970)"], ncols = 1, loc = "lc", frameon = False, prop={'size':legend_fs})

fig.save(os.getcwd()+"/output/plots/f02.png", dpi= 300)
fig.save(os.getcwd()+"/output/plots/f02.pdf", dpi= 300)

#------------------------------------------------------------------------------
# Print global mean air temperatures
#------------------------------------------------------------------------------

print('###########################')
print('Global Mean Air Temperature')
print('###########################')   
print("ZEMBA: " + str(zemba_tas_global))
print("NorESM2: " + str(noresm2_tas_global))
print("ERA5: " + str(era5_tas_global) + '\n')