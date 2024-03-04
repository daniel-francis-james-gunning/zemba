# -*- coding: utf-8 -*-
"""
plot figure 9 & 10 - results sections - LGM simulations...

@author: Daniel Gunning 
"""

import numpy as np
import proplot as pplt
import os
import pickle
import xarray as xr

# apths
output_path  = os.path.dirname(os.getcwd())
script_path  = os.path.dirname(output_path)
input_path   = script_path + '/input'

os.chdir(script_path)
from utilities import *

#----------------
# load zemba sims
#----------------

# list of experiments
experiments = {}

# pre-industrial
#---------------

with open(os.getcwd()+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
    experiments["pi"] = pickle.load(f)['StateYear']
with open(os.getcwd()+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
    Var = pickle.load(f)['Var']

# last glacial maximum
#---------------------

# ice fraction
with open(os.getcwd()+'/output/equilibrium/lgm_icef_moist_res5.0.pkl', 'rb') as f:
    experiments["lgm_icef"] = pickle.load(f)['StateYear']
    
# ice fraction + elevation
with open(os.getcwd()+'/output/equilibrium/lgm_icefh_moist_res5.0.pkl', 'rb') as f:
    experiments["lgm_icefh"] = pickle.load(f)['StateYear']

# ice fraction + elevation + co2
with open(os.getcwd()+'/output/equilibrium/lgm_icefh_co2_moist_res5.0.pkl', 'rb') as f:
    experiments["lgm_co2"] = pickle.load(f)['StateYear']
 
# ice fraction + elevation + co2 + insolation
with open(os.getcwd()+'/output/equilibrium/lgm_icefh_co2_inso_moist_res5.0.pkl', 'rb') as f:
    experiments["lgm_inso"] = pickle.load(f)['StateYear']
experiments["lgm"] = experiments["lgm_inso"] 

# ice fraction + elevation + co2 + inso + southward shift of ocean circulation
with open(os.getcwd()+'/output/equilibrium/lgm_icefh_co2_inso_oc10s_moist_res5.0.pkl', 'rb') as f:
    experiments["lgm_oc15s"] = pickle.load(f)['StateYear'] 

# + 75 % Overturning
with open(os.getcwd()+'/output/equilibrium/lgm_75%ov_moist_res5.0.pkl', 'rb') as f:
    experiments["lgm_75%ov"] = pickle.load(f)['StateYear']

# + 50% Overturning
with open(os.getcwd()+'/output/equilibrium/lgm_50%ov_moist_res5.0.pkl', 'rb') as f:
    experiments["lgm_50%ov"] = pickle.load(f)['StateYear']
    
#-------------------------
# Load data from zemba sims
#--------------------------

zemba = {}

# retrieve air temperatures
for names, values in experiments.items():
    
    # zonal mean temperature
    zemba[names+'_tas'] = values["Tax"].mean(axis=1)
    
    # global mean temperature
    zemba[names+'_tas_global'] = np.average(values["Tax"].mean(axis=1), weights = Var["sarea"])
    
    # temperature from 90S to 30S
    i1 = np.where(Var["latb"] == -90)[0][0]
    i2 = np.where(Var["latb"] == -30)[0][0] 
    zemba[names+'_tas_90S-30S'] = np.average(zemba[names+'_tas'][i1:i2], weights = Var["sarea"][i1:i2])
    
    # temperature from 30S to 30N
    i1 = np.where(Var["latb"] == -30)[0][0]
    i2 = np.where(Var["latb"] ==  30)[0][0] 
    zemba[names+'_tas_30S-30N'] = np.average(zemba[names+'_tas'][i1:i2], weights = Var["sarea"][i1:i2])
    
    # temperature from 30N to 90N
    i1 = np.where(Var["latb"] ==  30)[0][0]
    i2 = np.where(Var["latb"] ==  90)[0][0] 
    zemba[names+'_tas_30N-90N'] = np.average(zemba[names+'_tas'][i1:i2], weights = Var["sarea"][i1:i2])
     
# retrieve precipitation (in mm/day)
for names, values in experiments.items():
    
    # zonal mean precipitation
    zemba[names+'_precip'] = (values["precip_flux"].mean(axis=1)/1000)*(60*60*24)*1000
    
    # global mean precipitation
    zemba[names+'_precip_global'] = np.average(zemba[names+'_precip'], weights = Var["sarea"])
    
    # precipitation from 90S to 30S
    i1 = np.where(Var["latb"] == -90)[0][0]
    i2 = np.where(Var["latb"] == -30)[0][0] 
    zemba[names+'_precip_90S-30S'] = np.average(zemba[names+'_precip'][i1:i2], weights = Var["sarea"][i1:i2])
    
    # precipitation from 30S to 30N
    i1 = np.where(Var["latb"] == -30)[0][0]
    i2 = np.where(Var["latb"] ==  30)[0][0] 
    zemba[names+'_precip_30S-30N'] = np.average(zemba[names+'_precip'][i1:i2], weights = Var["sarea"][i1:i2])
    
    # precipitation from 30N to 90N
    i1 = np.where(Var["latb"] ==  30)[0][0]
    i2 = np.where(Var["latb"] ==  90)[0][0] 
    zemba[names+'_precip_30N-90N'] = np.average(zemba[names+'_precip'][i1:i2], weights = Var["sarea"][i1:i2])
 
# retrieve heat transport (PW)
for names, values in experiments.items():
    
    # zonal mean atmospheric transport
    zemba[names+'_atm'] = (values["mse_north"].mean(axis=1))/1e15
                                   
    # zonal mean dry static transport
    zemba[names+'_dry_north'] = (values["dry_north"].mean(axis=1))/1e15
                             
    # zonal mean dry latent transport
    zemba[names+'_latent_north'] = (values["latent_north"].mean(axis=1))/1e15
                                
    # zonal mean ocean overturning
    zemba[names+'_ocn_adv'] = np.concatenate((np.zeros((Var["idxnocs"].size)), 
                                            values["advf"].mean(axis=1)[0:Var['olatb'].size] + values["advf"].mean(axis=1)[5*Var['olatb'].size:6*Var['olatb'].size], 
                                            np.zeros((Var["idxnocn"].size)) ))/1e15
    
    # zonal mean eddy/gyre
    zemba[names+'_ocn_dif'] = np.concatenate((np.zeros((Var["idxnocs"].size)), 
                                            values["hdiffs"].mean(axis=1)[0:Var['olatb'].size], 
                                            np.zeros((Var["idxnocn"].size)) ))/1e15
    
    # zonal mean ocean heat transport
    zemba[names+'_ocn'] = np.concatenate(( np.zeros((Var["idxnocs"].size)), 
                                         values["hdiffs"].mean(axis=1)[0:Var['olatb'].size]+values["advf"].mean(axis=1)[0:Var['olatb'].size] + values["advf"].mean(axis=1)[5*Var['olatb'].size:6*Var['olatb'].size], 
                                         np.zeros((Var["idxnocn"].size)) ))/1e15
    
    # zonal mean total transport
    zemba[names+'_total'] = ( np.concatenate(( np.zeros((Var["idxnocs"].size)), 
                                            values["hdiffs"].mean(axis=1)[0:Var['olatb'].size]+values["advf"].mean(axis=1)[0:Var['olatb'].size] + values["advf"].mean(axis=1)[5*Var['olatb'].size:6*Var['olatb'].size], 
                                            np.zeros((Var["idxnocn"].size)) ))/1e15
                           
                           + (values["mse_north"].mean(axis=1))/1e15)
                           
#----------------------------
# Annan and Hargreaves (2022)
#----------------------------

# load data
#----------

# load annual data
annan_lat  = np.loadtxt(script_path+'/other_data/lgm/annan_lgm.txt', skiprows=5, usecols=0)
annan_mean = np.loadtxt(script_path+'/other_data/lgm/annan_lgm.txt', skiprows=5, usecols=1)
annan_max  = np.loadtxt(script_path+'/other_data/lgm/annan_lgm.txt', skiprows=5, usecols=2)
annan_min  = np.loadtxt(script_path+'/other_data/lgm/annan_lgm.txt', skiprows=5, usecols=3)

# surface area
zonal_area = 2*np.pi*(6371000**2)*np.cos(np.deg2rad(annan_lat))*np.deg2rad(np.diff(annan_lat)[0])

# multi-model regional mean
annan_90S_30S = np.average(annan_mean[0:30],  weights = zonal_area[0:30])
annan_30S_30N = np.average(annan_mean[30:60], weights = zonal_area[30:60])
annan_30N_90N = np.average(annan_mean[60:90], weights = zonal_area[60:90])

# global mean
annan_global = np.average(annan_mean, weights = zonal_area)

#--------------------------
# PMIP3 & PMIP4 MAT and MAP
#--------------------------

# load annual data
pmip_lat  = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=0)
pmip3_mat_mean = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=1)
pmip3_map_mean = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=2)
pmip4_mat_mean = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=3)
pmip4_map_mean = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=4)
pmip_mat_max   = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=5)
pmip_mat_min   = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=6)
pmip_map_max   = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=7)
pmip_map_min   = np.loadtxt(script_path+'/other_data/lgm/pmip_lgm.txt', skiprows=8, usecols=8)

# surface area
zonal_area = 2*np.pi*(6371000**2)*np.cos(np.deg2rad(pmip_lat))*np.deg2rad(np.diff(pmip_lat)[0])

# pmip3 mat regional mean
pmip3_mat_90S_30S = np.average(pmip3_mat_mean[0:60],  weights = zonal_area[0:60])
pmip3_mat_30S_30N = np.average(pmip3_mat_mean[60:120], weights = zonal_area[60:120])
pmip3_mat_30N_90N = np.average(pmip3_mat_mean[120:180], weights = zonal_area[120:180])

# pmip4 mat regional mean
pmip4_mat_90S_30S = np.average(pmip4_mat_mean[0:60],  weights = zonal_area[0:60])
pmip4_mat_30S_30N = np.average(pmip4_mat_mean[60:120], weights = zonal_area[60:120])
pmip4_mat_30N_90N = np.average(pmip4_mat_mean[120:180], weights = zonal_area[120:180])

# pmip3 map regional mean
pmip3_map_90S_30S = np.average(pmip3_map_mean[0:60],  weights = zonal_area[0:60])
pmip3_map_30S_30N = np.average(pmip3_map_mean[60:120], weights = zonal_area[60:120])
pmip3_map_30N_90N = np.average(pmip3_map_mean[120:180], weights = zonal_area[120:180])

# pmip4 map regional mean
pmip4_map_90S_30S = np.average(pmip4_map_mean[0:60],  weights = zonal_area[0:60])
pmip4_map_30S_30N = np.average(pmip4_map_mean[60:120], weights = zonal_area[60:120])
pmip4_map_30N_90N = np.average(pmip4_map_mean[120:180], weights = zonal_area[120:180])

# global mean
pmip3_mat_global = np.average(pmip3_mat_mean, weights = zonal_area)
pmip4_mat_global = np.average(pmip4_mat_mean, weights = zonal_area)
pmip3_map_global = np.average(pmip3_map_mean, weights = zonal_area)
pmip4_map_global = np.average(pmip4_map_mean, weights = zonal_area)

#---------------------------
# Plot LGM  - pre-industrial 
#---------------------------

# constants
#----------

zemba_color = "black"
zemba_lw    = 1
zemba_ls    = "-"

Osman_color = "blue5"
Osman_lw    = 1
Osman_ls    = "-."

PMIP4_color = "yellow5"
PMIP4_lw    = 1
PMIP4_ls    = "-."

PMIP3_color = "orange5"
PMIP3_lw    = 1
PMIP3_ls    = "-."

Tierney_color = "green5"
Tierney_lw    = 1
Tierney_ls    = "-."

Annan_color = "blue9"
Annan_lw    = 1
Annan_ls    = "-."

legend_fs = 7.

# formating
#----------

shape = [  
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4, 4, 4]]

# figure and axes
fig, axs = pplt.subplots(shape, figsize = (8,4), sharey = False, sharex = False, hspace=1.8, grid = False)

# fonts
axs.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)
            
# x-axis
axs.format(xlim = (-90, 90), xlocator = np.arange(-90, 120, 30), xminorlocator = np.arange(-90, 100, 10),)
        
# format top left plot
for i in np.arange(0, 1):
    
    # y-axis
    axs[i].format(ylim = (-28, 4), ylocator = np.arange(-28, 4+4, 4), yminorlocator = np.arange(-28, 4+4, 4))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False) 

# format top right plot
for i in np.arange(1, 2):
    
    # y-axis
    axs[i].format(ylim = (-2, 2), ylocator = np.arange(-2, 2+1), yminorlocator = np.arange(-2, 2+1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False) 

# format bottom plots
for i in np.arange(2, 3+1):
    
    # y-axis
    axs[i].format(ylim = (-1, 1), ylocator = np.arange(-1, 1+1, 0.5), yminorlocator = np.arange(-1, 1+1, 0.5))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
    # x-axis
    axs[i].format(xlabel = "Latitude", xformatter='deglat')

# titles
axs[0].format(title = r'(a) LGM − PI Zonal Temperatures ($^{\circ}$C)', titleloc = "left")
axs[1].format(title = r'(b) LGM − PI Zonal Precipitation (mm/day)', titleloc = "left")
axs[2].format(title = r'(c) LGM − PI Zonal Heat Transport (PW) for ZEMBA', titleloc = "left")
axs[3].format(title = r'(d) LGM − PI Atmospheric Heat Transport (PW) for ZEMBA', titleloc = "left")

# plot PI - LGM temperatures
#---------------------------

PMIP4_pi_lgm_line       = axs[0].plot(pmip_lat,  pmip4_mat_mean, color = PMIP4_color, lw = PMIP4_lw, ls = PMIP4_ls)
PMIP3_pi_lgm_line       = axs[0].plot(pmip_lat,  pmip3_mat_mean, color = PMIP3_color, lw = PMIP3_lw, ls = PMIP3_ls)
Annan_pi_lgm_tas_line   = axs[0].plot(annan_lat, annan_mean, color = Annan_color, lw = Annan_lw, linestyle = Annan_ls)
zemba_pi_lgm_line         = axs[0].plot(Var["lat"], zemba["lgm_tas"] - zemba["pi_tas"] , color = "black", lw = 1.5, ls = '-')

import matplotlib.patches as mpatches
empty_patch = mpatches.Patch(color="none")

# plot range of temperature from other datasets
Annan_range = axs[0].fill_between(annan_lat, annan_max, annan_min, color = "blue9", alpha = 0.2)
PMIP_range  = axs[0].fill_between(pmip_lat, pmip_mat_max, pmip_mat_min, color = PMIP4_color, alpha = 0.2)

# legend
axs[0].legend(handles = [zemba_pi_lgm_line, empty_patch, empty_patch, PMIP3_pi_lgm_line, PMIP4_pi_lgm_line, PMIP_range, Annan_pi_lgm_tas_line, Annan_range, empty_patch], 
              labels = ["ZEMBA", "", "", "PMIP3", "PMIP4", "PMIP3/PMIP4 range", "Annan2023", "Annan2023 range", ""], ncols = 3, loc = "ll", frameon = False, prop={'size':legend_fs})

# plot PI - LGM precipitation
#----------------------------

zemba_pi_lgm_line  = axs[1].plot(Var["lat"], zemba["lgm_precip"] - zemba["pi_precip"] , color = "black", lw = 1.5, ls = '-')
PMIP4_pi_lgm_line  = axs[1].plot(pmip_lat, pmip4_map_mean, color = PMIP4_color, lw = PMIP4_lw, ls = PMIP4_ls)
PMIP3_pi_lgm_line  = axs[1].plot(pmip_lat, pmip3_map_mean, color = PMIP3_color, lw = PMIP3_lw, ls = PMIP3_ls)

# range of precipitation from other datasets
PMIP_range = axs[1].fill_between(np.arange(-89.5, 89.5+1), pmip_map_max, pmip_map_min, color = PMIP4_color, alpha = 0.2)

# legend
axs[1].legend(handles = [zemba_pi_lgm_line, empty_patch, empty_patch, PMIP3_pi_lgm_line, PMIP4_pi_lgm_line, PMIP_range], 
              labels = ["ZEMBA", "", "", "PMIP3", "PMIP4", "PMIP3/PMIP4 range"], ncols = 3, loc = "ll", frameon = False, prop={'size':legend_fs})

# plot LGM - PI heat transport
#-----------------------------

zemba_pi_lgm_tot_line = axs[2].plot(Var["latb"], zemba["lgm_total"] - zemba["pi_total"], color = zemba_color, lw = 1.5, ls = '-')
zemba_pi_lgm_atm_line = axs[2].plot(Var["latb"], zemba["lgm_atm"] - zemba["pi_atm"], color = 'red9', lw = 1.5, ls = '-')
zemba_pi_lgm_oce_line = axs[2].plot(Var["latb"], zemba["lgm_ocn"] - zemba["pi_ocn"], color = 'blue9', lw = 1.5, ls = '-')
zemba_pi_lgm_oce_adv_line = axs[2].plot(Var["latb"], zemba["lgm_ocn_adv"] - zemba["pi_ocn_adv"], color = 'blue9', lw = 1., ls = ':')
zemba_pi_lgm_oce_dif_line = axs[2].plot(Var["latb"], zemba["lgm_ocn_dif"] - zemba["pi_ocn_dif"], color = 'blue9', lw = 0.5, ls = '-.')

# legend
axs[2].legend(handles = [zemba_pi_lgm_tot_line, zemba_pi_lgm_atm_line, empty_patch, zemba_pi_lgm_oce_line, zemba_pi_lgm_oce_adv_line, zemba_pi_lgm_oce_dif_line], 
              labels = ["Total", "Atmosphere", "", "Ocean", "Ocean - Overturning", "Ocean - Eddy/Gyre"], ncols = 3, loc = "ll", frameon = False, prop={'size':legend_fs})

# plot LGM - PI atmospheric heat transport
#-----------------------------------------

zemba_pi_lgm_atm_line = axs[3].plot(Var["latb"], zemba["lgm_atm"] - zemba["pi_atm"], color = 'red9', lw = 1.5, ls = '-')
zemba_pi_lgm_dry_line = axs[3].plot(Var["latb"], zemba["lgm_dry_north"] - zemba["pi_dry_north"], color = 'red9', lw = 1., ls = ':')
zemba_pi_lgm_lat_line = axs[3].plot(Var["latb"], zemba["lgm_latent_north"] - zemba["pi_latent_north"], color = 'green9', lw = 0.5, ls = '-.')

# legend
axs[3].legend(handles = [zemba_pi_lgm_atm_line, zemba_pi_lgm_dry_line, zemba_pi_lgm_lat_line], 
              labels = ["Atmosphere", "Atmosphere - Dry Static", "Atmosphere - Latent"], ncols = 2, loc = "ll", frameon = False, prop={'size':legend_fs})

# save figure
#------------

fig.save(os.getcwd()+"/output/plots/f09.png", dpi= 300)
fig.save(os.getcwd()+"/output/plots/f09.pdf", dpi= 300)


#-----------------------
# Plot LGM decomposition
#-----------------------

# formating
#----------

shape = [  
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],
        [1, 1, 1, 1, 2, 2, 2, 2],]


# figure and axes
fig, axs = pplt.subplots(shape, figsize = (9,3), sharey = False, sharex = False, hspace=1.5, grid = False)

# fonts
axs.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)
            
# x-axis
axs.format(xlim = (-90, 90), xlocator = np.arange(-90, 120, 30), xminorlocator = np.arange(-90, 100, 10), xlabel = "Latitude", xformatter='deglat')

# legend font size
legend_fs=7.

# format top left plot
for i in np.arange(0, 1):
    
    # y-axis
    axs[i].format(ylim = (-28, 4), ylocator = np.arange(-28, 4+4, 4), yminorlocator = np.arange(-28, 4+4, 4))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
# format top right plot
for i in np.arange(1, 2):
    
    # y-axis
    axs[i].format(ylim = (-2, 2), ylocator = np.arange(-2, 2+1), yminorlocator = np.arange(-2, 2+1))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")

# titles
axs[0].format(title = r'(A) LGM - PI Zonal Temperature ($^{\circ}$C) for various experiments',titleloc = 'left')
axs[1].format(title = r'(B) LGM - PI Zonal Precipitation (mm/day) for various experiments',titleloc = 'left')


# plot LGM - PI zonal temperatures
#---------------------------------

zemba_lgm_pi_icefh_line  = axs[0].plot(Var["lat"], zemba["lgm_icefh_tas"] - zemba["pi_tas"], color = "blue9", lw = 1., linestyle = "-.")
zemba_lgm_pi_co2_line    = axs[0].plot(Var["lat"], zemba["lgm_co2_tas"] - zemba["pi_tas"], color = "red9", lw = 1, linestyle = "-.")
zemba_lgm_pi_inso_line   = axs[0].plot(Var["lat"], zemba["lgm_inso_tas"] - zemba["pi_tas"], color = "black", lw = 1.5, linestyle = "-")
zemba_lgm_pi_oc10s_line  = axs[0].plot(Var["lat"], zemba["lgm_oc15s_tas"] - zemba["pi_tas"], color = "black", lw = 1, linestyle = "-.")
zemba_lgm_pi_75Ov_line   = axs[0].plot(Var["lat"], zemba["lgm_75%ov_tas"] - zemba["pi_tas"], color = "green5", lw = 1, linestyle = "-.")
zemba_lgm_pi_50Ov_line   = axs[0].plot(Var["lat"], zemba["lgm_50%ov_tas"] - zemba["pi_tas"], color = "purple", lw = 1, linestyle = "-.")

# range of temperature from other datasets
Annan_range = axs[0].fill_between(annan_lat, annan_max, annan_min, color = "blue9", alpha = 0.2)
PMIP_range  = axs[0].fill_between(np.arange(-89.5, 89.5+1), pmip_mat_max, pmip_mat_min, color = PMIP4_color, alpha = 0.2)

# legend       
axs[0].legend(handles = [zemba_lgm_pi_icefh_line, zemba_lgm_pi_co2_line, zemba_lgm_pi_inso_line, zemba_lgm_pi_oc10s_line, zemba_lgm_pi_75Ov_line, zemba_lgm_pi_50Ov_line], 
              labels = ["Ice", r'CO$_{2}$', "Inso", r'Oc: 15$^{\circ}$S', "75% Ov", "50% Ov"], 
              ncols = 1, loc = "ll", frameon = False, prop={'size':legend_fs}) 

# plot LGM - PI zonal precipitation
#----------------------------------

zemba_lgm_pi_icef_line   = axs[1].plot(Var["lat"], zemba["lgm_icefh_precip"] - zemba["pi_precip"], color = "blue9", lw = 1, linestyle = "-.")
zemba_lgm_pi_co2_line    = axs[1].plot(Var["lat"], zemba["lgm_co2_precip"] - zemba["pi_precip"], color = "red9", lw = 1, linestyle = "-.")
zemba_lgm_pi_inso_line   = axs[1].plot(Var["lat"], zemba["lgm_inso_precip"] - zemba["pi_precip"], color = "black", lw = 1, linestyle = "-")
zemba_lgm_pi_oc10s_line  = axs[1].plot(Var["lat"], zemba["lgm_oc15s_precip"] - zemba["pi_precip"], color = "black", lw = 1, linestyle = "-.")
zemba_lgm_pi_75Ov_line   = axs[1].plot(Var["lat"], zemba["lgm_75%ov_precip"] - zemba["pi_precip"], color = "green5", lw = 1, linestyle = "-.")
zemba_lgm_pi_50Ov_line   = axs[1].plot(Var["lat"], zemba["lgm_50%ov_precip"] - zemba["pi_precip"], color = "purple", lw = 1, linestyle = "-.")

# range of precipitation from other datasets
axs[1].fill_between(np.arange(-89.5, 89.5+1), pmip_map_max, pmip_map_min, color = PMIP4_color, alpha = 0.2)

# legend       
axs[1].legend(handles = [zemba_lgm_pi_icefh_line, zemba_lgm_pi_co2_line, zemba_lgm_pi_inso_line, zemba_lgm_pi_oc10s_line, zemba_lgm_pi_75Ov_line, zemba_lgm_pi_50Ov_line], 
              labels = ["Ice", r'CO$_{2}$', "Inso", r'Oc: 15$^{\circ}$S', "75% Ov", "50% Ov"], 
              ncols = 1, loc = "ll", frameon = False, prop={'size':legend_fs})                         


fig.save(os.getcwd()+"/output/plots/f10.png", dpi= 300)
fig.save(os.getcwd()+"/output/plots/f10.pdf", dpi= 300)


# Save global mean variables in DataFrame
#----------------------------------------
import pandas as pd

# surface air temperature
air_data = pd.DataFrame(np.array([[zemba["lgm_tas_90S-30S"]-zemba["pi_tas_90S-30S"], 
                                   zemba["lgm_tas_30S-30N"]-zemba["pi_tas_30S-30N"], 
                                   zemba["lgm_tas_30N-90N"]-zemba["pi_tas_30N-90N"], 
                                   zemba["lgm_tas_global"]-zemba["pi_tas_global"]],
                                  
                                    [annan_90S_30S,
                                     annan_30S_30N,
                                     annan_30N_90N,
                                     annan_global],
                                
                                    [pmip3_mat_90S_30S,
                                     pmip3_mat_30S_30N,
                                     pmip3_mat_30N_90N,
                                     pmip3_mat_global],
                              
                                    [pmip4_mat_90S_30S,
                                     pmip4_mat_30S_30N,
                                     pmip4_mat_30N_90N,
                                     pmip4_mat_global],
        
                              ]),
                        
                    index = ["Zemba", "Annan2021", "PMIP3", "PMIP4"],
                    columns = ["90S-30S", "30S-30N", "30N-90N", "Global"],).round(decimals=2)

# precipitation data
precip_data = pd.DataFrame(np.array([[zemba["lgm_precip_90S-30S"]-zemba["pi_precip_90S-30S"], 
                               zemba["lgm_precip_30S-30N"]-zemba["pi_precip_30S-30N"], 
                               zemba["lgm_precip_30N-90N"]-zemba["pi_precip_30N-90N"], 
                               zemba["lgm_precip_global"]-zemba["pi_precip_global"]],
                                      
                              [pmip3_map_90S_30S,
                               pmip3_map_30S_30N,
                               pmip3_map_30N_90N,
                               pmip3_map_global],
                              
                              [pmip4_map_90S_30S,
                               pmip4_map_30S_30N,
                               pmip4_map_30N_90N,
                               pmip4_map_global],
                              
                              ]),
                    index = ["Zemba", "PMIP3", "PMIP4"],
                    columns = ["90S-30S", "30S-30N", "30N-90N", "Global"]).round(decimals=2)

# temperature data for LGM decomposition
series_data_air = pd.DataFrame(np.array([[zemba["lgm_icefh_tas_90S-30S"]-zemba["pi_tas_90S-30S"], 
                               zemba["lgm_icefh_tas_30S-30N"]-zemba["pi_tas_30S-30N"], 
                               zemba["lgm_icefh_tas_30N-90N"]-zemba["pi_tas_30N-90N"], 
                               zemba["lgm_icefh_tas_global"]-zemba["pi_tas_global"]],
                                     
                                [zemba["lgm_co2_tas_90S-30S"]-zemba["pi_tas_90S-30S"], 
                                 zemba["lgm_co2_tas_30S-30N"]-zemba["pi_tas_30S-30N"], 
                                 zemba["lgm_co2_tas_30N-90N"]-zemba["pi_tas_30N-90N"], 
                                 zemba["lgm_co2_tas_global"]-zemba["pi_tas_global"]],
                                
                                [zemba["lgm_inso_tas_90S-30S"]-zemba["pi_tas_90S-30S"], 
                                 zemba["lgm_inso_tas_30S-30N"]-zemba["pi_tas_30S-30N"], 
                                 zemba["lgm_inso_tas_30N-90N"]-zemba["pi_tas_30N-90N"], 
                                 zemba["lgm_inso_tas_global"]-zemba["pi_tas_global"]],
                                
                                [zemba["lgm_oc15s_tas_90S-30S"]-zemba["pi_tas_90S-30S"], 
                                 zemba["lgm_oc15s_tas_30S-30N"]-zemba["pi_tas_30S-30N"], 
                                 zemba["lgm_oc15s_tas_30N-90N"]-zemba["pi_tas_30N-90N"], 
                                 zemba["lgm_oc15s_tas_global"]-zemba["pi_tas_global"]],
                                
                                [zemba["lgm_75%ov_tas_90S-30S"]-zemba["pi_tas_90S-30S"], 
                                 zemba["lgm_75%ov_tas_30S-30N"]-zemba["pi_tas_30S-30N"], 
                                 zemba["lgm_75%ov_tas_30N-90N"]-zemba["pi_tas_30N-90N"], 
                                 zemba["lgm_75%ov_tas_global"]-zemba["pi_tas_global"]],
                                
                                [zemba["lgm_50%ov_tas_90S-30S"]-zemba["pi_tas_90S-30S"], 
                                 zemba["lgm_50%ov_tas_30S-30N"]-zemba["pi_tas_30S-30N"], 
                                 zemba["lgm_50%ov_tas_30N-90N"]-zemba["pi_tas_30N-90N"], 
                                 zemba["lgm_50%ov_tas_global"]-zemba["pi_tas_global"]],

                              ]),
                           
                    index = ["Ice", "Co2", "Tnso", "Ocean Centre: 15S", "75% Overturning", "50% Overturning"],
                    columns = ["90S-30S", "30S-30N", "30N-90N", "Global"],).round(decimals=2)


# precipitation data for LGM decomposition
series_data_precip = pd.DataFrame(np.array([[zemba["lgm_icefh_precip_90S-30S"]-zemba["pi_precip_90S-30S"], 
                                zemba["lgm_icefh_precip_30S-30N"]-zemba["pi_precip_30S-30N"], 
                                zemba["lgm_icefh_precip_30N-90N"]-zemba["pi_precip_30N-90N"], 
                                zemba["lgm_icefh_precip_global"]-zemba["pi_precip_global"]],
                                      
                                [zemba["lgm_co2_precip_90S-30S"]-zemba["pi_precip_90S-30S"], 
                                 zemba["lgm_co2_precip_30S-30N"]-zemba["pi_precip_30S-30N"], 
                                 zemba["lgm_co2_precip_30N-90N"]-zemba["pi_precip_30N-90N"], 
                                 zemba["lgm_co2_precip_global"]-zemba["pi_precip_global"]],
                                
                                [zemba["lgm_inso_precip_90S-30S"]-zemba["pi_precip_90S-30S"], 
                                 zemba["lgm_inso_precip_30S-30N"]-zemba["pi_precip_30S-30N"], 
                                 zemba["lgm_inso_precip_30N-90N"]-zemba["pi_precip_30N-90N"], 
                                 zemba["lgm_inso_precip_global"]-zemba["pi_precip_global"]],
                                
                                [zemba["lgm_oc15s_precip_90S-30S"]-zemba["pi_precip_90S-30S"], 
                                 zemba["lgm_oc15s_precip_30S-30N"]-zemba["pi_precip_30S-30N"], 
                                 zemba["lgm_oc15s_precip_30N-90N"]-zemba["pi_precip_30N-90N"], 
                                 zemba["lgm_oc15s_precip_global"]-zemba["pi_precip_global"]],
                                
                                [zemba["lgm_75%ov_precip_90S-30S"]-zemba["pi_precip_90S-30S"], 
                                 zemba["lgm_75%ov_precip_30S-30N"]-zemba["pi_precip_30S-30N"], 
                                 zemba["lgm_75%ov_precip_30N-90N"]-zemba["pi_precip_30N-90N"], 
                                 zemba["lgm_75%ov_precip_global"]-zemba["pi_precip_global"]],
                                
                                [zemba["lgm_50%ov_precip_90S-30S"]-zemba["pi_precip_90S-30S"], 
                                 zemba["lgm_50%ov_precip_30S-30N"]-zemba["pi_precip_30S-30N"], 
                                 zemba["lgm_50%ov_precip_30N-90N"]-zemba["pi_precip_30N-90N"], 
                                 zemba["lgm_50%ov_precip_global"]-zemba["pi_precip_global"]],

                              ]),
                           
                    index = ["Ice", "Co2", "Tnso", "Ocean Centre: 15S", "75% Overturning", "50% Overturning"],
                    columns = ["90S-30S", "30S-30N", "30N-90N", "Global"],).round(decimals=2)


print('###########################')
print("Air Temperature")
print('###########################')
print(air_data)
print('\n')

print('###########################')
print("Precipitation")
print('###########################')
print(precip_data)
print('\n')

print('###########################')
print("LGM experiments - air temp")
print('###########################')
print(series_data_air)
print('\n')

print('###########################')
print("LGM experiments - precip")
print('###########################')
print(series_data_precip)
print('\n')


#----------------------------------------
# Supplementary figure 1 - heat transport
#----------------------------------------

# formating
#----------

shape = [  
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [1, 1, 1, 2, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [4, 4, 4, 5, 5, 5, 6, 6, 6],
        [7, 7, 7, 8, 8, 8, 9, 9, 9],
        [7, 7, 7, 8, 8, 8, 9, 9, 9],
        [7, 7, 7, 8, 8, 8, 9, 9, 9],
        [7, 7, 7, 8, 8, 8, 9, 9, 9],]

# figure and axes
fig, axs = pplt.subplots(shape, figsize = (12,6), sharey = False, sharex = False, hspace=1.5, grid = False)

# fonts
axs.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)

# format contour plots        
axs[0:1+1].format(ylim = (-90, 90), yformatter = 'deglat', ylocator = np.arange(-90, 120, 30), yminorlocator = np.arange(-90, 100, 10))
axs[3:4+1].format(ylim = (-90, 90), yformatter = 'deglat', ylocator = np.arange(-90, 120, 30), yminorlocator = np.arange(-90, 100, 10))
axs[6:7+1].format(ylim = (-90, 90), yformatter = 'deglat', ylocator = np.arange(-90, 120, 30), yminorlocator = np.arange(-90, 100, 10))

axs[0:1+1].format(xlim = (1, 365), xlocator = np.arange(15, 365, 30), xminorlocator = [],
            xticklabels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

axs[3:4+1].format(xlim = (1, 365), xlocator = np.arange(15, 365, 30), xminorlocator = [],
            xticklabels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

axs[6:7+1].format(xlim = (1, 365), xlocator = np.arange(15, 365, 30), xminorlocator = [],
            xticklabels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])

# format line plots
axs[2].format(xlim = (-90, 90), xformatter = 'deglat', xlocator = np.arange(-90, 120, 30), xminorlocator = np.arange(-90, 100, 10))
axs[5].format(xlim = (-90, 90), xformatter = 'deglat', xlocator = np.arange(-90, 120, 30), xminorlocator = np.arange(-90, 100, 10))
axs[8].format(xlim = (-90, 90), xformatter = 'deglat', xlocator = np.arange(-90, 120, 30), xminorlocator = np.arange(-90, 100, 10))
axs[2].format(ylim = (-5, 6), ylocator = np.arange(-5, 6+1, 1), yminorlocator = [])
axs[5].format(ylim = (-5, 6), ylocator = np.arange(-5, 6+1, 1), yminorlocator = [])
axs[8].format(ylim = (-5, 6), ylocator = np.arange(-5, 6+1, 1), yminorlocator = [])
axs[2].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
axs[5].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
axs[8].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")

for i in np.arange(0, 5+1): # remove x-axis in locations
    axs[i].xaxis.set_ticklabels([])
    axs[i].xaxis.set_ticks_position('none')
    axs[i].xaxis.set_tick_params(labelbottom=False) 
    
for i in [1,4,7]: # remove y-axis in locations
    axs[i].yaxis.set_ticklabels([])
    axs[i].yaxis.set_ticks_position('none')
    axs[i].yaxis.set_tick_params(labelbottom=False) 

# mesh grid
days, lat = np.meshgrid(Var["ndays"], Var["lat"])
 
# v max and min values
vmax  =  200
vmin  = -150
vmax2 = 150
vmin2 = -160

# titles
axs[0].format(title = r'(a) LGM total heat transport convergence (W/m$^{2}$)', titleloc = 'left')
axs[1].format(title = r'(b) LGM - PI total heat transport convergence (W/m$^{2}$)', titleloc = 'left')
axs[2].format(title = r'(c) Northward Total Heat Transport (PW)', titleloc = 'left')
axs[3].format(title = r'(d) LGM atmospheric heat transport convergence (W/m$^{2}$)', titleloc = 'left')
axs[4].format(title = r'(e) LGM - PI atmospheric heat transport convergence (W/m$^{2}$)', titleloc = 'left')
axs[5].format(title = r'(f) Northward Atmospheric Heat Transport (PW)', titleloc = 'left')
axs[6].format(title = r'(g) LGM ocean heat transport convergence (W/m$^{2}$)', titleloc = 'left')
axs[7].format(title = r'(h) LGM - PI ocean heat transport convergence (W/m$^{2}$)', titleloc = 'left')
axs[8].format(title = r'(j) Northward Ocean Heat Transport (PW)', titleloc = 'left')


# plot total heat transport
#--------------------------

# LGM seasonal
axs[0].contourf(days, lat, 
                 
                  experiments["lgm"]["mse_conv"] + (np.concatenate((np.zeros((Var["idxnocs"].size, 365)), experiments["lgm"]["ocean_conv"][0:Var["olat"].size, :], np.zeros((Var["idxnocn"].size, 365))))),
                 
                  vmin = vmin, vmax = vmax, colorbar = 'r', cmap = 'vik')

# LGM - PI seasonal
axs[1].contourf(days, lat, 
                 
                  experiments["lgm"]["mse_conv"] + (np.concatenate((np.zeros((Var["idxnocs"].size, 365)), experiments["lgm"]["ocean_conv"][0:Var["olat"].size, :], np.zeros((Var["idxnocn"].size, 365)))))
                  -
                  (experiments["pi"]["mse_conv"] + (np.concatenate((np.zeros((Var["idxnocs"].size, 365)), experiments["pi"]["ocean_conv"][0:Var["olat"].size, :], np.zeros((Var["idxnocn"].size, 365)))))),
                 
                  vmin = vmin2, vmax = vmax2, colorbar = 'r', cmap = 'vik')

# LGM and PI annual-mean northward transport
line1 = axs[2].plot(Var["latb"], 
            experiments["lgm"]["mse_north"].mean(axis=1)/1e15 + np.concatenate((np.zeros((Var["idxnocs"].size)), experiments["lgm"]["advf"].mean(axis=1)[0:Var['olatb'].size] + experiments["lgm"]["advf"].mean(axis=1)[5*Var['olatb'].size:6*Var['olatb'].size] + experiments["lgm"]["hdiffs"].mean(axis=1)[0:Var['olatb'].size], np.zeros((Var["idxnocn"].size))))/1e15,
            color = "blue9",
            )
line2 = axs[2].plot(Var["latb"], 
            experiments["pi"]["mse_north"].mean(axis=1)/1e15 + np.concatenate((np.zeros((Var["idxnocs"].size)), experiments["pi"]["advf"].mean(axis=1)[0:Var['olatb'].size] + experiments["pi"]["advf"].mean(axis=1)[5*Var['olatb'].size:6*Var['olatb'].size] + experiments["pi"]["hdiffs"].mean(axis=1)[0:Var['olatb'].size], np.zeros((Var["idxnocn"].size))))/1e15,
            color = "black"
            )
    
axs[2].legend(handles = [line1, line2], labels = ["LGM", "PI"], frameon = False, prop={'size':7})                         


# plot atmospheric heat transport
#--------------------------------

# LGM seasonal
axs[3].contourf(days, lat, 
                 
                  experiments["lgm"]["mse_conv"],
                 
                  vmin = vmin, vmax = vmax, colorbar = 'r', cmap = 'vik')

# LGM - PI seasonal
axs[4].contourf(days, lat, 
                 
                  experiments["lgm"]["mse_conv"]
                  -
                  experiments["pi"]["mse_conv"],
                 
                  vmin = vmin2, vmax = vmax2, colorbar = 'r', cmap = 'vik')

# LGM and PI annual-mean northward transport
axs[5].plot(Var["latb"], 
            
            experiments["lgm"]["mse_north"].mean(axis=1)/1e15,
            
            color = "blue9"
            )
axs[5].plot(Var["latb"], 
            
            experiments["pi"]["mse_north"].mean(axis=1)/1e15,
            
            color = "black"
            )

# legend       
axs[5].legend(handles = [line1, line2], 
              labels = ["LGM", "PI"], 
              frameon = False, prop={'size':6})  



# plot ocean heat transport
#--------------------------

# LGM seasonal
axs[6].contourf(days, lat, 
                 
                  (np.concatenate((np.zeros((Var["idxnocs"].size, 365)), experiments["lgm"]["ocean_conv"][0:Var["olat"].size, :], np.zeros((Var["idxnocn"].size, 365))))),
                 
                  vmin = vmin, vmax = vmax, colorbar = 'r', cmap = 'vik')

# LGM - PI seasonal
axs[7].contourf(days, lat, 
                 
                  (np.concatenate((np.zeros((Var["idxnocs"].size, 365)), experiments["lgm"]["ocean_conv"][0:Var["olat"].size, :], np.zeros((Var["idxnocn"].size, 365)))))
                  -
                  (np.concatenate((np.zeros((Var["idxnocs"].size, 365)), experiments["pi"]["ocean_conv"][0:Var["olat"].size, :], np.zeros((Var["idxnocn"].size, 365))))),
                 
                  vmin = vmin2, vmax = vmax2, colorbar = 'r', cmap = 'vik')

# LGM and PI annual-mean northward transport
axs[8].plot(Var["latb"], 
            np.concatenate((np.zeros((Var["idxnocs"].size)), experiments["lgm"]["advf"].mean(axis=1)[0:Var['olatb'].size] + experiments["lgm"]["advf"].mean(axis=1)[5*Var['olatb'].size:6*Var['olatb'].size] + experiments["lgm"]["hdiffs"].mean(axis=1)[0:Var['olatb'].size], np.zeros((Var["idxnocn"].size))))/1e15,
            color = "blue9"
            )
axs[8].plot(Var["latb"], 
            np.concatenate((np.zeros((Var["idxnocs"].size)), experiments["pi"]["advf"].mean(axis=1)[0:Var['olatb'].size] + experiments["pi"]["advf"].mean(axis=1)[5*Var['olatb'].size:6*Var['olatb'].size] + experiments["pi"]["hdiffs"].mean(axis=1)[0:Var['olatb'].size], np.zeros((Var["idxnocn"].size))))/1e15,
            color = "black"
            )
# legend       
axs[8].legend(handles = [line1, line2], 
              labels = ["LGM", "PI"], 
              frameon = False, prop={'size':6})  

# contour plots for LGM sea ice extent
axs[1].contour(days, lat, experiments["lgm"]["si_fraction"], levels = np.arange(0.2, 0.8, 0.3), ls = "-.", lw = 0.5, alpha = 1.0, color = "black")
axs[4].contour(days, lat, experiments["lgm"]["si_fraction"], levels = np.arange(0.2, 0.8, 0.3), ls = "-.", lw = 0.5, alpha = 1.0, color = "black")
axs[7].contour(days, lat, experiments["lgm"]["si_fraction"], levels = np.arange(0.2, 0.8, 0.3), ls = "-.", lw = 0.5, alpha = 1.0, color = "black")

# contour plots for PI sea ice extent
axs[1].contour(days, lat, experiments["pi"]["si_fraction"], levels = np.arange(0.2, 0.8, 0.3), ls = "-.", lw = 0.5, alpha = 1.0, color = "purple")
axs[4].contour(days, lat, experiments["pi"]["si_fraction"], levels = np.arange(0.2, 0.8, 0.3), ls = "-.", lw = 0.5, alpha = 1.0, color = "purple")
axs[7].contour(days, lat, experiments["pi"]["si_fraction"], levels = np.arange(0.2, 0.8, 0.3), ls = "-.", lw = 0.5, alpha = 1.0, color = "purple")

# save
fig.save(os.getcwd()+"/output/plots/fA1.png", dpi= 300)
fig.save(os.getcwd()+"/output/plots/fA1.pdf", dpi= 300)