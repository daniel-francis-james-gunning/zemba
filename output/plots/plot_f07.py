# -*- coding: utf-8 -*-
"""
plot figure 7 - results sections - pre-industrial transport

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

with open(output_path+'/output_equili_pi_res5.0.pkl', 'rb') as f:
    pi_sim = pickle.load(f)
    
pi    = pi_sim['StateYear']
Var   = pi_sim['Var']
INPUT = pi_sim['Input']


# northward flux of moist static energy in the atmosphere
zemba_atm = np.nanmean(pi["mse_north"], axis = 1)/1e15
zemba_atm_max_sh = -np.min(zemba_atm[0:int(Var["idxeq"])])
zemba_atm_max_nh = np.max(zemba_atm[int(Var["idxeq"]):])

# northward flux of dry static energy in the atmosphere
zemba_dse = np.nanmean(pi["dry_north"], axis = 1)/1e15

# northward flux of latent energy in the atmosphere
zemba_latent = np.nanmean(pi["latent_north"], axis = 1)/1e15

# northward flux of ocean heat
zemba_ocean = np.nanmean(pi['advf'][0:Var['olatb'].size, :] + pi['advf'][Var['olatb'].size*5:Var['olatb'].size*6, :] + pi["hdiffs"][0:Var['olatb'].size, :], axis = 1)/1e15 
zemba_ocean = np.concatenate((np.zeros((Var["idxnocs"].size)), zemba_ocean, np.zeros((Var["idxnocn"].size))))
zemba_ocean_max_sh = -np.min(zemba_ocean[0:int(Var["idxeq"])])
zemba_ocean_max_nh = np.max(zemba_ocean[int(Var["idxeq"]):])

#-----------------------------------
# load pre-industrial NorESM2 output
#-----------------------------------

def inferred_heat_transport(energy_in, lat):
    
    '''
    Infers northward heat transport from energy imbalance.
    '''
    
    # load modules
    from scipy import integrate
    
    # latitude in radians
    latr = np.deg2rad(lat)
    
    # cosine of latitude
    coslat = np.cos(latr)
    
    # weighted-mean energy flux
    field = coslat*energy_in
    
    # integral of energy flux for each latitude
    integral = integrate.cumtrapz(field, x=latr, initial=0)
    
    # scale
    result = (1E-15 * 2 * np.math.pi * 6371e3**2 * integral)
    
    return result

# load annual data
noresm2_lat     = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=0)
noresm2_pr      = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=2)
noresm2_prsn    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=3)
noresm2_evap    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=4)
noresm2_rsdt    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=5)
noresm2_rsut    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=6)
noresm2_rsds    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=7)
noresm2_rsus    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=8)
noresm2_rlut    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=9)
noresm2_rlds    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=10)
noresm2_rlus    = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=11)
noresm2_shf     = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=12)
noresm2_lhf     = np.loadtxt(input_path+'/other_data/noresm2/noresm2_annual.txt', skiprows=5, usecols=13)

# net radiation at TOA
rtnet_noresm2 = (noresm2_rsdt - noresm2_rsut) - noresm2_rlut

# net radiation at SFC
rsnet_noresm2 = (noresm2_rsds - noresm2_rsus) + (noresm2_rlds - noresm2_rlus)

# net radiation of atmosphere
ratmnet_noresm2 = rtnet_noresm2 - rsnet_noresm2

# upwards sensible heat flux
shf_noresm2 = noresm2_shf

# evaporation latent heat flux 
lhf_noresm2 =  noresm2_evap*2.5e6

# evaporation 
evap_noresm2 =  noresm2_evap

# precipitation 
precip_noresm2 =  noresm2_pr

# precipitation latent heat flux  
precip_latent_noresm2 =  noresm2_pr*2.5e6

# snowfall latent heat flux  
snowfall_latent_noresm2 =  noresm2_prsn*334000

# net heat flux into atmosphere
Fatm_noresm2 = ratmnet_noresm2 + lhf_noresm2 + snowfall_latent_noresm2 + shf_noresm2

# net heat flux into surface
Fs_noresm2 = rsnet_noresm2 - snowfall_latent_noresm2 - shf_noresm2 - lhf_noresm2

# latent heat flux
Flatent_noresm2 = lhf_noresm2 - precip_latent_noresm2  

# total heat transport from TOA energy flux imbalance
noresm2_total = inferred_heat_transport(rtnet_noresm2, noresm2_lat)

# atmospheric heat transport from atmospheric energy flux imbalance
noresm2_atm = inferred_heat_transport(Fatm_noresm2, noresm2_lat)

# ocean heat transport from surface energy flux imbalance
noresm2_ocean = inferred_heat_transport(Fs_noresm2, noresm2_lat)

# latent heat transport in atmosphere from moisture imbalance
noresm2_latent = inferred_heat_transport(Flatent_noresm2, noresm2_lat)

# dry static transport in atmosphereas residual
noresm2_dse = noresm2_atm - noresm2_latent 

# maximum and minimum fluxes
noresm2_ocean_max_sh = -np.min(noresm2_ocean[0:int(noresm2_lat.size/2)])
noresm2_ocean_max_nh = np.max(noresm2_ocean[int(noresm2_lat.size/2):])
noresm2_atm_max_sh   = -np.min(noresm2_atm[0:int(noresm2_lat.size/2)])
noresm2_atm_max_nh   = np.max(noresm2_atm[int(noresm2_lat.size/2):])

#---------
# plotting
#---------

# constants
#----------

zemba_color = "black"
zemba_lw    = 1
zemba_ls    = "-"

noresm_color = "blue9"
noresm_lw    = 1
noresm_ls    = "-"

era5_color = "red9"
era5_lw    = 1
era5_ls    = "-"

legend_fs = 7.

# shape
shape = [  
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],]

# figure and axes
fig, axs = pplt.subplots(shape, figsize = (10,3), sharey = False, sharex = False, grid = False)


# fonts
axs.format(ticklabelsize=7, ticklabelweight='normal', ylabelsize=8, ylabelweight='normal',
            xlabelsize=8, xlabelweight='normal', titlesize=8, titleweight='normal',)

# x-axis
locatorx      = np.arange(-90, 120, 30)
minorlocatorx = np.arange(-90, 100, 10)
axs.format(xminorlocator = minorlocatorx, xlocator = locatorx, xlim = (-90, 90))
axs.format(xlabel = "Latitude", xformatter='deglat')   

# format top left subplot
for i in np.arange(0,1):
    
    # y-axis
    axs[i].format(ylim = (-6, 6), yminorlocator=np.arange(-6, 6+1, 1), ylocator = np.arange(-6, 6+2, 2))
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
      
# format middle plot
for i in np.arange(1,2):
    
    # y-axis
    axs[i].format(ylim = (-6, 6), yminorlocator=[], ylocator = np.arange(-6, 6+1, 1))
    axs[i].yaxis.set_ticklabels([])
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
# format right plot
for i in np.arange(2,3):
    
    # y-axis
    axs[i].format(ylim = (-6, 6), yminorlocator=[], ylocator = np.arange(-6, 6+1, 1))
    axs[i].yaxis.set_ticklabels([])
    
    # hline
    axs[i].axhline(0, Var["lat"].min(), Var["lat"].max(), color = "black", lw = 0.2, linestyle ="--")
    
# titles
axs[0].format(title = r'(a) Total Heat Transport (PW)', titleloc = 'left')
axs[1].format(title = r'(b) Atmospheric and Ocean Heat Transport (PW)', titleloc = 'left')
axs[2].format(title = r'(c) Dry Static and Latent Heat Transport (PW)', titleloc = 'left')

# plot total heat transport
#--------------------------

zemba_total_line    = axs[0].plot(Var["latb"], zemba_atm + zemba_ocean, color = zemba_color, lw = zemba_lw, ls = zemba_ls)
noresm_total_line = axs[0].plot(noresm2_lat, noresm2_total, color = noresm_color, lw = noresm_lw, ls = noresm_ls)

# legend
axs[0].legend(handles = [zemba_total_line, noresm_total_line],
              labels  = ["ZEMBA", "NorESM2"], 
                          frameon = False, loc = "ul", bbox_to_anchor=(0.2, 0.9), 
                          ncols = 1, prop={'size':legend_fs})


# plot atmospheric + ocean
#-------------------------

zemba_atm_line = axs[1].plot(Var["latb"], zemba_atm, color = zemba_color, lw = zemba_lw, ls = zemba_ls)
noresm_atm_line = axs[1].plot(noresm2_lat, noresm2_atm, color = noresm_color, lw = noresm_lw, ls = noresm_ls)
zemba_ocean_line = axs[1].plot(Var["latb"], zemba_ocean, color = zemba_color, lw = zemba_lw, ls = ":")
noresm_ocean_line = axs[1].plot(noresm2_lat, noresm2_ocean, color = noresm_color, lw = noresm_lw, ls = ":")

# legend
axs[1].legend(handles = [zemba_atm_line, noresm_atm_line],
              labels  = ["ZEMBA (Atmosphere)", "NorESM2 (Atmosphere)"], 
                          frameon = False, loc = "ul", bbox_to_anchor=(0.1, 0.9), 
                          ncols = 1, prop={'size':legend_fs})

# legend
axs[1].legend(handles = [zemba_ocean_line, noresm_ocean_line],
              labels  = ["ZEMBA (Ocean)", "NorESM2 (Ocean)"], 
                          frameon = False, loc = "lr", bbox_to_anchor=(0.9, 0.2), 
                          ncols = 1, prop={'size':legend_fs})

# plot atmospheric partition
#---------------------------

zemba_dse_line       = axs[2].plot(Var["latb"], zemba_dse, color = zemba_color, lw = zemba_lw, ls = zemba_ls)
noresm_dse_line    = axs[2].plot(noresm2_lat, noresm2_dse, color = noresm_color, lw = noresm_lw, ls = noresm_ls)
zemba_latent_line    = axs[2].plot(Var["latb"], zemba_latent, color = zemba_color, lw = zemba_lw, ls = ":", alpha = 0.5)
noresm_latent_line = axs[2].plot(noresm2_lat, noresm2_latent, color = noresm_color, lw = noresm_lw, ls = ":", alpha = 0.5)

# legend
axs[2].legend(handles = [zemba_dse_line, noresm_dse_line],
              labels  = ["ZEMBA (Dry Static)", "NorESM2 (Dry Static)"], 
                          frameon = False, loc = "ul", bbox_to_anchor=(0.2, 0.9), 
                          ncols = 1, prop={'size':legend_fs})

# legend
axs[2].legend(handles = [zemba_latent_line, noresm_latent_line,],
              labels  = ["ZEMBA (Latent)","NorESM2 (Latent)"], 
                          frameon = False, loc = "lr", bbox_to_anchor=(0.9, 0.15), 
                          ncols = 1, prop={'size':legend_fs})

# save figure
fig.save(os.getcwd()+"/output/plots/f07.png", dpi= 300)
fig.save(os.getcwd()+"/output/plots/f07.pdf", dpi= 300)

print('###########################')
print("Peak Atmospheric Heat Transport....")
print('###########################')

print("pyzemba (SH): " + str(round(zemba_atm_max_sh, 2)))
print("pyzemba (NH): " + str(round(zemba_atm_max_nh, 2)))

print("NorESM2 (SH): " + str(round(noresm2_atm_max_sh, 2)))
print("NorESM2 (NH): " + str(round(noresm2_atm_max_nh, 2))+'\n')

# print("ERA5 (SH): " + str(round(era5_atm_max_sh, 2)))
# print("ERA5 (NH): " + str(round(era5_atm_max_nh, 2)))

print('###########################')
print("Peak Ocean Heat Transport....")
print('###########################')

print("pyzemba (SH): " + str(round(zemba_ocean_max_sh, 2)))
print("pyzemba (NH): " + str(round(zemba_ocean_max_nh, 2)))

print("NorESM2 (SH): " + str(round(noresm2_ocean_max_sh, 2)))
print("NorESM2 (NH): " + str(round(noresm2_ocean_max_nh, 2))+'\n')

# print("ERA5 (SH): " + str(round(era5_ocean_max_sh, 2)))
# print("ERA5 (NH): " + str(round(era5_ocean_max_nh, 2)))









