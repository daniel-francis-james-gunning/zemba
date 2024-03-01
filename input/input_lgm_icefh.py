# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

LGM input file.

<-----------------------------HERE = parameters/settings that can be changed

<-----------------------------COMMENT/UNCOMMENT = sections that can be commented/uncommented for different settings 
"""

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit
import xarray as xr
import os
import pandas as pd

# name of run.
name = 'output/equilibrium/lgm_icefh'

# path
script_path = os.path.dirname(os.getcwd())   # MIGHT NEED TO CHANGE PATHS TO ACCESS *OTHER DATA* FOLDER

#--------------
# Model version  <-------------------------------------------------------------------------------COMMENT/UNCOMMENT
#--------------

input_version = 'moist'   # moist EBM  

# input_version = 'dry'   # classic 'dry' EBM

#-----------------------------------------------------------------
# Model settings [e.g. for turning processes/feedbacks on or off.]
#-----------------------------------------------------------------

settings_zemba = Dict.empty(key_type=types.unicode_type, value_type=types.int8[:])

# snowfall [ 0 = OFF, 1 = ON (uniform coverage), 2 = ON (fractional coverage) ]
settings_zemba["snow"]=np.array([2], dtype = "int8") # <---------------------------------------- HERE
if input_version == 'dry':
    settings_zemba["snow"][0] = 0
    
# hydrological cycle [ 0 = OFF , 1 = ON ]
settings_zemba["hydro"]=np.array([1], dtype = "int8")  # <-------------------------------------- HERE
if input_version == 'dry':
    settings_zemba["hydro"][0] = 0 
    
# sea ice [ 0 = OFF , 1 = ON ]
settings_zemba["seaice"]=np.array([1], dtype = "int8") # <-------------------------------------- HERE

# meridional heat transport [ 0 = OFF , 1 = ON ]
settings_zemba["transport"]=np.array([1], dtype = "int8") # <----------------------------------- HERE

# hadley cell parameterization [ 0 = OFF , 1 = ON ]
settings_zemba["hadley_cell"]=np.array([1], dtype = "int8") # <--------------------------------- HERE

# atmospheric heat transport [ 0 = OFF , 1 = ON ]
settings_zemba["atm_transport"]=np.array([1], dtype = "int8") # <------------------------------- HERE

# ocean heat transport [ 0 = OFF , 1 = ON]
settings_zemba["ocn_transport"]=np.array([1], dtype = "int8") # <------------------------------- HERE

settings_zemba["height"]=np.array([0], dtype = "int8") 

# model version [ 0 = classic 'dry', 1 = moist]
if input_version == 'moist':
    settings_zemba['version'] = np.array([1], dtype = "int8")
if input_version == 'dry':
    settings_zemba['version'] = np.array([0], dtype = "int8")
if input_version != 'moist' and input_version != 'dry':
    raise Exception("Model version must be either 'moist' or 'dry'")

#------------
# Model input 
#------------

input_zemba = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])

# Set-up
#-------

input_zemba['nyrs']=np.array([3000.]) # no. of model years...           # <---------------------- HERE
input_zemba['occ']=np.array([-5.])    # mid-point of ocean circulation...# <--------------------- HERE
input_zemba['ikyr']=np.array([0.])    # kyr BP for orbital parameters# <------------------------- HERE

# Resolution [in degrees of latitude]
#-----------

input_zemba['res']=np.array([5.])  # <----------------------------------------------------------- HERE

if input_zemba['res'][0] == 5.:
    lat   = np.arange(-87.5, 87.5+5., 5.)
    latb  = np.arange(-90, 90+5., 5.)
    olat  = np.arange(-67.5, 77.5+5., 5.)
    olatb = np.arange(-70, 80+5., 5.)
    
if input_zemba['res'][0] == 2.5:
    lat   = np.arange(-88.75, 88.75+2.5, 2.5)
    latb  = np.arange(-90, 90+2.5, 2.5)
    olat  = np.arange(-68.75, 78.75+2.5, 2.5)
    olatb = np.arange(-70, 80+2.5, 2.5)
    
if input_zemba['res'][0] == 1.:
    lat = np.arange(-89.5, 89.5+1., 1.)
    latb = np.arange(-90, 90+1., 1.)
    olat = np.arange(-69.5, 79.5+1., 1.)
    olatb = np.arange(-70, 80+1., 1.)
    
if input_zemba['res'][0] != 5. and input_zemba['res'][0] != 2.5 and input_zemba['res'][0] != 1.:
    raise Exception("Sorry, the choices of model resolution are 1, 2.5 and 5")
        

# [option to modify strength of overturning, e.g. 1.03 is 3% increase]
#---------------------------------------------------------------------

input_zemba['strength_of_overturning']=np.array([1.]) # <------------------------------------------ HERE

# [option to modify strength of insolation]
#------------------------------------------

input_zemba['strength_of_insolation']=np.array([1.])  # <------------------------------------------ HERE

# [option to keep snow over land fixed to the pre-industrial.]  <---------------------------------- COMMENT/UNCOMMENT
#-------------------------------------------------------------

settings_zemba['fixed_snow'] = np.array([0], dtype = "int8") #--> OFF.
fixed_snow_fraction = None
fixed_snow_thick    = None
fixed_snow_melt     = None

# settings_zemba['fixed_snow'] = np.array([1], dtype = "int8") #--> ON.
# with open(os.getcwd()+'\\sensitivity_experiments\\pi\\pi_'+input_version+'_res5.0_fyr.pkl', 'rb') as f:
#     pi = pickle.load(f)
# fixed_snow_fraction = pi['snow_fraction_land']
# fixed_snow_thick    = pi['snow_thick']             
# fixed_snow_melt     = pi['melt']

# [option to keep sea ice fixed to the pre-industrial.  <------------------------------------------ COMMENT/UNCOMMENT
#-----------------------------------------------------

settings_zemba['fixed_seaice'] = np.array([0], dtype = "int8") #--> OFF.
fixed_si_fraction = None
fixed_si_thick = None
fixed_si_melt = None

# settings_zemba['fixed_seaice'] = np.array([1], dtype = "int8") #--> ON.
# with open(os.getcwd()+'\\sensitivity_experiments\\pi\\pi_'+input_version+'_res5.0_fyr.pkl', 'rb') as f:
#     pi = pickle.load(f)
# fixed_si_fraction = pi['si_fraction'] 
# fixed_si_thick    = pi['si_thick']       
# fixed_si_melt     = pi['si_melt_flux']    
    
# [option to keep land albedo fixed to the pre-industrial.]   <----------------------------------- COMMENT/UNCOMMENT
#----------------------------------------------------------

settings_zemba['fixed_land_albedo'] = np.array([0], dtype = "int8") #--> OFF.
fixed_land_albedo = None 
# settings_zemba['fixed_land_albedo'] = np.array([1], dtype = "int8") #--> ON.
# with open(os.getcwd()+'\\sensitivity_experiments\\pi\\pi_'+input_version+'_res5.0_fyr.pkl', 'rb') as f:
#     pi = pickle.load(f)
# fixed_land_albedo = pi['alpha_land']
    

# [option to keep ocean albedo fixed to the pre-industrial.]  <----------------------------------- COMMENT/UNCOMMENT
#-----------------------------------------------------------

settings_zemba['fixed_ocean_albedo'] = np.array([0], dtype = "int8") #--> OFF.
fixed_ocean_albedo = None
# settings_zemba['fixed_ocean_albedo'] = np.array([1], dtype = "int8") #--> ON.
# with open(os.getcwd()+'\\sensitivity_experiments\\pi\\pi_'+input_version+'_res5.0_fyr.pkl', 'rb') as f:
#     pi = pickle.load(f)
# fixed_ocean_albedo = pi['alpha_ocean'] 
    

# Land fractions <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#---------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_lf  = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[1], sep='\t').to_numpy()[:,0]
input_zemba['land_fraction'] = np.interp(lat, ice6g_lat, ice6g_lf)
#|
#|-> Ensure Antarctic continent occupies entire area south of 75S
if input_zemba['res'] == 1.:
    input_zemba['land_fraction'][0:12] = 1.
if input_zemba['res'] == 2.5:
    input_zemba['land_fraction'][0:6] = 1.      
if input_zemba['res'] == 5.:
    input_zemba['land_fraction'][0:3] = 1. 
    
# Aqua planet (no land.)
# input_zemba['land_fraction'] = np.zeros((lat.size))

# Land fractions boundaries
#--------------------------

input_zemba["land_fraction_bounds"] = np.interp(latb, lat, input_zemba["land_fraction"])


# Land elevation (m) <------------------------------------------------------------------------------COMMENT/UNCOMMENT
#-------------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
# ice6g_lat   = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_orog  = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[3], sep='\t').to_numpy()[:,0]
# input_zemba['land_height'] = np.interp(lat, ice6g_lat, ice6g_orog)

# LGM from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat   = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_orog  = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[4], sep='\t').to_numpy()[:,0]
input_zemba['land_height'] = np.interp(lat, ice6g_lat, ice6g_orog)


# Flat.
# input_zemba['land_height'] = np.zeros(( lat.size ))

# Ice fractions <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#--------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
# ice6g_lat = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_if  = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[5], sep='\t').to_numpy()[:,0]
# input_zemba['ice_fraction'] = np.interp(lat, ice6g_lat, ice6g_if)

# LGM from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_if  = pd.read_csv(script_path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[6], sep='\t').to_numpy()[:,0]
input_zemba['ice_fraction'] = np.interp(lat, ice6g_lat, ice6g_if)

# No ice.
# input_zemba['ice_fraction'] = np.zeros(( lat.size ))


# Cloud cover <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#------------

# NorESM2-MM PI Control Simulation (Seland et al., 2020.)
noresm2_lat   = pd.read_csv(script_path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[0], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]
noresm2_ccl   = pd.read_csv(script_path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[2], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]/100
noresm2_cco   = pd.read_csv(script_path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[3], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]/100
input_zemba['ccl'] = np.interp(lat, noresm2_lat, noresm2_ccl)
input_zemba['cco'] = np.interp(lat, noresm2_lat, noresm2_cco)

# Constant and ubiqitous cloud cover of 0.5 (for experimental purposes)
# input_zemba['ccl'] = np.zeros((lat.size))+0.5
# input_zemba['cco'] = np.zeros((lat.size))+0.5



# Other key model parameters
#---------------------------

# turbulent heat fluxes (m/s)
input_zemba['tbhfcl']=np.array([0.006])  # land  # <------------------------------------------ HERE
input_zemba['tbhfco']=np.array([0.005])  # ocean # <------------------------------------------ HERE

# cloud optical depth.
input_zemba['tau']=np.array([3.0]) # <-------------------------------------------------------- HERE

# atm. co2 conc.
input_zemba['co2']=np.array([284. * (44.01/28.97) * 1e-6]) # <-------------------------------- HERE

# albedo.
input_zemba['alphag']=np.array([0.15])    # bare ground  # <---------------------------------- HERE
input_zemba['alphas']=np.array([0.8])     # 'cold' snow  # <---------------------------------- HERE
input_zemba["alphaws"]=np.array([0.4])    # 'wet' snow 
input_zemba['alphasimx']=np.array([0.65]) # 'cold' sea ice  # <------------------------------- HERE
input_zemba["alphasimn"]=np.array([0.4])  # 'wet' sea ice  # <-------------------------------- HERE
input_zemba["alphai"]=np.array([0.8])     # land ice  # <------------------------------------- HERE

# maximum relative humidity (kg/kg)
input_zemba["r"]=np.array([80.]) # <---------------------------------------------------------- HERE

# GHG amplification factor.
input_zemba['GHG_amp']=np.array([1.3]) # <---------------------------------------------------- HERE

# eddy/gyre diffusion coefficient (m/s) [OCEAN]
input_zemba['do'] = np.zeros((olatb.size))
idxoc = np.where(olatb==input_zemba['occ'])[0][0]

input_zemba['do'][0:idxoc] = 5e10 / (60*60*24*365) # SH <------------------------------------- HERE
input_zemba['do'][idxoc:]  = 4e11 / (60*60*24*365) # NH <------------------------------------- HERE

# vertical diffusion coefficient (m/s)
input_zemba['dz0']=np.array([5e3 / (60*60*24*365)]) # <--------------------------------------- HERE

# interior horizontal diffusion coefficient (m/s)
input_zemba["dh"]=np.array([1.5e10 / (60*60*24*365)]) #<-------------------------------------- HERE

# atm. transport
#---------------

# hadley cell constant.
input_zemba["hadley_constant"] = np.array([1.03]) # maximum relative humidity <--------------- HERE
    
# atm. diffusion coefficient. (m/s)
input_zemba["dt"] = np.zeros((latb.size)) 
idxc = np.where(latb==0.)[0][0]

input_zemba['dt'][0:idxc] = 0.9e6 # SH <------------------------------------------------------- HERE
input_zemba['dt'][idxc:] = 1.1e6 # NH <-------------------------------------------------------- HERE



