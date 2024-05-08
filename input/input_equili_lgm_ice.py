<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

LGM (ice only) input file...

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
import pickle

# NAME OF RUN.
name = 'equili_lgm_ice'

# PATH
path = os.getcwd()   # MIGHT NEED TO CHANGE PATHS TO ACCESS *OTHER DATA* FOLDER

#--------------
# MODEL VERSION  <-------------------------------------------------------------------------------COMMENT/UNCOMMENT
#--------------

input_version = 'moist'  # moist EBM

# input_version = 'dry'    # dry 'classic' EBM

#------------
# Model prior  <-------------------------------------------------------------------------------COMMENT/UNCOMMENT
#------------

# prior = 'PI'  # initalize with pre-industrial run...

# prior = 'LGM' # initalize with last-glacial maximum run...

prior = None    # intialize with arbitrary state...

#---------------
# Model settings [e.g. for turning processes/feedbacks on or off.]
#---------------

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

# influence of height [ 0 = OFF , 1 = ON]
settings_zemba["height"]=np.array([0], dtype = "int8")        # <------------------------------- HERE

# model version [ 0 = classic 'dry', 1 = moist]
if input_version == 'moist':
    settings_zemba['version'] = np.array([1], dtype = "int8")
if input_version == 'dry':
    settings_zemba['version'] = np.array([0], dtype = "int8")
if input_version != 'moist' and input_version != 'dry':
    raise Exception("Model version must be either: 'moist' or 'dry'")

#------------
# Model input 
#------------

input_zemba = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])

# set-up
#-------

input_zemba['nyrs']=np.array([3000.]) # no. of model years...            # <-------------------- HERE
input_zemba['occ']=np.array([-5.])    # mid-point of ocean circulation...# <-------------------- HERE
input_zemba['ikyr']=np.array([0.])    # kyr BP for orbital parameters    # <-------------------- HERE

# Resolution [in degrees of latitude]
#-----------

input_zemba['res']=np.array([5.0])  # <-------------------------------------------------------- HERE

if input_zemba['res'][0] == 5.:  # if 5 degree
    lat   = np.arange(-87.5, 87.5+5., 5.)
    latb  = np.arange(-90, 90+5., 5.)
    olat  = np.arange(-67.5, 77.5+5., 5.)
    olatb = np.arange(-70, 80+5., 5.)
    
if input_zemba['res'][0] == 2.5:   # if 2.5 degree
    lat   = np.arange(-88.75, 88.75+2.5, 2.5)
    latb  = np.arange(-90, 90+2.5, 2.5)
    olat  = np.arange(-68.75, 78.75+2.5, 2.5)
    olatb = np.arange(-70, 80+2.5, 2.5)
    
if input_zemba['res'][0] == 1.:    # if 1. degree
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
# with open(path+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
#     pi = pickle.load(f)['StateYear']
# fixed_snow_fraction = pi['snow_fraction_land']
# fixed_snow_thick    = pi['snow_thick']             
# fixed_snow_melt     = pi['melt']


# [option to keep sea ice fixed to the pre-industrial.]  <----------------------------------------- COMMENT/UNCOMMENT
#------------------------------------------------------

settings_zemba['fixed_seaice'] = np.array([0], dtype = "int8") #--> OFF.
fixed_si_fraction = None
fixed_si_thick = None
fixed_si_melt = None

# settings_zemba['fixed_seaice'] = np.array([1], dtype = "int8") #--> ON.
# with open(path+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
#     pi = pickle.load(f)['StateYear']
# fixed_si_fraction = pi['si_fraction'] 
# fixed_si_thick    = pi['si_thick']       
# fixed_si_melt     = pi['si_melt_flux']    
    
# [option to keep land albedo fixed to the pre-industrial.]   <----------------------------------- COMMENT/UNCOMMENT
#----------------------------------------------------------

settings_zemba['fixed_land_albedo'] = np.array([0], dtype = "int8") #--> OFF.
fixed_land_albedo = None 

# settings_zemba['fixed_land_albedo'] = np.array([1], dtype = "int8") #--> ON.
# with open(path+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
#     pi = pickle.load(f)['StateYear']
# fixed_land_albedo = pi['alpha_land']
    

# [option to keep ocean albedo fixed to the pre-industrial.]  <----------------------------------- COMMENT/UNCOMMENT
#-----------------------------------------------------------

settings_zemba['fixed_ocean_albedo'] = np.array([0], dtype = "int8") #--> OFF.
fixed_ocean_albedo = None

# settings_zemba['fixed_ocean_albedo'] = np.array([1], dtype = "int8") #--> ON.
# with open(path+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
#     pi = pickle.load(f)['StateYear']
# fixed_ocean_albedo = pi['alpha_ocean'] 
    

# Land fractions <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#---------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_lf  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[1], sep='\t').to_numpy()[:,0]
input_zemba['land_fraction'] = np.interp(lat, ice6g_lat, ice6g_lf)

# LGM from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
# ice6g_lat = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_lf  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[2], sep='\t').to_numpy()[:,0]
# input_zemba['land_fraction'] = np.interp(lat, ice6g_lat, ice6g_lf)

#|
#|-> Ensure Antarctic continent occupies entire area south of 75S
if input_zemba['res'] == 1.:
    input_zemba['land_fraction'][0:15] = 1.
if input_zemba['res'] == 2.5:
    input_zemba['land_fraction'][0:6] = 1.      
if input_zemba['res'] == 5.:
    input_zemba['land_fraction'][0:3] = 1.
    
#|
#|-> Ensure north sea occupies entire area north of 80N
if input_zemba['res'] == 1.:
    input_zemba['land_fraction'][-10:] = 0.
if input_zemba['res'] == 2.5:
    input_zemba['land_fraction'][-4:] = 0.      
if input_zemba['res'] == 5.:
    input_zemba['land_fraction'][-2:] = 0. 
    
# Aqua planet (no land.)
# input_zemba['land_fraction'] = np.zeros((lat.size))

# Land fractions boundaries
#--------------------------

input_zemba["land_fraction_bounds"] = np.interp(latb, lat, input_zemba["land_fraction"])


# Land elevation (m) <------------------------------------------------------------------------------COMMENT/UNCOMMENT
#-------------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
# ice6g_lat   = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_orog  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[3], sep='\t').to_numpy()[:,0]
# input_zemba['land_height'] = np.interp(lat, ice6g_lat, ice6g_orog)

# LGM from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat   = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_orog  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[4], sep='\t').to_numpy()[:,0]
input_zemba['land_height'] = np.interp(lat, ice6g_lat, ice6g_orog)

# Flat.
# input_zemba['land_height'] = np.zeros(( lat.size ))

# Ice fractions <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#--------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
# ice6g_lat = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_if  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[5], sep='\t').to_numpy()[:,0]
# input_zemba['ice_fraction'] = np.interp(lat, ice6g_lat, ice6g_if)

# LGM from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_if  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[6], sep='\t').to_numpy()[:,0]
input_zemba['ice_fraction'] = np.interp(lat, ice6g_lat, ice6g_if)

# No ice.
# input_zemba['ice_fraction'] = np.zeros(( lat.size ))


# Cloud cover <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#------------

# NorESM2-MM PI Control Simulation (Seland et al., 2020.)
noresm2_lat   = pd.read_csv(path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[0], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]
noresm2_ccl   = pd.read_csv(path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[2], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]/100
noresm2_cco   = pd.read_csv(path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[3], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]/100
input_zemba['ccl'] = np.interp(lat, noresm2_lat, noresm2_ccl)
input_zemba['cco'] = np.interp(lat, noresm2_lat, noresm2_cco)

# Constant and ubiqitous cloud cover of 0.5 (for experimental purposes)
# input_zemba['ccl'] = np.zeros((lat.size))+0.5
# input_zemba['cco'] = np.zeros((lat.size))+0.5


# Other key model parameters
#---------------------------

# turbulent heat fluxes (m/s)
input_zemba['tbhfcl']=np.array([0.01])  # land  # <------------------------------------------ HERE
input_zemba['tbhfco']=np.array([0.006])  # ocean # <------------------------------------------ HERE

# cloud optical depth.
input_zemba['tau']=np.array([3.0]) # <-------------------------------------------------------- HERE

# atm. co2 conc.
input_zemba['co2']=np.array([284. * (44.01/28.97) * 1e-6]) # <-------------------------------- HERE

# albedo.
input_zemba['alphag']=np.array([0.15])    # bare ground  # <---------------------------------- HERE
input_zemba['alphas']=np.array([0.80])     # 'cold' snow  # <---------------------------------- HERE
input_zemba["alphaws"]=np.array([0.4])    # 'wet' snow 
input_zemba['alphasimx']=np.array([0.7]) # 'cold' sea ice  # <------------------------------- HERE
input_zemba["alphasimn"]=np.array([0.7])  # 'wet' sea ice  # <-------------------------------- HERE
input_zemba["alphai"]=np.array([0.8])     # land ice  # <------------------------------------- HERE

# maximum relative humidity (kg/kg)
input_zemba["r"]=np.array([80.]) # <---------------------------------------------------------- HERE

# GHG amplification factor.
input_zemba['GHG_amp']=np.array([1.3]) # <---------------------------------------------------- HERE

# eddy/gyre diffusion coefficient (m/s) [OCEAN]
input_zemba['do'] = np.zeros((olatb.size))
idxoc = np.where(olatb==input_zemba['occ'])[0][0]

input_zemba['do'][0:idxoc] = 5e10 / (60*60*24*365) # SH <------------------------------------- HERE
input_zemba['do'][idxoc:]  = 5e10 / (60*60*24*365) # NH <------------------------------------- HERE

# vertical diffusion coefficient (m/s)
input_zemba['dz0']=np.array([5e3 / (60*60*24*365)]) # <--------------------------------------- HERE

# interior horizontal diffusion coefficient (m/s)
input_zemba["dh"]=np.array([1.5e10 / (60*60*24*365)]) #<-------------------------------------- HERE

# hadley cell constant.
input_zemba["hadley_constant"] = np.array([1.03]) # maximum relative humidity <--------------- HERE
    
# atm. diffusion coefficient. (m/s)
input_zemba["dt"] = np.zeros((latb.size))
idxc = np.where(latb==0.)[0][0]
input_zemba['dt'][0:idxc] = 0.7e6 # SH <------------------------------------------------------- HERE
input_zemba['dt'][idxc:]  = 0.84e6 # NH <------------------------------------------------------- HERE
=======
# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

LGM (ice only) input file...

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
import pickle

# NAME OF RUN.
name = 'equili_lgm_ice'

# PATH
path = os.getcwd()   # MIGHT NEED TO CHANGE PATHS TO ACCESS *OTHER DATA* FOLDER

#--------------
# MODEL VERSION  <-------------------------------------------------------------------------------COMMENT/UNCOMMENT
#--------------

input_version = 'moist'  # moist EBM

# input_version = 'dry'    # dry 'classic' EBM

#------------
# Model prior  <-------------------------------------------------------------------------------COMMENT/UNCOMMENT
#------------

# prior = 'PI'  # initalize with pre-industrial run...

# prior = 'LGM' # initalize with last-glacial maximum run...

prior = None    # intialize with arbitrary state...

#---------------
# Model settings [e.g. for turning processes/feedbacks on or off.]
#---------------

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

# influence of height [ 0 = OFF , 1 = ON]
settings_zemba["height"]=np.array([0], dtype = "int8")        # <------------------------------- HERE

# model version [ 0 = classic 'dry', 1 = moist]
if input_version == 'moist':
    settings_zemba['version'] = np.array([1], dtype = "int8")
if input_version == 'dry':
    settings_zemba['version'] = np.array([0], dtype = "int8")
if input_version != 'moist' and input_version != 'dry':
    raise Exception("Model version must be either: 'moist' or 'dry'")

#------------
# Model input 
#------------

input_zemba = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])

# set-up
#-------

input_zemba['nyrs']=np.array([3000.]) # no. of model years...            # <-------------------- HERE
input_zemba['occ']=np.array([-5.])    # mid-point of ocean circulation...# <-------------------- HERE
input_zemba['ikyr']=np.array([0.])    # kyr BP for orbital parameters    # <-------------------- HERE

# Resolution [in degrees of latitude]
#-----------

input_zemba['res']=np.array([5.0])  # <-------------------------------------------------------- HERE

if input_zemba['res'][0] == 5.:  # if 5 degree
    lat   = np.arange(-87.5, 87.5+5., 5.)
    latb  = np.arange(-90, 90+5., 5.)
    olat  = np.arange(-67.5, 77.5+5., 5.)
    olatb = np.arange(-70, 80+5., 5.)
    
if input_zemba['res'][0] == 2.5:   # if 2.5 degree
    lat   = np.arange(-88.75, 88.75+2.5, 2.5)
    latb  = np.arange(-90, 90+2.5, 2.5)
    olat  = np.arange(-68.75, 78.75+2.5, 2.5)
    olatb = np.arange(-70, 80+2.5, 2.5)
    
if input_zemba['res'][0] == 1.:    # if 1. degree
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
# with open(path+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
#     pi = pickle.load(f)['StateYear']
# fixed_snow_fraction = pi['snow_fraction_land']
# fixed_snow_thick    = pi['snow_thick']             
# fixed_snow_melt     = pi['melt']


# [option to keep sea ice fixed to the pre-industrial.]  <----------------------------------------- COMMENT/UNCOMMENT
#------------------------------------------------------

settings_zemba['fixed_seaice'] = np.array([0], dtype = "int8") #--> OFF.
fixed_si_fraction = None
fixed_si_thick = None
fixed_si_melt = None

# settings_zemba['fixed_seaice'] = np.array([1], dtype = "int8") #--> ON.
# with open(path+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
#     pi = pickle.load(f)['StateYear']
# fixed_si_fraction = pi['si_fraction'] 
# fixed_si_thick    = pi['si_thick']       
# fixed_si_melt     = pi['si_melt_flux']    
    
# [option to keep land albedo fixed to the pre-industrial.]   <----------------------------------- COMMENT/UNCOMMENT
#----------------------------------------------------------

settings_zemba['fixed_land_albedo'] = np.array([0], dtype = "int8") #--> OFF.
fixed_land_albedo = None 

# settings_zemba['fixed_land_albedo'] = np.array([1], dtype = "int8") #--> ON.
# with open(path+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
#     pi = pickle.load(f)['StateYear']
# fixed_land_albedo = pi['alpha_land']
    

# [option to keep ocean albedo fixed to the pre-industrial.]  <----------------------------------- COMMENT/UNCOMMENT
#-----------------------------------------------------------

settings_zemba['fixed_ocean_albedo'] = np.array([0], dtype = "int8") #--> OFF.
fixed_ocean_albedo = None

# settings_zemba['fixed_ocean_albedo'] = np.array([1], dtype = "int8") #--> ON.
# with open(path+'/output/equilibrium/pi_moist_res5.0.pkl', 'rb') as f:
#     pi = pickle.load(f)['StateYear']
# fixed_ocean_albedo = pi['alpha_ocean'] 
    

# Land fractions <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#---------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_lf  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[1], sep='\t').to_numpy()[:,0]
input_zemba['land_fraction'] = np.interp(lat, ice6g_lat, ice6g_lf)

# LGM from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
# ice6g_lat = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_lf  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[2], sep='\t').to_numpy()[:,0]
# input_zemba['land_fraction'] = np.interp(lat, ice6g_lat, ice6g_lf)

#|
#|-> Ensure Antarctic continent occupies entire area south of 75S
if input_zemba['res'] == 1.:
    input_zemba['land_fraction'][0:15] = 1.
if input_zemba['res'] == 2.5:
    input_zemba['land_fraction'][0:6] = 1.      
if input_zemba['res'] == 5.:
    input_zemba['land_fraction'][0:3] = 1.
    
#|
#|-> Ensure north sea occupies entire area north of 80N
if input_zemba['res'] == 1.:
    input_zemba['land_fraction'][-10:] = 0.
if input_zemba['res'] == 2.5:
    input_zemba['land_fraction'][-4:] = 0.      
if input_zemba['res'] == 5.:
    input_zemba['land_fraction'][-2:] = 0. 
    
# Aqua planet (no land.)
# input_zemba['land_fraction'] = np.zeros((lat.size))

# Land fractions boundaries
#--------------------------

input_zemba["land_fraction_bounds"] = np.interp(latb, lat, input_zemba["land_fraction"])


# Land elevation (m) <------------------------------------------------------------------------------COMMENT/UNCOMMENT
#-------------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
# ice6g_lat   = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_orog  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[3], sep='\t').to_numpy()[:,0]
# input_zemba['land_height'] = np.interp(lat, ice6g_lat, ice6g_orog)

# LGM from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat   = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_orog  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[4], sep='\t').to_numpy()[:,0]
input_zemba['land_height'] = np.interp(lat, ice6g_lat, ice6g_orog)

# Flat.
# input_zemba['land_height'] = np.zeros(( lat.size ))

# Ice fractions <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#--------------

# Present-day from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
# ice6g_lat = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_if  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[5], sep='\t').to_numpy()[:,0]
# input_zemba['ice_fraction'] = np.interp(lat, ice6g_lat, ice6g_if)

# LGM from ICE6G_C dataset (Argus et al., 2014; Peltier et al., 2014)
ice6g_lat = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
ice6g_if  = pd.read_csv(path+'/other_data/ice6g/ice6g.txt', skiprows=7, usecols=[6], sep='\t').to_numpy()[:,0]
input_zemba['ice_fraction'] = np.interp(lat, ice6g_lat, ice6g_if)

# No ice.
# input_zemba['ice_fraction'] = np.zeros(( lat.size ))


# Cloud cover <----------------------------------------------------------------------------------COMMENT/UNCOMMENT
#------------

# NorESM2-MM PI Control Simulation (Seland et al., 2020.)
noresm2_lat   = pd.read_csv(path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[0], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]
noresm2_ccl   = pd.read_csv(path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[2], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]/100
noresm2_cco   = pd.read_csv(path+'/other_data/noresm2/noresm2_clouds.txt', skiprows=4, usecols=[3], sep='\t', encoding='ISO-8859-1').to_numpy()[:,0]/100
input_zemba['ccl'] = np.interp(lat, noresm2_lat, noresm2_ccl)
input_zemba['cco'] = np.interp(lat, noresm2_lat, noresm2_cco)

# Constant and ubiqitous cloud cover of 0.5 (for experimental purposes)
# input_zemba['ccl'] = np.zeros((lat.size))+0.5
# input_zemba['cco'] = np.zeros((lat.size))+0.5


# Other key model parameters
#---------------------------

# turbulent heat fluxes (m/s)
input_zemba['tbhfcl']=np.array([0.01])  # land  # <------------------------------------------ HERE
input_zemba['tbhfco']=np.array([0.006])  # ocean # <------------------------------------------ HERE

# cloud optical depth.
input_zemba['tau']=np.array([3.0]) # <-------------------------------------------------------- HERE

# atm. co2 conc.
input_zemba['co2']=np.array([284. * (44.01/28.97) * 1e-6]) # <-------------------------------- HERE

# albedo.
input_zemba['alphag']=np.array([0.15])    # bare ground  # <---------------------------------- HERE
input_zemba['alphas']=np.array([0.80])     # 'cold' snow  # <---------------------------------- HERE
input_zemba["alphaws"]=np.array([0.4])    # 'wet' snow 
input_zemba['alphasimx']=np.array([0.7]) # 'cold' sea ice  # <------------------------------- HERE
input_zemba["alphasimn"]=np.array([0.7])  # 'wet' sea ice  # <-------------------------------- HERE
input_zemba["alphai"]=np.array([0.8])     # land ice  # <------------------------------------- HERE

# maximum relative humidity (kg/kg)
input_zemba["r"]=np.array([80.]) # <---------------------------------------------------------- HERE

# GHG amplification factor.
input_zemba['GHG_amp']=np.array([1.3]) # <---------------------------------------------------- HERE

# eddy/gyre diffusion coefficient (m/s) [OCEAN]
input_zemba['do'] = np.zeros((olatb.size))
idxoc = np.where(olatb==input_zemba['occ'])[0][0]

input_zemba['do'][0:idxoc] = 5e10 / (60*60*24*365) # SH <------------------------------------- HERE
input_zemba['do'][idxoc:]  = 5e10 / (60*60*24*365) # NH <------------------------------------- HERE

# vertical diffusion coefficient (m/s)
input_zemba['dz0']=np.array([5e3 / (60*60*24*365)]) # <--------------------------------------- HERE

# interior horizontal diffusion coefficient (m/s)
input_zemba["dh"]=np.array([1.5e10 / (60*60*24*365)]) #<-------------------------------------- HERE

# hadley cell constant.
input_zemba["hadley_constant"] = np.array([1.03]) # maximum relative humidity <--------------- HERE
    
# atm. diffusion coefficient. (m/s)
input_zemba["dt"] = np.zeros((latb.size))
idxc = np.where(latb==0.)[0][0]
input_zemba['dt'][0:idxc] = 0.7e6 # SH <------------------------------------------------------- HERE
input_zemba['dt'][idxc:]  = 0.84e6 # NH <------------------------------------------------------- HERE
>>>>>>> 6d2a955a0f7b9234f646b2b4581e061e93cfcb09
