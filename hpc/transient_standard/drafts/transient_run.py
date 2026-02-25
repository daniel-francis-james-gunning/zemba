# -*- coding: utf-8 -*-
"""
Script to run ZEMBA for give input file.
"""

# import packages
import pandas      as     pd
import numpy       as     np
from   numba.core  import types
from   numba.typed import Dict
from   numba       import njit
from   time        import process_time
from   numpy       import genfromtxt
import os
import sys
import pickle
import importlib

# import ZEMBA code
from zemba            import *
from initialize_zemba import *
from solar_forcing    import *
from utilities        import *

kyr_init = 2450
kyr_end  = 1200
kyr_dt   = 1

name_of_input_file = 'input_transient_preMPTgl'

# load ice fraction data
# ice_df      = pd.read_csv('ice_fraction.txt', sep='\t', encoding='utf16').to_numpy()
# start_index = np.where(ice_df[:,0] == kyr_end)[0][0]
# end_index   = np.where(ice_df[:,0] == kyr_init)[0][0]
# ice_data    = ice_df[end_index:start_index+1,:]
# ice_data    = np.flip(ice_data, axis=0)

# prescibe constant ice cover in south
# ice6g_lat        = pd.read_csv('other_data/ice6g/ice6g.txt', skiprows=7, usecols=[0], sep='\t').to_numpy()[:,0]
# ice6g_if         = pd.read_csv('other_data/ice6g/ice6g.txt', skiprows=7, usecols=[5], sep='\t').to_numpy()[:,0]
# ice6g_if         = np.interp(np.arange(-87.5, 87.5+5, 5), ice6g_lat, ice6g_if)
# ice_data[:,0:11] = ice6g_if[0:11]

# load co2 data
# co2_df      = pd.read_csv('CO2.txt',sep='\t').to_numpy()
# start_index = np.where(co2_df[:,0] == kyr_end)[0][0]
# end_index   = np.where(co2_df[:,0] == kyr_init)[0][0]
# co2_data    = co2_df[end_index:start_index+1,:]
# co2_data    = np.flip(co2_data, axis=0)

# Reference present-day orbital parameters
# ========================================

orb_pd = import_orbital_parameters_v2(0.)

# ===============
# Load input file
# ===============


# load input file ...
file           = importlib.import_module(name_of_input_file)  
name           = file.name                      # simulation name 
input_zemba    = file.input_zemba               # input dictionary           
settings_zemba = file.settings_zemba            # settings dictionary    
version        = file.input_version             # model version   

# =========
# CO2 Trend
# =========

kyr_sz   = np.arange(kyr_init, kyr_end-kyr_dt, -kyr_dt).size
co2_trend = np.linspace(300, 180, kyr_sz)

  
# ==============  
# Load constants
# ==============  
    
Var = get_constants(input_zemba)

# ====================== 
# Initialize model state
# ====================== 

State = initialize_state(Var, input_zemba)



# dictionary for saving model output
#-----------------------------------

zemba_output = {}

# insolation 
zemba_output['I']   = np.zeros((int(kyr_init-kyr_end) + 1,  Var['lat'].size, 365))

# orbital parameters
zemba_output['pre'] = np.zeros(( int(kyr_init-kyr_end) + 1))
zemba_output['ecc'] = np.zeros(( int(kyr_init-kyr_end) + 1))
zemba_output['obl'] = np.zeros(( int(kyr_init-kyr_end) + 1))

# surface temperature 
zemba_output['Ta']   = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['Tl']   = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['Tos']  = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# atmospheric humidity
zemba_output['Q']   = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))

# co2
zemba_output['CO2'] = np.zeros((int(kyr_init-kyr_end) + 1))

# radiation for atmospheric layer
zemba_output['rsatmnet']  = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))
zemba_output['rlatmnet']  = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))
zemba_output['rlut']      = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))
zemba_output['rlds']      = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))


# land ice
# zemba_output['ice_area_nh'] = np.zeros((int(kyr_init-kyr_end) + 1))
# zemba_output['ice_area_sh'] = np.zeros((int(kyr_init-kyr_end) + 1))

# sea ice
zemba_output['si_fraction'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# snow cover [land]
zemba_output['sc_fraction'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# atm. transport
zemba_output['atm_dry']    = np.zeros((int(kyr_init-kyr_end) + 1, Var['latb'].size))
zemba_output['atm_latent'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['latb'].size))

# ocn. transport
zemba_output['ocean'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['olatb'].size))

# precipitation
zemba_output['precip']    = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# snowfall
zemba_output['snowfall']    = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# surface albedo
zemba_output['land_albedo']  = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['ocean_albedo'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# sensible, latent and longwave heat fluxes at the surface
zemba_output['shf']       = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))
zemba_output['lhf_evap']  = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))
zemba_output['rlus']      = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))

# sensible, latent and longwave heat fluxes at the ocean surface
zemba_output['shf_ocean']       = np.zeros((int(kyr_init-kyr_end) + 1,  Var['lat'].size, 365))
zemba_output['lhf_evap_ocean']  = np.zeros((int(kyr_init-kyr_end) + 1,  Var['lat'].size, 365))
zemba_output['rlus_ocean']      = np.zeros((int(kyr_init-kyr_end) + 1,  Var['lat'].size, 365))


#--------------
# run the model
#--------------

key_variables = ['Tax']

input_zemba['nyrs']=np.array([1000.*kyr_dt])


i=0 # index

for kyr in np.arange(kyr_init, kyr_end-kyr_dt, -kyr_dt):  # for each time step...

    if 'CO2_trend' in settings_zemba:
        if settings_zemba['CO2_trend'][0]==1:
            input_zemba['co2']=np.array([co2_trend[i] * (44.01/28.97) * 1e-6])

    
    # retrieve orbital parameters
    # ===========================
    
    orb = import_orbital_parameters_v2(kyr)
    
    
    # Option to fix orbital parameters
    # ================================
    
    # option to fix obliquity to PD
    if settings_zemba['fixed_obliquity_flag'] == 1:
        orb['obl'] = orb_pd['obl']
        
    # option to fix climatic precession to PD
    if settings_zemba['fixed_precession_flag'] == 1:
        orb['pre'] = orb_pd['pre']
        orb['ecc'] = orb_pd['ecc']
    
    # store orbital parameters
    # ========================
    
    zemba_output['pre'][i] = orb['pre']
    zemba_output['ecc'][i] = orb['ecc']
    zemba_output['obl'][i] = orb['obl']
    
    # insolation forcing
    # ==================
    
    ecc, obl, pre, I, znth_sw, znth_dw = calculate_daily_insolation(orb, Var["latr"], Var["ndays"], day_type = 1)
    
    # Option to fix land and ocean albedo
    # ===================================
    
    # fixed ocean albedo
    if settings_zemba['oceanalbedo_flag'] == 1:
        fixed_ocean_albedo = np.transpose(genfromtxt("pi_ocean_albedo.txt", delimiter=",", skip_header=3, usecols=np.arange(1,37)))
    else: 
        fixed_ocean_albedo = None
        
    
    # fixed land albedo
    if settings_zemba['landalbedo_flag'] == 1:
        fixed_land_albedo = np.transpose(genfromtxt("pi_land_albedo.txt", delimiter=",", skip_header=3, usecols=np.arange(1,37)))
    else: 
        fixed_land_albedo = None
        
    # Option to fix sea ice and snow cover
    # ====================================
    
    # fixed sea ice
    if settings_zemba['seaice_flag'] == 1:
        fixed_si_fraction = np.transpose(genfromtxt("si_fraction.txt", delimiter=",", skip_header=3, usecols=np.arange(1,37)))
        fixed_si_temp     = np.transpose(genfromtxt("si_temperature.txt", delimiter=",", skip_header=3, usecols=np.arange(1,37)))
    else: 
        fixed_si_fraction = None
        fixed_si_temp     = None
        
    
    # fixed snow cover
    if settings_zemba['snowcover_flag'] == 1:
        fixed_sc_fraction = np.transpose(genfromtxt("sc_fraction.txt", delimiter=",", skip_header=3, usecols=np.arange(1,37)))
    else: 
        fixed_sc_fraction = None
        
        
        
    # Run the model
    # =============

    State, StateYear, StateAnnual = ebm(Var, State, input_zemba, I, znth_dw, settings_zemba, key_variables, fixed_land_albedo=fixed_land_albedo, fixed_ocean_albedo=fixed_ocean_albedo, fixed_si_fraction=fixed_si_fraction, fixed_si_temp=fixed_si_temp, fixed_sc_fraction=fixed_sc_fraction)
    
    # Save model output 
    # =================
    
    # temperature
    zemba_output['Ta'][i,:,:]   = StateYear['Ta']
    zemba_output['Tl'][i,:,:]   = StateYear['Tl']
    zemba_output['Tos'][i,:,:]  = StateYear['Tos']
    
    # insolation
    zemba_output['I'][i,:,:]    = I
    
    # co2
    zemba_output['CO2'][i]      = input_zemba['co2']
    
    # sea ice
    zemba_output['si_fraction'][i,:,:] = StateYear['si_fraction']
    
    # atmospheric humidity
    zemba_output['Q'][i,:] = StateYear['Q'].mean(axis=1)
    
    # radiation for atmospheric layer
    zemba_output['rsatmnet'][i,:]  = StateYear['rsatmnet'].mean(axis=1)
    zemba_output['rlatmnet'][i,:]  = StateYear['rlatmnet'].mean(axis=1)
    zemba_output['rlut'][i,:]      = StateYear['rlut'].mean(axis=1)
    zemba_output['rlds'][i,:]      = StateYear['rlds'].mean(axis=1)

    # land ice
    # zemba_output['ice_area_nh'][i] = StateYear['ice_area_nh'][0,:].mean()
    # zemba_output['ice_area_sh'][i] = StateYear['ice_area_sh'][0,:].mean()
    
    # snow
    zemba_output['sc_fraction'][i,:,:] = StateYear['snow_fraction_land']
    
    # atm heat transport
    zemba_output['atm_dry'][i,:]    = StateYear['dry_north'].mean(axis=1)
    zemba_output['atm_latent'][i,:] = StateYear['latent_north'].mean(axis=1)
    
    # ocn. transport
    zemba_output['ocean'][i,:] = StateYear['advf'][0:31,:].mean(axis=1)+StateYear['advf'][155:186,:].mean(axis=1)+StateYear['hdiffs'][0:31,:].mean(axis=1)
    
    # precipitation
    zemba_output['precip'][i,:,:] = StateYear['precip_flux']
    
    # snowfall
    zemba_output['snowfall'][i,:,:] = StateYear['snowfall_flux']
    
    # albedo
    zemba_output['land_albedo'][i,:,:]  = StateYear['alpha_land']
    zemba_output['ocean_albedo'][i,:,:] = StateYear['alpha_ocean']
    
    # sensible, latent and longwave heat fluxes at the surface
    zemba_output['shf'][i,:]      = StateYear['shf'].mean(axis=1)
    zemba_output['lhf_evap'][i,:] = StateYear['lhf_evap'].mean(axis=1)
    zemba_output['rlus'][i,:]     = StateYear['rlus'].mean(axis=1)
    
    # sensible, latent and longwave heat fluxes at the ocean surface
    zemba_output['shf_ocean'][i,:,:]      = StateYear['shf_ocean']
    zemba_output['lhf_evap_ocean'][i,:,:] = StateYear['lhf_evap_ocean']
    zemba_output['rlus_ocean'][i,:,:]     = StateYear['rlus_ocean']
    
    i=i+1
    
# Save other model data 
# =====================
 
# nested dict with all ouput
zemba_output['Var']          = {}
zemba_output['Settings']     = {}
zemba_output['Input']        = {}

for k,v in Var.items():
    zemba_output['Var'][k] = v
for k,v in settings_zemba.items():
    zemba_output['Settings'][k] = v 
for k,v in input_zemba.items():
    zemba_output['Input'][k] = v 

# save nested python dict
output_file_name = f"output_{name}_{version}_res{input_zemba['res'][0]}.pkl"
with open(output_file_name, 'wb') as f:
    pickle.dump(zemba_output, f)
