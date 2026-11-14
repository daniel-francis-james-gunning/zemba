# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

Template for running an equillibrium simulation of ZEMBA.
"""

# modules
import os
import sys
import pickle
import importlib
import numpy as np
from   numba.core  import types
from   numba.typed import Dict
from   numba       import njit
from   time        import perf_counter
from   numpy       import genfromtxt

# internal modules
from zemba import *
from solar_forcing import *
from initialize_zemba import initialize_state as iz
from initialize_zemba import get_constants as gc

# paths
script_path = os.getcwd()
input_path  = script_path + '/input/'
output_path = script_path + '/output/'

# -------------------------------------------------
# --- Present-day orbital parameters (reference ---
# -------------------------------------------------

orb_pd = import_orbital_parameters_v2(0.)

# -------------
# --- Input ---
# -------------

input_file_name = 'input_transient_preMPTgl'

# time frame
kyr_init = 1250
kyr_end  = 1200
kyr_dt   = 1

# load
os.chdir(input_path)
file = importlib.import_module(f"{input_file_name}") 
os.chdir(script_path)
name           = file.name                      # simulation name 
input_zemba    = file.input_zemba               # input dictionary           
settings_zemba = file.settings_zemba            # settings dictionary    
version        = file.input_version             # model version   

# ----------------------
# --- Initialization ---
# ----------------------

# load constants
Var = gc(input_zemba)  
    
# initialize model state
iniState = iz(Var, input_zemba) 

# -----------------------------
# --- Dict for model output ---
# -----------------------------

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
zemba_output['rlatmnet']  = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))
zemba_output['rsdsnet_land']  = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['rsdsnet_ocean']  = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['rlut']      = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))
zemba_output['rlds']      = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size))

# sea ice
zemba_output['si_fraction'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# snow cover [land]
zemba_output['sc_fraction'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['melt'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['pmelt'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['snow_thick'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# atm. transport
zemba_output['atm_dry']    = np.zeros((int(kyr_init-kyr_end) + 1, Var['latb'].size, 365))
zemba_output['atm_latent'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['latb'].size, 365))

# ocn. transport
zemba_output['ocean'] = np.zeros((int(kyr_init-kyr_end) + 1, Var['latb'].size, 365))

# precipitation
zemba_output['precip']    = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

# snowfall
zemba_output['snowfall']    = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))
zemba_output['snowfall_rate_land']    = np.zeros((int(kyr_init-kyr_end) + 1, Var['lat'].size, 365))

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
        
    
# ---------------------
# --- Run the model ---
# ---------------------

State = iniState.copy()
res   = input_zemba['res'][0]
input_zemba['nyrs']=np.array([1000.*kyr_dt])
i=0 # index

t0 = perf_counter()
for kyr in np.arange(kyr_init, kyr_end-kyr_dt, -kyr_dt): # for each time step...


    # retrieve orbital parameters
    # ---------------------------
    
    orb = import_orbital_parameters_v2(kyr)
     
    # Option to fix orbital parameters
    # --------------------------------
    
    # option to fix obliquity to PD
    if settings_zemba['fixed_obliquity_flag'] == 1:
        orb['obl'] = orb_pd['obl']
        
    # option to fix climatic precession to PD
    if settings_zemba['fixed_precession_flag'] == 1:
        orb['pre'] = orb_pd['pre']
        orb['ecc'] = orb_pd['ecc']
        
    # store orbital parameters
    # ------------------------
    
    zemba_output['pre'][i] = orb['pre']
    zemba_output['ecc'][i] = orb['ecc']
    zemba_output['obl'][i] = orb['obl']
    
    # insolation forcing
    # ------------------
    
    ecc, obl, pre, I, znth_sw, znth_dw = calculate_daily_insolation(orb, Var["latr"], Var["ndays"], day_type = 1)
    
    # option to fix sea ice and snow cover
    # ------------------------------------
    
    # sea ice
    if settings_zemba['seaice_flag'] == 1:
        uc = np.arange(1, Var["lat"].size + 1)
        fixed_si_fraction = np.transpose(genfromtxt(f"{output_path}/si_fraction_pi_res{res}.txt",
                                         delimiter=",", skip_header=3, usecols=uc))
        fixed_si_temp     = np.transpose(genfromtxt(f"{output_path}/si_temperature_pi_res{res}.txt",
                                         delimiter=",", skip_header=3, usecols=uc))
    else:
        fixed_si_fraction = None
        fixed_si_temp     = None
        
        
    # snow cover
    if settings_zemba['snowcover_flag'] == 1:
        fixed_sc_fraction = np.transpose(genfromtxt(f"{output_path}/sc_fraction_pi_res{res}.txt",
                                             delimiter=",", skip_header=3, usecols=uc))
    else:
        fixed_sc_fraction = None
        
    
    # call ZEMBA
    # -----------

    State, StateYear = run_model(State, input_zemba, settings_zemba, Var, I, znth_dw,
                                 fixed_si_fraction=fixed_si_fraction, 
                                 fixed_si_temp=fixed_si_temp, 
                                 fixed_sc_fraction=fixed_sc_fraction)
    
    # save model output 
    # ----------------- 
    
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
    zemba_output['rsdsnet_land'][i,:,:]  = StateYear['rsdsnet_land']
    zemba_output['rsdsnet_ocean'][i,:,:] = StateYear['rsdsnet_ocean']
    

    zemba_output['sc_fraction'][i,:,:] = StateYear['snow_fraction_land']
    zemba_output['melt'][i,:,:] = StateYear['melt']
    zemba_output['pmelt'][i,:,:] = StateYear['pmelt']
    zemba_output['snow_thick'][i,:,:] = StateYear['snow_thick']
    
    # atm heat transport
    zemba_output['atm_dry'][i,:,:]    = StateYear['dry_north']
    zemba_output['atm_latent'][i,:,:] = StateYear['latent_north']
    
    # ocn. transport
    nlatb = Var['latb'].size
    zemba_output['ocean'][i,:,:] = (StateYear['advf'][0:nlatb,:]
                                    + StateYear['advf'][5*nlatb:6*nlatb,:]
                                    + StateYear['hdiffs'][0:nlatb,:])
    
    # precipitation
    zemba_output['precip'][i,:,:] = StateYear['precip_flux']
    
    # snowfall
    zemba_output['snowfall'][i,:,:] = StateYear['snowfall_flux']
    zemba_output['snowfall_rate_land'][i,:,:] = StateYear['snowfall_rate_land']
    
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
        

# -----------------------
# --- Save model data ---
#------------------------

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
    
t1 = perf_counter()
print("Mean time:", t1-t0)
        


    
