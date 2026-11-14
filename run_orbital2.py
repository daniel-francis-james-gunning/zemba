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
import h5py
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

input_file_name = 'input_transient_preMPT_precession'

# time frame
kyr_init = 1210
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

# ------------------------
# --- Create HDF5 file ---
# ------------------------

nkyr  = int(kyr_init - kyr_end) + 1
nlat  = Var['lat'].size
nlatb = Var['latb'].size

output_file_name = f"{output_path}/output_{name}_{version}_res{input_zemba['res'][0]}.h5"
f = h5py.File(output_file_name, 'w') # keep open for whole run

# daily fields [nkyr, nlat, 365]
Ins  = f.create_dataset('Ins',   (nkyr, nlat, 365), dtype='f8')
Ta   = f.create_dataset('Ta',  (nkyr, nlat, 365), dtype='f8')
Tl   = f.create_dataset('Tl',  (nkyr, nlat, 365), dtype='f8')
Tos  = f.create_dataset('Tos', (nkyr, nlat, 365), dtype='f8')
rsdsnet_land = f.create_dataset('rsdsnet_land',   (nkyr, nlat, 365), dtype='f8')
rsdsnet_ocean = f.create_dataset('rsdsnet_ocean',  (nkyr, nlat, 365), dtype='f8')
si_fraction = f.create_dataset('si_fraction', (nkyr, nlat, 365), dtype='f8')
sc_fraction = f.create_dataset('sc_fraction', (nkyr, nlat, 365), dtype='f8')
melt = f.create_dataset('melt', (nkyr, nlat, 365), dtype='f8')
pmelt = f.create_dataset('pmelt', (nkyr, nlat, 365), dtype='f8')
snow_thick = f.create_dataset('snow_thick', (nkyr, nlat, 365), dtype='f8')
snowfall_rate_land = f.create_dataset('snowfall_rate_land', (nkyr, nlat, 365), dtype='f8')
land_albedo = f.create_dataset('land_albedo', (nkyr, nlat, 365), dtype='f8')
ocean_albedo = f.create_dataset('ocean_albedo', (nkyr, nlat, 365), dtype='f8')
shf_ocean= f.create_dataset('shf_ocean', (nkyr, nlat, 365), dtype='f8')
lhf_evap_ocean = f.create_dataset('lhf_evap_ocean', (nkyr, nlat, 365), dtype='f8')
rlus_ocean = f.create_dataset('rlus_ocean', (nkyr, nlat, 365), dtype='f8')

# annual fields [nkyr, nlat]
Q        = f.create_dataset('Q',   (nkyr, nlat), dtype='f8')
precip   = f.create_dataset('precip',   (nkyr, nlat), dtype='f8')
snowfall = f.create_dataset('snowfall',   (nkyr, nlat), dtype='f8')
shf = f.create_dataset('shf',   (nkyr, nlat), dtype='f8')
lhf_evap = f.create_dataset('lhf_evap',   (nkyr, nlat), dtype='f8')
rlus = f.create_dataset('rlus',   (nkyr, nlat), dtype='f8')


# scalar fields [nkyr]
ecc  = f.create_dataset('ecc', (nkyr,), dtype='f8')
pre  = f.create_dataset('pre', (nkyr,), dtype='f8')
obl  = f.create_dataset('obl', (nkyr,), dtype='f8')
CO2  = f.create_dataset('CO2', (nkyr,), dtype='f8')

# edge-grid fields [nkyr, nlatb, 365]
atm_dry    = f.create_dataset('atm_dry',    (nkyr, nlatb, 365), dtype='f8')
atm_latent = f.create_dataset('atm_latent', (nkyr, nlatb, 365), dtype='f8')
ocean      = f.create_dataset('ocean',      (nkyr, nlatb, 365), dtype='f8')


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
        
    
    # insolation forcing
    # ------------------
    
    e, o, p, I, znth_sw, znth_dw = calculate_daily_insolation(orb, Var["latr"], Var["ndays"], day_type = 1)
    
    # option to fix sea ice and snow cover
    # ------------------------------------
    
    # sea ice
    if settings_zemba['seaice_flag'] == 1:
        uc = np.arange(1, Var["lat"].size + 1)
        fixed_si_fraction = np.transpose(genfromtxt(f"{input_path}/si_fraction_pi_res{res}.txt",
                                         delimiter=",", skip_header=3, usecols=uc))
        fixed_si_temp     = np.transpose(genfromtxt(f"{input_path}/si_temperature_pi_res{res}.txt",
                                         delimiter=",", skip_header=3, usecols=uc))
    else:
        fixed_si_fraction = None
        fixed_si_temp     = None
        
        
    # snow cover
    if settings_zemba['snowcover_flag'] == 1:
        fixed_sc_fraction = np.transpose(genfromtxt(f"{input_path}/sc_fraction_pi_res{res}.txt",
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
    
    # scalar fields [nkyr]
    ecc[i] = orb['ecc']              
    pre[i] = orb['pre']
    obl[i] = orb['obl']
    CO2[i] = input_zemba['co2']
    
    # annual fields [nkyr, nlat]
    Q[i, :] = StateYear['Q'].mean(axis=1)
    precip[i, :] = StateYear['precip_flux'].mean(axis=1)
    snowfall[i, :] = StateYear['snowfall_flux'].mean(axis=1)
    shf[i, :] = StateYear['shf'].mean(axis=1)
    lhf_evap[i, :] = StateYear['lhf_evap'].mean(axis=1)
    rlus[i, :] = StateYear['rlus'].mean(axis=1)
    
    # daily fields [nkyr, nlat, 365]
    Ins[i, :, :]  = I
    Ta[i, :, :]  = StateYear['Ta']
    Tl[i, :, :]  = StateYear['Tl']
    Tos[i, :, :]  = StateYear['Tos']
    rsdsnet_land[i, :, :]  = StateYear['rsdsnet_land']
    rsdsnet_ocean[i, :, :]  = StateYear['rsdsnet_ocean']
    si_fraction[i, :, :]  = StateYear['si_fraction']
    sc_fraction[i, :, :]  = StateYear['snow_fraction_land']
    melt[i, :, :]  = StateYear['melt']
    pmelt[i, :, :]  = StateYear['pmelt']
    snow_thick[i, :, :]  = StateYear['snow_thick']
    snowfall_rate_land[i, :, :]  = StateYear['snowfall_rate_land']
    land_albedo[i, :, :]  = StateYear['alpha_land']
    ocean_albedo[i, :, :]  = StateYear['alpha_ocean']
    shf_ocean[i, :, :]  = StateYear['shf_ocean']
    lhf_evap_ocean[i, :, :]  = StateYear['lhf_evap_ocean']
    rlus_ocean[i, :, :]  = StateYear['rlus_ocean']
    
    # edge-grid fields [nkyr, nlatb, 365]
    atm_dry[i, :, :]  = StateYear['dry_north']
    atm_latent[i, :, :]  = StateYear['latent_north']
    ocean[i, :, :] = (StateYear['advf'][0:nlatb,:]
                      + StateYear['advf'][5*nlatb:6*nlatb,:]
                      + StateYear['hdiffs'][0:nlatb,:])
    

    i=i+1
        

# -----------------------
# --- Save model data ---
#------------------------

meta = {
    'Var':      {k: v for k, v in Var.items()},
    'Settings': {k: v for k, v in settings_zemba.items()},
    'Input':    {k: v for k, v in input_zemba.items()},
}
meta_file_name = f"{output_path}/output_{name}_{version}_res{input_zemba['res'][0]}_meta.pkl"
with open(meta_file_name, 'wb') as g:
    pickle.dump(meta, g)

f.close()

t1 = perf_counter()
print("Mean time:", t1-t0)

import psutil
mem = psutil.Process(os.getpid()).memory_info().rss / 1e9
print(f"Peak-ish RAM (GB): {mem:.2f}")


        


    
