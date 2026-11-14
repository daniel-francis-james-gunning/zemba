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

# -------------
# --- Input ---
# -------------

input_file_name = 'input_equili_pi'

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

# --------------------------
# --- Insolation Forcing ---
# --------------------------

# load orbital parameters
orb = import_orbital_parameters_v2( input_zemba['ikyr'][0] ) 

# insolation forcing
ecc, obl, pre, I, znth_sw, znth_dw = calculate_daily_insolation(orb, Var["latr"], Var["ndays"], day_type = 1)

# --------------------------------------------
# --- Option to fix sea ice and snow cover ---
# --------------------------------------------

res   = input_zemba['res'][0]

# sea ice
if settings_zemba['seaice_flag'] == 1:
    uc = np.arange(1, Var["lat"].size + 1)
    fixed_si_fraction = np.transpose(genfromtxt(f"{output_path}/si_fraction_pi_res{res}.csv",
                                     delimiter=",", skip_header=3, usecols=uc))
    fixed_si_temp     = np.transpose(genfromtxt(f"{output_path}/si_temperature_pi_res{res}.csv",
                                     delimiter=",", skip_header=3, usecols=uc))
else:
    fixed_si_fraction = None
    fixed_si_temp     = None
    
    
# snow cover
if settings_zemba['snowcover_flag'] == 1:
    fixed_sc_fraction = np.transpose(genfromtxt(f"{output_path}/sc_fraction_pi_res{res}.csv",
                                         delimiter=",", skip_header=3, usecols=uc))
else:
    fixed_sc_fraction = None
        
    
# ---------------------
# --- Run the model ---
# ---------------------
 
State, StateYear = run_model(iniState, input_zemba, settings_zemba, Var, I, znth_dw,
                             fixed_si_fraction=fixed_si_fraction, 
                             fixed_si_temp=fixed_si_temp, 
                             fixed_sc_fraction=fixed_sc_fraction)
        

# -----------------------
# --- Save model data ---
#------------------------

# nested dict with all ouput
zemba_output = {}
zemba_output['Var']           = {}
zemba_output['Settings']      = {}
zemba_output['Input']         = {}
zemba_output['State']         = {}
zemba_output['StateYear']     = {}
zemba_output['StateAnnual']   = {}

    
for k,v in Var.items():
    zemba_output['Var'][k] = v
for k,v in settings_zemba.items():
    zemba_output['Settings'][k] = v 
for k,v in input_zemba.items():
    zemba_output['Input'][k] = v 
for k,v in State.items():
    zemba_output['State'][k] = v 
for k,v in StateYear.items():
    zemba_output['StateYear'][k] = v 

# save nested python dict
#with open('%s/output_%s_res%s.pkl' % (output_path, name, str(input_zemba['res'][0])), 'wb') as f:
        #pickle.dump(dict(zemba_output), f)
#   
    
# delete input file
#------------------
# del input_zemba
# del settings_zemba
# del name           
# del version
# del Var
# del State
# del StateYear
# del StateAnnual

    
   