# -*- coding: utf-8 -*-

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
from   pathlib      import Path

script_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(script_path))
input_path  = script_path / 'input'
output_path = script_path / 'output'


# internal modules
from zemba import *
from solar_forcing import *
from initialize_zemba import initialize_state as iz
from initialize_zemba import get_constants as gc

# ================================
# Orbital parameter configurations
# ================================

# pre-industrial
orb_pi  = {'pre': np.deg2rad(100.33+180), 'obl': np.deg2rad(23.459), 'ecc': 0.016764}

# last interglacial
orb_lig = {'pre': np.deg2rad(275.41+180), 'obl': np.deg2rad(24.040),'ecc': 0.039378}

# =================
# CO2 concentration
# =================

co2_pi  = np.array([284.2 * (44.01/28.97) * 1e-6])

co2_lig = np.array([275. * (44.01/28.97) * 1e-6])


# Load pre-industrial input settings
# ==================================

os.chdir(input_path)

# load input file ...
file           = importlib.import_module('input_equili_pi')  
name           = file.name              # simulation name 
input_zemba    = file.input_zemba       # input dictionary           
settings_zemba = file.settings_zemba    # settings dictionary    
version        = file.input_version     # model version   

os.chdir(script_path)

# load constants
Var = gc(input_zemba)  
    
# initialize model state
iniState = iz(Var, input_zemba) 


# load the pre-industrial orbital forcing
ecc, obl, pre, piI, znth_sw, piznth_dw = calculate_daily_insolation(orb_pi, Var["latr"], Var["ndays"], day_type = 1)

# load the lpre-industrial CO2 level
input_zemba['co2'] = co2_pi


# run the the pre-industrial simulation
# =====================================

piState, piStateYear = run_model(iniState, input_zemba, settings_zemba, Var, piI, piznth_dw)

# run the last interglacial simulation
# =====================================

# load the last interglacial orbital forcing
ecc, obl, pre, ligI, ligznth_sw, ligznth_dw = calculate_daily_insolation(orb_lig, Var["latr"], Var["ndays"], day_type = 1)

# load the last interglacial CO2 level
input_zemba['co2'] = co2_lig

ligState, ligStateYear = run_model(iniState, input_zemba, settings_zemba, Var, ligI, ligznth_dw)


# save
# ====

zemba_output = { # nested dict with all output
                'Var'         : dict(Var),
                'Settings'    : dict(settings_zemba),
                'Input'       : dict(input_zemba),
                'piStateYear' : dict(piStateYear),
                'ligStateYear': dict(ligStateYear),
                'ligI'        : {"ligI": ligI},
                'piI'         : {"ligI": piI},
            
               }

# save nested python dict
output_file_name = f"output/output_orb_lig_{version}_res{input_zemba['res'][0]}.pkl"
with open(output_file_name, 'wb') as f:
    pickle.dump(zemba_output, f)



