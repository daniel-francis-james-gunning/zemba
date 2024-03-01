# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

Running equillibrium simulation of ZEMBA.

Experiments included:
    
    Pre-Industrial
    
    LGM (only ice)
    LGM (only ice + co2)
    LGM (ice + co2 + inso)
    LGM + shift in ocean circulation centre to 15S
    LGM + shift in ocean circulation centre to 15S + 25% decrease overturning
    LGM + shift in ocean circulation centre to 15S + 50% decrease overturning
    
    2x CO2
    +2% Insolation
"""

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit
import os
import sys
import pickle
import importlib
from time import process_time

from initialize_zemba import *
from zemba import *
from solar_forcing import *
from utilities import *

script_path = os.getcwd()
input_path  = script_path + '/input'
output_path = script_path + '/output'


#--------------------
# List of experiments [input files]
#--------------------

input_list = [ #'input_pi', 
                'input_lgm_icef',
                'input_lgm_icefh',
                'input_lgm_icefh_co2',
                'input_lgm_icefh_co2_inso',
                'input_lgm_icefh_co2_inso_oc10s',
                'input_lgm_75%ov',
                'input_lgm_50%ov',
                # 'input_2xCO2',
                # 'input_2%Inso',
              ]


#----------------------------------------------
# Run the model iteratively through experiments
#----------------------------------------------

for exp in input_list:
    
    # load input file
    #----------------
    
    os.chdir(input_path)
    file = importlib.import_module(exp)
    os.chdir(script_path)
    name           = file.name                         # run name
    input_zemba    = file.input_zemba                  # input dictionary
    settings_zemba = file.settings_zemba               # settings dictionary
    version        = file.input_version                # version

    
    fixed_land_albedo  = file.fixed_land_albedo        # fixed land albedo
    fixed_ocean_albedo = file.fixed_ocean_albedo       # fixed ocean albedo
                                    
    fixed_snow_fraction = file.fixed_snow_fraction     # fixed land snow
    fixed_snow_thick    = file.fixed_snow_thick
    fixed_snow_melt     = file.fixed_snow_melt
                                            
    fixed_si_fraction = file.fixed_si_fraction         # fix sea ice
    fixed_si_thick    = file.fixed_si_thick
    fixed_si_melt     = file.fixed_si_melt
    
    # load constants
    #---------------
    
    Var = get_constants(input_zemba)
    
    # initialize model state
    #-----------------------
    
    State = initialize_state(Var, input_zemba)
    
    # insolation forcing
    #-------------------
    
    # load orbital parameters
    orb = import_orbital_parameters_v2( input_zemba['ikyr'][0] ) 

    # insolation forcing
    ecc, obl, pre, I, znth_sw, znth_dw = calculate_daily_insolation(orb, Var["latr"], Var["ndays"], day_type = 1)
    I = I * input_zemba['strength_of_insolation'] # modify strength (experimental)
    
    #----------------------------------
    # Change ocean overturning strength
    #----------------------------------

    State["ww"] = State["ww"] * input_zemba['strength_of_overturning']
    
    # Run the model
    #--------------

    t1_start = process_time() # start timer
    
    State, StateYear, StateAnnual = ebm(Var, State, input_zemba, I, znth_dw, settings_zemba,
                                        
                                        # [option to fix albedo]
                                        fixed_land_albedo=fixed_land_albedo,
                                        fixed_ocean_albedo=fixed_ocean_albedo,
                                        
                                        # [option to fix snow over land]
                                        fixed_snow_fraction = fixed_snow_fraction,
                                        fixed_snow_thick    = fixed_snow_thick,
                                        fixed_snow_melt     = fixed_snow_melt,
                                        
                                        # [option to fix sea ice]
                                        fixed_si_fraction = fixed_si_fraction,
                                        fixed_si_thick    = fixed_si_thick,
                                        fixed_si_melt     = fixed_si_melt)
    
    t1_stop = process_time()  # end timer
    print("Elapsed time:", t1_stop - t1_start)
    
    # Save model data 
    #----------------
    
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
    for k,v in StateAnnual.items():
        zemba_output['StateAnnual'][k] = v 
    
    # save nested python dict
    with open(script_path+'/' + name + '_'+ version +'_res'+str(input_zemba['res'][0])+'.pkl', 'wb') as f:
            pickle.dump(dict(zemba_output), f)
   
    
    # delete input file
    #------------------
    del input_zemba
    del settings_zemba
    del name           
    del version
    del fixed_land_albedo 
    del fixed_ocean_albedo                                
    del fixed_snow_fraction 
    del fixed_snow_thick    
    del fixed_snow_melt                                           
    del fixed_si_fraction 
    del fixed_si_thick    
    del fixed_si_melt 
    del Var
    del State
    del StateYear
    del StateAnnual
    
    
#----------------------
# Call plotting scripts
#----------------------

ndir = output_path+"/plots/"

# os.chdir(ndir)
# exec(open("plot_f01.py").read())

os.chdir(ndir)
exec(open("plot_f02.py").read())

os.chdir(ndir)  
exec(open("plot_f03.py").read())

os.chdir(ndir)  
exec(open("plot_f04.py").read())

os.chdir(ndir)  
exec(open("plot_f05.py").read())

os.chdir(ndir)  
exec(open("plot_f06.py").read())

os.chdir(ndir)  
exec(open("plot_f07.py").read())

os.chdir(ndir)  
exec(open("plot_f08.py").read())

os.chdir(ndir)  
exec(open("plot_f9&10.py").read())    
