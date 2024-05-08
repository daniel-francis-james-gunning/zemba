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
input_path  = script_path + '/input/'
output_path = script_path + '/output/'


#--------------------
# List of experiments [input files]
#--------------------

input_list = [  
    
                # pre-industrial standard ...
                #------------------------
                
                'input_equili_pi',
                
                # lgm-standard ...
                #-------------
                
                # 'input_equili_lgm',
                
                # 'input_equili_lgm_ice',
                
                # 'input_equili_lgm_ice_co2',
                
                # 'input_equili_lgm_ice_co2_inso',
                
                # 'input_equili_lgm_ice_co2_inso_oc15s',
                
                # 'input_equili_lgm_75%ov',
                
                # 'input_equili_lgm_50%ov',
                
                
                # other sensitivity experiments
                #------------------------------
                
                # 'input_equili_2xCO2',

                # 'input_equili_2%inso',
                
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
    
    # load data from input file
    #--------------------------
    
    # run name
    name           = file.name 

    # input dictionary                
    input_zemba    = file.input_zemba   

    # settings dictionary           
    settings_zemba = file.settings_zemba   

    # version        
    version        = file.input_version   
             
    # fixed land albedo
    fixed_land_albedo  = file.fixed_land_albedo 

    # fixed ocean albedo   
    fixed_ocean_albedo = file.fixed_ocean_albedo       
                                    
    # fixed land snow
    fixed_snow_fraction = file.fixed_snow_fraction     
    fixed_snow_thick    = file.fixed_snow_thick
    fixed_snow_melt     = file.fixed_snow_melt
    
    # fix sea ice                                        
    fixed_si_fraction = file.fixed_si_fraction         
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
    
    # Change ocean overturning strength
    #----------------------------------

    State["ww"] = State["ww"] * input_zemba['strength_of_overturning']
    
    # Key variables to save for every year
    #-------------------------------------
    
    # key_variables = ['Tlgx', 'Tosg', 'Tsgx', 'Tagx']
    key_variables = ['Tax']
    
    # Run the model
    #--------------

    t1_start = process_time() # start timer
    
    State, StateYear, StateAnnual = ebm(Var, State, input_zemba, I, znth_dw, settings_zemba, key_variables,
                                        
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
    with open('%soutput_%s_res%s.pkl' % (output_path, name, str(input_zemba['res'][0])), 'wb') as f:
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

ndir = script_path+"/output/plots/"

os.chdir(ndir)
exec(open("plot_f01.py").read())

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
