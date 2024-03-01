# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 16:48:01 2023

@author: dgu041
"""

#-----------------------
# Import modules/scripts
#-----------------------

# modules
#--------
import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit
import os
import sys
import pickle
import proplot as pplt

#--------------------------------
# Mean ocean temperature function
#--------------------------------

@njit(nogil = True)
def mean_ocean_temperature(to, Var):
    
    """
    Returns mean ocean temperature.
    """
    
    # flattened ocean depth
    depth = np.concatenate((np.repeat(Var["odepth"][0], Var["lat"].size),
                            np.repeat(Var["odepth"][1], Var["lat"].size),
                            np.repeat(Var["odepth"][2], Var["lat"].size),
                            np.repeat(Var["odepth"][3], Var["lat"].size),
                            np.repeat(Var["odepth"][4], Var["lat"].size),
                            np.repeat(Var["odepth"][5], Var["lat"].size)))
    
    
    # flattened ocean area (comprising six layers)
    area = np.concatenate((Var["oarea"], Var["oarea"], Var["oarea"],
                          Var["oarea"], Var["oarea"], Var["oarea"]))
    

    # flattened ocean volume (comprising six layers)
    volume = area*depth

    # global mean ocean temperature 
    to_avg = np.nansum(to*volume)/volume.sum() 
    
    return to_avg


# import scripts
#---------------

# current directory
cdir = os.getcwd()

# sensitivity experiment directory (parent directory)
pdir = os.path.dirname(os.getcwd())

# change to script directory 
os.chdir(pdir)

# import scripts from script directory
from initialize_test2 import *
from ebm_test2 import *
from solar_forcing import *
from utilities import *
from plotting_functions import *
from ebm_input import *
from ebm_settings import *

# annual mean insolation forcing
#-------------------------------

inso = np.array([174.27954748, 177.13146278, 182.97989641, 192.18982171,
       205.69439134, 226.13557337, 249.74356041, 273.78407601,
       297.19535424, 319.35343609, 339.83164204, 358.3126514 ,
       374.54947531, 388.34616621, 399.54761729, 408.03393363,
       413.7172542 , 416.53995826, 416.47369954, 413.51898232,
       407.70515757, 399.09083923, 387.7648625 , 373.84807002,
       357.49648265, 338.90692137, 318.32720118, 296.07541536,
       272.57895658, 248.46243211, 224.78818635, 204.29100003,
       190.74110677, 181.49688344, 175.6254384 , 172.76197344])

znth = np.array([0.12510296, 0.13295567, 0.14903198, 0.17440044, 0.21287371,
       0.2684849 , 0.31778083, 0.36347357, 0.40582195, 0.44472868,
       0.48000753, 0.5114518 , 0.53886007, 0.56204823, 0.58085616,
       0.59515173, 0.60483338, 0.60983181, 0.61011104, 0.60566886,
       0.59653685, 0.58277994, 0.56449534, 0.54181088, 0.51488238,
       0.48388961, 0.44902946, 0.4105035 , 0.36849135, 0.32307955,
       0.27398315, 0.2183102 , 0.17972609, 0.15435548, 0.13830767,
       0.13047991])

# load constants
#---------------

# get constants
Var = get_constants(topography = "PI", resolution = 5., nyrs= 6000.) # set to PI topography

# initialize pre-industrial climate state
State = initialize_state(Var, INPUT, -5)

# number of years
nyrs = 3000

# initialize arrays
#------------------

# ocean temperature
To_last_year = np.zeros((State["To"].size, 365)) # ocean temperature
To_all_years = np.zeros((State["To"].size, nyrs, 365)) # ocean temperature

# outgoing longwave radiation
olw_last_year = np.zeros((Var["lat"].size, 365))

# outgoing longwave radiation
shf_last_year = np.zeros((Var["lat"].size, 365))

# outgoing longwave radiation
lhf_last_year = np.zeros((Var["lat"].size, 365))

# ocean convergence
oc_conv_last_year = np.zeros((180, 365))

# ocean velocity
#---------------

State["u1"], State["u2"], State["ww"] = horizontal_ocean_velocities(Var["olat"], Var["olatb"], Var["olatr"], Var["olatbr"], Var["dlatr"], State["idxcos"], np.array([80.]), np.array([-70.]), np.zeros((Var["olat"].size))+0.7, np.zeros((Var["olatb"].size))+0.7, Var["r_earth"], State["ww"], Var["odepth"])

# Surface area of ocean at grid centres
oa = (
            # east-west width of latitude band
            (2*np.pi*Var["r_earth"]*0.7*np.cos(Var["olatr"]))
            * 
            # north-south length of latitude band 
            (Var["r_earth"]*Var["dlatr"])
            )

# East-west width of ocean at grid boundaries
ow = (2*np.pi*Var["r_earth"]*0.7*np.cos(Var["olatbr"]))


# boundary for ocean south
sbi = np.where(Var["latb"] == -70.)[0][0]

# boundary for ocean north
nbi = np.where(Var["latb"] == 80.)[0][0]

co = np.where(Var["olatb"] == State["idxcos"])[0][0]

# empty layer
#------------



# some constants
#---------------

# divide To (ocean temperature) into layers

To1 = State["To"][Var["idxolyr1"].astype("i8")].copy()
To1 = To1[sbi:nbi]

rhospc = Var["ocean_rho"]*Var["ocean_sphc"]


sz = Var["olatb"].size 



# run model
#----------

for year in np.arange(0, 3000): # for every year

    
    for day in np.arange(0, 365): # for every day
    
    
        #-----------------------------
        # Calculate the surface albedo
        #-----------------------------
        
        # Over land
        #----------
        
        State["alpha_land"] = surface_albedo_land_v2(Var["lat"], Var["land_height"], Var["lr"], INPUT["alphag"], State["Tal"], State["Tl"], INPUT["alphas"], INPUT["alphaws"], INPUT["alphai"], znth, Var["land_mask"], Var["ice_fraction"])
            
            
        # Over ocean
        #-----------
        
        State["alpha_ocean"] = surface_albedo_ocean(Var["lat"], State["si_fraction"], State["si_melt_flux"], INPUT["alphasimn"], INPUT["alphasimx"], znth, Var["ocean_mask"])
            
        #-----------------------------------------------------
        # Calculate the shortwave radiative fluxes (in W m^-2)
        #-----------------------------------------------------

        # Shortwave radiative fluxes
        State["rsut_land"], State["rsds_land"], State["rsus_land"]    =  SWR_param(Var["ccl"], State["Tal"], State["alpha_land"], State["alpha_land"], znth, Var["land_height"], inso, INPUT["tau"])
        State["rsut_ocean"], State["rsds_ocean"], State["rsus_ocean"] =  SWR_param(Var["cco"], State["Tao"], State["alpha_ocean"], State["alpha_ocean"], znth, Var["ocean_height"], inso, INPUT["tau"])

        # Net downwards shortwave radiation at TOA
        State["rsdtnet_land"]  = inso - State["rsut_land"]
        State["rsdtnet_ocean"] = inso - State["rsut_ocean"] 
        State["rsdtnet"]       = weighted_average(Var, State["rsdtnet_land"], State["rsdtnet_ocean"])
        
        # Net downwards shortwave radiation at surface
        State["rsdsnet_land"]  = State["rsds_land"] - State["rsus_land"]
        State["rsdsnet_ocean"] = State["rsds_ocean"] - State["rsus_ocean"]

        # Shortwave radiation absorbed by atmosphere
        State["rsatmnet_land"]  = State["rsdtnet_land"]  - State["rsdsnet_land"]
        State["rsatmnet_ocean"] = State["rsdtnet_ocean"] - State["rsdsnet_ocean"]
        
        
        # save weighted averages for land and ocean components
        State["rsds"]      = weighted_average(Var, State["rsds_land"], State["rsds_ocean"])
        State["rsus"]      = weighted_average(Var, State["rsus_land"], State["rsus_ocean"])
        State["rsut"]      = weighted_average(Var, State["rsut_land"], State["rsut_ocean"])
        State["rsdsnet"]   = weighted_average(Var, State["rsdsnet_land"], State["rsdsnet_ocean"])
        State["rsatmnet"]  = weighted_average(Var, State["rsatmnet_land"], State["rsatmnet_ocean"])

        #----------------------------------------------------
        # Calculate the longwave radiative fluxes (in W m^-2)
        #----------------------------------------------------

        # Longwave radiation fluxes
        State["rlus_land"], State["rlds_land"], State["rlut_land"]    = LWR_param(Var["sigma"], INPUT["co2"], State["Tal"], State["Tl"], State["Talgx"], Var["land_height"], Var["ccl"], Var["cloud_emissivity"])
        State["rlus_ocean"], State["rlds_ocean"], State["rlut_ocean"] = LWR_param(Var["sigma"], INPUT["co2"], State["Tao"], State["Tos"], State["Taogx"], Var["ocean_height"], Var["cco"], Var["cloud_emissivity"])
        
        # Weighted average of upwards longwave radiation at TOA
        State["rlut"] = weighted_average(Var, State["rlut_land"], State["rlut_ocean"])
        
        # Net downwards longwave radiation at surface
        State["rldsnet_land"]  = State["rlds_land"]  - State["rlus_land"]
        State["rldsnet_ocean"] = State["rlds_ocean"] - State["rlus_ocean"]
        
        # Surface absorbed longwave radiation
        State["rldsnet_land"]  = State["rlds_land"]  - State["rlus_land"]
        State["rldsnet_ocean"] = State["rlds_ocean"] - State["rlus_ocean"]

        # Longwave radiation absorbed by atmosphere
        State["rlatmnet_land"]  = State["rlus_land"]  - (State["rlds_land"]  + State["rlut_land"])
        State["rlatmnet_ocean"] = State["rlus_ocean"] - (State["rlds_ocean"] + State["rlut_ocean"])
        
        # save weighted averages for land and ocean components
        State["rlus"]      = weighted_average(Var, State["rlus_land"], State["rlus_ocean"])
        State["rlds"]      = weighted_average(Var, State["rlds_land"], State["rlds_ocean"])
        State["rldsnet"]   = weighted_average(Var, State["rldsnet_land"], State["rldsnet_ocean"])
        State["rlatmnet"]  = weighted_average(Var, State["rlatmnet_land"], State["rlatmnet_ocean"])
        
        #-------------------------------------------------
        # Calculate the total radiative fluxes (in W m^-2)
        #-------------------------------------------------
        
        # Net downwards radiation at TOA
        State["rtnet_land"]  = State["rsdtnet_land"] - State["rlut_land"]
        State["rtnet_ocean"] = State["rsdtnet_ocean"] - State["rlut_ocean"]
        State["rtnet"]       = State["rsdtnet"] - State["rlut"]
        
        # Net downwards radiation at surface
        State["rsnet_land"]  = State["rsdsnet_land"] + State["rldsnet_land"]
        State["rsnet_ocean"] = State["rsdsnet_ocean"] + State["rldsnet_ocean"]
        
        # Radiation absorbed by atmosphere
        State["ratmnet_land"]  = State["rsatmnet_land"] + State["rlatmnet_land"]
        State["ratmnet_ocean"] = State["rsatmnet_ocean"] + State["rlatmnet_ocean"]
        
        # save weighted averages for land and ocean components
        State["rsnet"]       = State["rsdsnet"] + State["rldsnet"]
        State["ratmnet"]     = State["rsatmnet"] + State["rlatmnet"]
        

        #-----------------------------------------------------------
        # Calculate the vertical sensible heat exchanges (in W m^-2)
        #-----------------------------------------------------------

        # Over land
        #----------
        
        State["shf_land"] = sensible_heat_flux(State["Tl"], State["Tal"], Var["atm_sphc"], Var["atm_rho"], INPUT["tbhfcl"], Var["land_mask"])
        
        # Over ocean
        #-----------
        
        State["shf_ocean"] = sensible_heat_flux(State["Tos"], State["Tao"], Var["atm_sphc"], Var["atm_rho"], INPUT["tbhfco"], Var["ocean_mask"])
        
        
        # Save weighted average
        #----------------------
        
        State["shf"] = weighted_average(Var, State["shf_land"], State["shf_ocean"])
        
        
        #-----------------------------------------------------
        # Calculate the evaporation flux (in kg m^-2 s^-1) and
        # associated latent heat flux (in W m^-2)
        #-----------------------------------------------------

        # Over land
        #----------
        
        State["evap_flux_land"], State["lhf_evap_land"] = surface_evaporation(State["Tl"], State["Tal"], State["Q_land"], Var["land_height"], Var["Lv"], INPUT["r"], Var["atm_rho"], INPUT["tbhfcl"], Var["W_land"], hydro=0, height=0)

        # Over ocean
        #-----------
        
        State["evap_flux_ocean"], State["lhf_evap_ocean"] = surface_evaporation(State["Tos"], State["Tao"], State["Q_ocean"], Var["ocean_height"], Var["Lv"], INPUT["r"], Var["atm_rho"], INPUT["tbhfco"], Var["W_ocean"], hydro=0, height=0)
        
        
        # Save weighted average
        #----------------------
        
        State["evap_flux"] = weighted_average(Var, State["evap_flux_land"], State["evap_flux_ocean"])
        State["lhf_evap"]  = weighted_average(Var,  State["lhf_evap_land"], State["lhf_evap_ocean"])
        
        #--------------------
        # Atmospheric Heating
        #--------------------

        State["Tal"] = State["Tal"] + Var["secs_in_day"] / Var["atm_hc"]  * (State["ratmnet_land"] + State["shf_land"] + State["lhf_evap_land"])

        State["Tao"] = State["Tao"] + Var["secs_in_day"] / Var["atm_hc"]  * (State["ratmnet_ocean"] + State["shf_ocean"] + State["lhf_evap_ocean"])


        #-------------------------
        # Surface Heating of ocean
        #-------------------------

        State["To"][Var["idxolyr1"].astype('i8')] = State["To"][Var["idxolyr1"].astype('i8')] + Var["secs_in_day"] / Var["ocean_hc1"]  * (State["rsnet_ocean"] - State["shf_ocean"] - State["lhf_evap_ocean"] ) #- State["lhf_snowfall_ocean"]*(1-State["si_fraction"]) )

        #------------------------
        # Surface heating of land
        #------------------------

        # Heat and cool all land surfaces (both snow/ice covered and bare land)
        State["Tl"] = State["Tl"] + Var["secs_in_day"] / Var["land_hc"] * (State["rsnet_land"] - State["shf_land"] - State["lhf_evap_land"])
            
        # Mixing
        #-------
        State["Ta"]  = weighted_average(Var, State["Tal"], State["Tao"])
        
        
    
        # ocean transport
        #----------------
                     
        advfs, advfs_conv, advfb, advfb_conv, vadvf, vadvf_conv, hdiffi, hdiffi_conv, vdiff, vdiff_conv, hdiffs, hdiffs_conv = ocean_fluxes(State["To"],
                                Var["ocean_rho"], Var["ocean_sphc"], Var["olatb"], ow, oa, State["u1"], State["u2"], Var["olat"], State["ww"], Var["r_earth"], Var["dlatr"], Var["odepth"], Var["odepth_diff"], State["dz"], INPUT["dosh"], INPUT["donh"], INPUT["dh"], State["idxcos"], Var)
        
        
        # ocean heating/cooling  from transport
        #--------------------------------------
        
        State["To"][Var["idxoc"].astype('int')] = State["To"][Var["idxoc"].astype('int')] + ( Var["secs_in_day"]/Var["ocean_hc"] * (advfs_conv + advfb_conv + vadvf_conv + hdiffs_conv + hdiffi_conv + vdiff_conv) )
        
        # atm transport
        #--------------
        
        hadley_cell=1
        
        mse_north, mse_conv, hadley_latent, hadley_dry, eddy_latent, eddy_dry, dry_north, dry_conv, latent_north, latent_conv = moist_heat_transport_v2(Var["lat"], Var["latb"], Var["latbr"], Var["dlatr"], Var["swidth"], Var["sarea"], Var["r_earth"], Var["mean_height"], Var["atm_sphc"], Var["atm_depth"], Var["atm_rho"], Var["Lv"], INPUT["r"], State["Ta"], State["Q"], INPUT["dt"], hadley_cell, Var["idxeq"], hydro=0, height=0)
        
        # atm heating/cooling from transport
        #--------------------------------------
        
        State["Tal"] = State["Tal"] + (Var["secs_in_day"]) / Var["atm_hc"]  * (mse_conv)
        State["Tao"] = State["Tao"] + (Var["secs_in_day"]) / Var["atm_hc"]  * (mse_conv)
        
        #---------------------------------
        # Calculate mean ocean temperature
        #---------------------------------
        
        State["mot"] = np.array([mean_ocean_temperature(State["To"], Var)])

        #-----------------------------------------------------------
        # Set atmospheric temperatures/humdities to weighted average
        #-----------------------------------------------------------

        # atmospheric temperatures
        State["Ta"]  = weighted_average(Var, State["Tal"], State["Tao"])
        State["Tal"] = State["Ta"].copy()
        State["Tao"] = State["Ta"].copy()
    

        #------------------------------
        # Sea ice formation and melting
        #------------------------------
            
        # calculate changes in sea-ice cover
        State["To"][Var["idxolyr1"].astype('i8')], State["Tsi"], State["si_thick"], State["si_volume"], State["si_fraction"], State["si_melt_flux"] = sea_ice(State["To"][Var["idxolyr1"].astype('i8')], State["Tsi"], State["si_volume"], State["snowfall_flux_ocean"], State["si_fraction"], Var["lat"], Var["idxal"].astype('i8'), Var["ice_rho"], Var["Lm"],
                    Var["Ksi"], Var["ocean_hc1"], Var["oarea"], Var["secs_in_day"], Var["sea_ice_sphc"], Var["si_ithick"])

        # calculate areal extent of sea-ice in each hemisphere
        State["si_area_nh"][0] = np.nansum(State["si_fraction"][Var["idxnh"].astype('i8')]*Var["oarea"][Var["idxnh"].astype('i8')])
        State["si_area_sh"][0] = np.nansum(State["si_fraction"][Var["idxsh"].astype('i8')]*Var["oarea"][Var["idxsh"].astype('i8')])

        #--------------------------
        # Surface ocean temperature
        #--------------------------

        # Set ocean surface temperatures
        State["Tos"] = np.copy(State["To"][Var["idxolyr1"].astype('i8')])

    
        # Set to sea-ice temperature when ocean totally covered by sea-ice.
        i = np.where(State["si_fraction"] == 1)
        State["Tos"][i] = np.copy(State["Tsi"][i])
            
        #------------------------------------------
        # Calculate the average surface temperature
        #------------------------------------------
        
        State["Ts"] = weighted_average(Var, State["Tl"], State["Tos"])
        
        #------------------------
        # Global mean temperature
        #------------------------

        State["Tag"][0]    = global_mean(State["Ta"], Var)
        State["Tagx"][0]   = global_mean(State["Ta"] - (Var["lr"] * Var["mean_height"]), Var)
        State["Talgx"][0]  = global_nanmean(State["Tal"] - (Var["lr"] * Var["land_height"]), Var)
        State["Taogx"][0]  = global_nanmean(State["Tao"] - (Var["lr"] * Var["ocean_height"]), Var)
        State["Tsg"][0]    = global_mean(State["Ts"], Var)
        State["Tsgx"][0]   = global_mean(State["Ts"] - (Var["lr"] * Var["mean_height"]), Var)
        State["rtimb"][0]  = global_mean(State["rtnet"], Var)
        
        
        
        To_all_years[:,year,day] = State["To"]
        
        if year == 2999: # if last year
            
            # temperature
            To_last_year[:,day]  = State["To"]
            
            # outgoing longwave
            olw_last_year[:,day]  = State["rsnet_ocean"]
            
            # shf
            shf_last_year[:,day]  = State["shf_ocean"]
            
            # outgoing longwave
            lhf_last_year[:,day]  = State["lhf_evap_ocean"]
            
            # ocean heat transport convergence
            oc_conv_last_year[:,day] = (advfs_conv + advfb_conv + vadvf_conv + hdiffs_conv + hdiffi_conv + vdiff_conv) 
            
            
# plot ocean temperature
#-----------------------

# initialize arrays
mot = np.zeros((3000)) # mean ocean temperature
sot = np.zeros((3000)) # surface ocean temperature


for i in np.arange(0, 3000):
    
    mot[i] = mean_ocean_temperature(To_all_years[:,i,:].mean(axis=1), Var)
    
    sot[i] = np.nansum(To_all_years[:,i,:].mean(axis=1)[Var["idxolyr1"].astype('int')] * Var["oarea"]) / np.nansum(Var["oarea"])
    
fig, axs = pplt.subplots()
axs.plot(np.arange(1,3000+1), mot, color = "red9", label = ["MOT"])
axs.plot(np.arange(1,3000+1), sot, color = "blue9", label = "SOT")
axs.legend()


# plot surface energy balance for last model year
#------------------------------------------------

fig2, axs2 = pplt.subplots()

# axs2.plot(Var["lat"], inso, color = "yellow9", label = "ASR")
axs2.plot(Var["lat"], olw_last_year.mean(axis=1), color = "yellow", label = "AR")
axs2.plot(Var["lat"], -shf_last_year.mean(axis=1), color = "red9", label = "SHF")
axs2.plot(Var["lat"], -lhf_last_year.mean(axis=1), color = "green9", label = "LHF")
axs2.plot(Var["olat"], oc_conv_last_year[0:Var["olat"].size,:].mean(axis=1), color = "orange9", label = "OHT")
axs2.plot(Var["lat"],
                      olw_last_year.mean(axis=1)-
                      shf_last_year.mean(axis=1) -
                      lhf_last_year.mean(axis=1)
                       + np.concatenate((np.zeros((4)), oc_conv_last_year[0:Var["olat"].size,:].mean(axis=1), np.zeros((2)))), color = "black", label = "Balance"
                      )

axs2.legend()
                      
                      

        
            
            