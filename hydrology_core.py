# -*- coding: utf-8 -*-

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange

# Constants
# ---------

E0  = 610.78         # Reference vapor pressure for CC equation (Pa)
RV  = 461.52         # Specific gas constant of water vapor (J kg^-1 K^-1)
RD  = 287.0          # Specific gas constant of dry air (J kg^-1 K^-1)
T0  = 273.16         # Reference temperature for CC equation (K)

TEMP_R = 288.15      # reference temperature (K)
R_UNIV = 8.314       # universal gas constant (J/(mol.K))
M_AIR  = 0.029       # molar mass of Earth's air (kg/mol)
P0     = 101325.0    # mean atmopsheric surface pressure (Pa) 
GRAV   = 9.81        # gravitational acceleration (m s^-2)

@njit(nogil=True)
def surface_pressure(h):

    p = P0 * ((TEMP_R - 0.0065*h)/TEMP_R) ** (
        (-GRAV * M_AIR)/(R_UNIV * -0.0065)
    )

    return p


@njit(nogil=True)
def saturation_specific_humidity(Lv, T, h):
    
    '''
    Uses a given temperature and pressure to calculate the saturation
    specific humidity-- the mass of water vapor per mass of moist air, 
    expressed as kg/kg. 
    '''
    
    # calculate the saturation vapor pressure (e_s) from the Clasius -
    #-----------------------------------------------------------------
    # Clapeyron relation (in Pa)
    #---------------------------

    e_s = E0 * np.exp(-(Lv/RV) * ((1/T) - (1/T0)))
    
    # calculate the surface pressure as it changes with height
    #---------------------------------------------------------

    p = surface_pressure(h)
    
    # calculate the saturation specific humidity, q_s, in kg/kg.
    #-----------------------------------------------------------

    q_s = (RD/RV) * (e_s/p)

    return q_s


@njit(nogil=True)
def surface_evaporation(ts, ta, qa, surface_height, Lv, r, atm_rho, lhfr,
                        w, hydro, height):
    
    '''
    Calculates the surface evaporation flux (in kg m^-2 s^-1), rate (in m/yr) 
    and associated heat flux (in W m^-2) for land or ocean, depending on 
    the "surface_type" keyword. Evaporation flux calculated using the 
    traditional bulk aerodynamical formula. Positive is upwards.
    '''
    
    lapse = 0.0065 * surface_height

    # Saturation specific humidity of the surface
    #--------------------------------------------
    
    if height == 0:
    
        # lapse-rate modified surface temperature 
        tsz = ts
        
        # saturation specific humidity of surface
        qss = saturation_specific_humidity(Lv, tsz, 0.)
    
    elif height == 1:
    
        # lapse-rate modified surface temperature 
        tsz = ts - lapse
        
        # saturation specific humidity of surface
        qss = saturation_specific_humidity(Lv, tsz, surface_height)
    
 
    # Saturation specific humidity of the near-surface atmosphere
    #------------------------------------------------------------
    
    if hydro==0: # if hydrological cycle is turned off
    
        if height == 0:
            
            # lapse-rate modified surface temperature
            taz = ta 
            
            # saturation specific humidity of near-surface air
            qsa = saturation_specific_humidity(Lv, taz, 0.)
            
        elif height == 1:
            
            # lapse-rate modified surface air temperature   
            taz = ta - lapse
            
            # saturation specific humidity of near-surface air
            qsa = saturation_specific_humidity(Lv, taz, surface_height)
        
        # specific humidity of near-surface air
        q = qsa * (r/100)
        
    elif hydro==1: # if hydrological cycle is turned on
    
        # specific humidity of near-surface air (as determined by hydrological cycle)
        q = qa 
        

    # upwards evaporative flux (kg m^-2 s^-1)
    #----------------------------------------
    
    evaporation_flux = atm_rho * lhfr * w * (qss - q) 
    

    # upwards latent heat flux loss (W m^-2)
    #---------------------------------------
    
    evaporation_lhf = evaporation_flux * Lv
    
    
    return evaporation_flux, evaporation_lhf



@njit(nogil = True)
def precipitation(ta, qa, surface_height, Lv, r, atm_rho, atm_depth, secs_in_day, 
                  mask, water_rho, height):
    
    '''
    Calculates the precipitation flux (in kg m^-2 s^-1), rate (in m/yr)
    and associated heat flux (in W m^-2). 
    '''
    
    lapse = 0.0065 * surface_height
    
    # Saturation specific humidity of the atmosphere
    #-----------------------------------------------
    
    if height == 0:
        
        # surface air temperature 
        taz = ta
        
        # saturation specific humidity of near-surface air
        qsa = saturation_specific_humidity(Lv, taz, 0.)
        
    if height == 1:
        
        # surface air temperature 
        taz = ta - lapse
        
        # saturation specific humidity of near-surface air
        qsa = saturation_specific_humidity(Lv, taz, surface_height)
    

    # Relative humidity of near-surface air
    #--------------------------------------
    
    rs = (qa / qsa) * 100
    

    # Precipitation flux (kg / m^-2 s^-1)
    #------------------------------------
    
    if rs > r:
        
        # precipitation flux
        precipitation_flux = (atm_rho * atm_depth) / (3*secs_in_day) * (qa - ( (r/100) * qsa) )
        precipitation_flux *= mask
        
        # Precipitation rate (m/day) 
        precipitation_rate = (precipitation_flux / water_rho) * secs_in_day
        
        # Latent heat flux of precipitation (W m^-2)        
        precipitation_heat_flux = precipitation_flux * Lv
        
    else:
        
        # precipitation flux
        precipitation_flux = 0.
        precipitation_flux *= mask
        
        # Precipitation rate (m/day) 
        precipitation_rate = 0.
        
        # Latent heat flux of precipitation (W m^-2)        
        precipitation_heat_flux = 0.
        

    return precipitation_flux, precipitation_rate, precipitation_heat_flux


