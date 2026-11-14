# -*- coding: utf-8 -*-

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange

_COS80 = np.cos(np.deg2rad(80.0))
_COS60 = np.cos(np.deg2rad(60.0))

@njit(nogil=True, cache=True)
def ocean_albedo(si_frac, si_melt_flux, alpha_si_min, alpha_si_max, cza, mask):
    
    """
    Calculates the surface albedo over ocean
    """
    
    # --- Sea ice albedo ---
    # ----------------------
    
    alpha_si = 0.0
    
    if (si_frac > 0.0) and (si_melt_flux > 0.0):
        
        alpha_si = alpha_si_min
        
    elif (si_frac > 0.0) and (si_melt_flux <= 0.0):
        
        alpha_si = alpha_si_max
            
        
    # --- Modify albedo for high zenith angles ---
    #---------------------------------------------
        
    # Albedo value corrected by taking into account the solar zenith angle
    # dependancy- follwing equation 5 & 6 of Lefebre et al., (2003)
    
    cosz = min(cza, _COS80)

    # where the zenith angle is greater than 60 degrees
    if (si_frac > 0.0) and (cza < _COS60):
        
        alpha_si = alpha_si + max(0.0, 0.032 * 0.5 * (3.0/(1.0+4.0*cosz) - 1.0))
        

    # --- Open ocean albedo ---
    #--------------------------
    
    # Calculates the broadband surface ocean albedo as a function of the 
    # solar zenith angle, based on the parameterization of Taylor et al.,
    # (1996)
     
    alpha_open = 0.037 / ( (1.1 * cza**1.4) + 0.15)
    
    # ---Average ocean albedo ---
    #----------------------------
    
    # Weighted average
    return ( alpha_open * (1.0 - si_frac) + alpha_si * si_frac ) * mask



@njit(nogil=True, cache=True)
def land_albedo_temperature(tal, tl, land_height, alphacs, alphaws, cza, alphai,
                            alphag, ice_frac, mask):
    
    """
    Calculates land albedo as function of temperature when hydrological cycle 
    is turned off.
    """
    
    lapse = 0.0065 * land_height
    
    # --- snow fraction ---
    #----------------------
    
    # elevation modified surface air temperature
    talz = tal - lapse
    
    snow_frac = 0.0
    
    if talz < 260.0: # completelty snow covered 
        snow_frac = 1.0
    
    elif (talz >= 260.0) and (talz <= 280.): # partially snow covered
        snow_frac = 0.05 * (280 - talz)
        
    elif talz > 280.0: # no snow covered band
        snow_frac = 0.0
        
    # --- snow albedo ---
    #---------------------
    
    # elevation modified surface temperature
    tlz = tl - lapse
    
    alpha_snow = 0.0

    if tlz < 263.0: # completely cold covered snow
        alpha_snow = alphacs
        
    elif (tlz >= 263.0) and (tlz <= 273.): # mixed snow albedo
        alpha_snow = alphacs + (alphaws - alphacs) * ( (tlz - 263.0) / 10.0)

    elif tlz > 273.0: # completely warm covered snow
        alpha_snow = alphaws
        
    # --- modify snow albedo for zenith angle ---
    #--------------------------------------------
    
    # Albedo value corrected by taking into account the solar zenith angle
    # dependancy- follwing equation 5 & 6 of Lefebre et al., (2003): 
    # Modelling od snow and ice melt at ETH camp (West Greenalnd): A study
    # of surface albedo
    
    cosz = min(cza, _COS80)
    
    # where the zenith angle is greater than 60
    if cza < _COS60:
        alpha_snow = alpha_snow + max(0.0, 0.032 * 0.5 * (3.0/(1.0+4.0*cosz) - 1.0))
        
    # --- albedo over ice surfaces ---
    #---------------------------------
    
    alpha_ice = alphai * (1.0 - snow_frac) + alpha_snow * snow_frac 
    
    # --- albedo over bare ground surfaces ---
    # ----------------------------------------
    
    alpha_ground = alphag * (1.0 - snow_frac) + alpha_snow * snow_frac 
    
    
    #  --- average land albedo [weighted for ice fraction] --- 
    # --------------------------------------------------------
    
    return ( alpha_ground * (1.0 - ice_frac) + alpha_ice * ice_frac )*mask, snow_frac*mask 
    
    
   
@njit(nogil=True, cache=True)
def land_albedo_snowcover(tl, land_height, alphacs, alphaws, cza, alphai, 
                          alphag, snow_frac, ice_frac, mask):
    
    '''
    Calculates the land surface albedo when the hydrological cycle is turned on.
    '''
    
    lapse = 0.0065 * land_height
    
    # --- Lapse-rate modified land surface temperature ---
    # ----------------------------------------------------
    
    tlz = tl - lapse # modified for lapse rate effect
    
    # --- Temperature-dependant snow albedo over bare ground ---
    # ----------------------------------------------------------
    
    alphasg = 0.0
    
    if tlz < 263.0: # completely cold covered snow
        alphasg = alphacs
        
    elif (tlz >= 263.0) and (tlz <= 273.): # mixed snow albedo
        alphasg = alphacs + (alphaws - alphacs) * ( (tlz - 263.0) / 10.0)

    elif tlz > 273.0: # completely warm covered snow
        alphasg = alphaws
        
    # --- Temperature-dependant snow albedo over ice ---
    # --------------------------------------------------
    
    alphasi = 0.0
        
    if tlz < 263.0: # completely cold covered snow
        alphasi = alphacs
        
    elif (tlz >= 263.0) and (tlz <= 273.): # mixed snow albedo
        alphasi = alphacs + (alphaws - alphacs) * ( (tlz - 263.0) / 10.0)

    elif tlz > 273.0: # completely warm covered snow
        alphasi = alphaws
        
    # --- Modify albedo for high zenith angles ---
    #---------------------------------------------
        
    # Albedo value corrected by taking into account the solar zenith angle
    # dependancy- follwing equation 5 & 6 of Lefebre et al., (2003)
    
    cosz = min(cza, _COS80)

    # where the zenith angle is greater than 60 degrees
    if cza < _COS60:
        
        alphasg = alphasg + max(0.0, 0.032 * 0.5 * (3.0/(1.0+4.0*cosz) - 1.0))
        alphasi = alphasi + max(0.0, 0.032 * 0.5 * (3.0/(1.0+4.0*cosz) - 1.0))
        
    # --- albedo over land-ice ---
    # ----------------------------
    
    alpha_ice = alphai * (1.0 - snow_frac) + alphasi * snow_frac 
    
    # --- albedo over bare ground surfaces ---
    # ----------------------------------------
    
    alpha_ground = alphag * (1.0 - snow_frac) + alphasg * snow_frac 
    
    
    #  --- average land albedo [weighted for ice fraction] --- 
    # --------------------------------------------------------
    
    return ( alpha_ground * (1.0 - ice_frac) + alpha_ice * ice_frac ) * mask 





        
        
    
    
        


        
   
    
    


   
        
        
            
           