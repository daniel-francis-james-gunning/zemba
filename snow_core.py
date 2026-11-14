# -*- coding: utf-8 -*-

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange

@njit(nogil = True, cache=False)
def snowfall(tas, precip_flux, land_height, Lm, ice_rho):
    
    '''
    Determines the snowfall fraction, average snowfall flux and the associated 
    latent heat flux.
    '''
    
    lapse = 0.0065 * land_height
    
    # --- Snowfall fraction ---
    #--------------------------
    
    # modify surface temperature for elevation-lapse rate effect
    tasz = tas - lapse
    
    snowfall_frac = 0.0
    
    # complete snow cover
    if tasz < 260.0: 
        snowfall_frac = 1.0
    
    # partial snow
    elif (tasz >= 260.0) and (tasz <= 280.0):
        snowfall_frac = 0.05 * (280 - tasz)
        
    # no snow  
    elif tasz > 280.0:
        snowfall_frac = 0.0
    
    # --- Average snowfall flux (kg m^-2 s^-1) ---
    # --------------------------------------------
    
    snowfall_flux = precip_flux * snowfall_frac
    
    # --- Snowfall rate (m per day) ---
    # ---------------------------------
    
    snowfall_rate = (snowfall_flux / ice_rho) * (60*60*24) 
    
    # --- Latent heat flux of snowfall (W m^-2) ---
    # ---------------------------------------------
    
    snowfall_lhf = snowfall_flux * Lm
    
    return snowfall_frac, snowfall_flux, snowfall_rate, snowfall_lhf


@njit(nogil = True, cache=False)
def snow_accumulation(snow_frac, snow_thick, sarea, snow, snowfall_rate, 
                      snowfall_frac, precip_flux, icerho):
    
    '''
    Determines changes in snowpack thickness and snowpack fraction of the land
    surface during accumulation
        
        Optional:
            
            snow = 1: Snowfall is evenly distributed over latitude band
            
            snow = 2: Snowfall covers area of latitude band according to the 
                      snowfall fraction
    '''
    
    # --- Snowpack variables ---
    # --------------------------
    
    # area 
    snow_area = sarea*snow_frac
    
    # volume
    snow_volume = snow_thick * snow_area
    
    
    # --- Modification to snow fraction and thickness ---
    # ---------------------------------------------------
      
    
    if snow == 1: # if snowfall fraction is evenly distributed
        
        # new snow thickness
        snow_thick += snowfall_rate
        
        # new snow fraction
        if snow_thick > 0.0:
            snow_frac = 1.0
        elif snow_thick <= 0.0:
            snow_frac = 0.0
        if snow_thick > 100.0:
            snow_thick = 100.0
            

    if snow == 2: # if snowfall fraction is unevenly distributed
        
        # if snowfall fraction is greater than snowpack fraction
        if (snowfall_frac > 0.0) and (snowfall_frac >= snow_frac):
            # new snow fraction
            snow_frac = snowfall_frac
            # new snow area
            snow_area = snow_frac*sarea
            # new snow thickness
            snow_thick = (snow_volume/snow_area) + ((precip_flux/icerho)*86400)
            
            
        # if snowfall fraction is less than snowpack fraction   
        if (snowfall_frac > 0.0) and (snowfall_frac < snow_frac):
            # additonal snow volume (m^3)        
            add_snow = (snowfall_frac*sarea) * ( (precip_flux/icerho)*86400)
            # new snow thickness
            snow_thick += (add_snow/snow_area)
            if snow_thick > 100.0:
                snow_thick = 100.0
                
    return snow_frac, snow_thick


# @njit(nogil = True, cache=False)
# def snow_melt(tl, land_height, snow, snow_thick, K, land_hc, Lm, ice_rho, 
#               snow_frac, secs_in_day, mask):
    
#     '''
#     Determines changes in snowpack thickness and snowpack fraction of the land
#     surface during melting:
        
#         Optional:
            
#             snow = 1: Snowfall is evenly distributed over latitude band
            
#             snow = 2: Snowfall covers area of latitude band according to the 
#                       snowfall fraction
#     '''
    
#     lapse = 0.0065 * land_height
#     snow_frac0 = snow_frac
#     snow_thick0 = snow_thick
#     melt_rate = 0.0
#     pmelt_rate = 0.0
    
#     # --- Surface temperature ---
#     # ---------------------------
    
#     tlz = tl - lapse # modified for lapse rate effect
    
    
#     # --- Average snow melt rate over snow area ---
#     # ---------------------------------------------
    
#     if snow == 1: # if snowfall fraction is evenly distributed
        
#         # when land is snow-covered with temperatures exceeding 0C.
#         if (snow_thick > 0.0) and (tlz > K):
            
#             # melt snow with excess temperature
#             melt_rate = (tlz - K) * ( (land_hc) / (Lm*ice_rho) )
            
#             # potential melt rate (for SMB calc.)
#             pmelt_rate = melt_rate
            
#             # reset temperatures of snow-covered land to 0 C.
#             tlz = K
            
#     if snow == 2: # if snowfall fraction is unevenly distributed
        
#         # when land is snow-covered with temperatures exceeding 0C.
#         if (snow_thick > 0.0) and (tlz > K):
            
#             # melt snow with excess temperature
#             melt_rate = (tlz - K) * ( land_hc/(Lm*ice_rho) ) * (1/snow_frac0)
            
#             # potential melt rate (for SMB calc.)
#             pmelt_rate = melt_rate
            
#             # reset temperatures of snow-covered land to 0C.
#             tlz = K
            
    

            
#     # --- Modify snow thickness due to ablation ---
#     # ---------------------------------------------
     
#     if snow == 1: # if snowfall fraction is evenly distributed
    
#         if melt_rate > 0.0:
        
#             # new snow thickness
#             snow_thick -= melt_rate
        
        
#     if snow == 2: # if snowfall fraction is unevenly distributed
        
#         # equally partition average snow melt into vertical and lateral melting
        
#         if melt_rate > 0.0:
        
#             # new snow thickness
#             snow_thick -=  melt_rate*0.5
            
#             # new snow fraction
#             snow_frac -= ( (melt_rate*0.5) * snow_frac0 ) / snow_thick
            
#             # convert nan's to zeros
#             if np.isnan(snow_frac) == True:
#                 snow_frac = 0.0
            


            
    
#     # --- Convert excess melt to surface heating ---
#     # ----------------------------------------------
    

#     # locate areas of excess melt
#     if (melt_rate>0.0) and (melt_rate >= snow_thick0):
    
#         # amount of excess melt
#         excess_melt = melt_rate - snow_thick0
    
#         if snow == 1: # if snowfall fraction is evenly distributed
        
#             # correct excess melt 
#             melt_rate = snow_thick0
            
#             # additional surface heating
#             tlz += (excess_melt*ice_rho*Lm) / land_hc
            
#             # correct snow thickness and snow fraction
#             snow_thick = 0.0
#             snow_frac  = 0.0
            
#         if snow == 2: # if snowfall fraction is unevenly distributed
            
#             # correct excess melt 
#             melt_rate = snow_thick0
            
#             # additional surface heating
#             tlz += (excess_melt*snow_frac0*ice_rho*Lm) / land_hc
            
#             # correct snow thickness and snow fraction
#             snow_thick = 0.0
#             snow_frac  = 0.0
            
      

    
#     # --- Latent heat flux of melt (averaged over total area) ---
#     # -----------------------------------------------------------
    
#     melt_flux = ( (melt_rate*snow_frac0*ice_rho)/secs_in_day) * Lm
    
    
#     # --- "Re-modify" temperature for lapse rate effect ---
#     # -----------------------------------------------------
    
#     tl = tlz + lapse
    
#     return snow_thick*mask, snow_frac*mask, tl, melt_rate*mask, pmelt_rate*mask, melt_flux*mask
    


@njit(nogil = True, cache=False)
def snow_melt(tl, land_height, snow, snow_thick, K, land_hc, Lm, ice_rho, 
              snow_frac, secs_in_day, mask):
    
    '''
    Determines changes in snowpack thickness and snowpack fraction of the land
    surface during melting:
        
        Optional:
            
            snow = 1: Snowfall is evenly distributed over latitude band
            
            snow = 2: Snowfall covers area of latitude band according to the 
                      snowfall fraction
    '''
    
    lapse = 0.0065 * land_height
    snow_frac0 = snow_frac
    snow_thick0 = snow_thick
    melt_rate = 0.0
    pmelt_rate = 0.0
    
    # --- Surface temperature ---
    # ---------------------------
    
    tlz = tl - lapse # modified for lapse rate effect
    
    
    # --- Average snow melt rate over snow area ---
    # ---------------------------------------------
    
    if snow == 1: # if snowfall fraction is evenly distributed
    
        if tlz > K:
            
            # potential melt rate (for SMB calc.)
            pmelt_rate = (tlz - K) * ( (land_hc) / (Lm*ice_rho) )
            
            if snow_thick > 0.0:
                
                melt_rate = pmelt_rate
                
                # reset temperatures of snow-covered land to 0 C.
                tlz = K
                
                
        
    if snow == 2: # if snowfall fraction is unevenly distributed
    
        if tlz > K:
            
            # potential melt rate (for SMB calc.)
            pmelt_rate = (tlz - K) * ( land_hc/(Lm*ice_rho) ) * (1/snow_frac0)
            
            if snow_thick > 0.0:
                
                
                melt_rate = pmelt_rate

                # reset temperatures of snow-covered land to 0C.
                tlz = K
            
            
    # --- Modify snow thickness due to ablation ---
    # ---------------------------------------------
     
    if snow == 1: # if snowfall fraction is evenly distributed
    
        if melt_rate > 0.0:
        
            # new snow thickness
            snow_thick -= melt_rate
        
        
    if snow == 2: # if snowfall fraction is unevenly distributed
        
        # equally partition average snow melt into vertical and lateral melting
        
        if melt_rate > 0.0:
        
            # new snow thickness
            snow_thick -=  melt_rate*0.5
            
            # new snow fraction
            snow_frac -= ( (melt_rate*0.5) * snow_frac0 ) / snow_thick
            
            # convert nan's to zeros
            if np.isnan(snow_frac) == True:
                snow_frac = 0.0
            


            
    
    # --- Convert excess melt to surface heating ---
    # ----------------------------------------------
    

    # locate areas of excess melt
    if (melt_rate>0.0) and (melt_rate >= snow_thick0):
    
        # amount of excess melt
        excess_melt = melt_rate - snow_thick0
    
        if snow == 1: # if snowfall fraction is evenly distributed
        
            # correct excess melt 
            melt_rate = snow_thick0
            
            # additional surface heating
            tlz += (excess_melt*ice_rho*Lm) / land_hc
            
            # correct snow thickness and snow fraction
            snow_thick = 0.0
            snow_frac  = 0.0
            
        if snow == 2: # if snowfall fraction is unevenly distributed
            
            # correct excess melt 
            melt_rate = snow_thick0
            
            # additional surface heating
            tlz += (excess_melt*snow_frac0*ice_rho*Lm) / land_hc
            
            # correct snow thickness and snow fraction
            snow_thick = 0.0
            snow_frac  = 0.0
            
      

    
    # --- Latent heat flux of melt (averaged over total area) ---
    # -----------------------------------------------------------
    
    melt_flux = ( (melt_rate*snow_frac0*ice_rho)/secs_in_day) * Lm
    
    
    # --- "Re-modify" temperature for lapse rate effect ---
    # -----------------------------------------------------
    
    tl = tlz + lapse
    
    return snow_thick*mask, snow_frac*mask, tl, melt_rate*mask, pmelt_rate*mask, melt_flux*mask
    


    

    



    
    
    
    
    

