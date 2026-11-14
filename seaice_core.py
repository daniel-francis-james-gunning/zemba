# -*- coding: utf-8 -*-

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange

@njit(nogil = True, cache=False)
def sea_ice(to, sivol, tsi, si_frac, rholm, ksi, ocean_hc, o_area, step, snfl,
            ice_rho, sea_ice_sphc, si_ithick, lm, omask):
    
    '''
    Computes sea-ice volume and the fraction of ocean surface
    area based on the surface ocean temperature. Based on the 
    parameterizations of Gildor and Tziperman (2001).
    '''
    
    # -- set the sea ice vars from previous time-step ---
    #----------------------------------------------------
    
    sivol0 = sivol       # sea ice volume (m^3)
    to0 = to             # surface ocean temperature
    tsi0 = tsi           # sea-ice temperature
    si_frac0 = si_frac   # sea-ice fraction
    
    # --- initialize ---
    # ------------------
    
    h_avail = 0.0        # Energy available for sea-ice formation (W)
    h_need1 = 0.0        # Energy needed to melt all sea-ice (W)
    h_need2 = 0.0        # Energy needed to melt all sea-ice (W)
    si_melt_flux = 0.0   # Sea ice melt flux (in W/m2)
    sf_vol = 0.0         # snowfall in m^3
    si_thick = 0.0       # Sea ice thickness (m)
    exc_melt = 0.0       # Excess melt ()
    
    # --- sea ice formation ---
    # -------------------------
    
    if to0 < ksi: # if surface ocean temperature is below freezing point
    
        # calculate energy available for surface freezing (in Watts or J/s)
        h_avail = ( (ocean_hc*o_area) / step) * (ksi-to0)
        
        # convert snowfall into sea ice volume
        sf_vol = (snfl/ice_rho) * (o_area*si_frac0) 
        
        # update sea ice volume
        sivol = sivol0 + step * ( (h_avail/rholm) + sf_vol)
        
        # reset surface ocean temperature to freezing
        to = ksi
        
    # --- sea ice melting ---
    # -----------------------
    
    elif (to0 > ksi) and (sivol0 > 0.0): # if surface ocean temperature is above freezing point
    
        # calculate energy available for surface melting (in Watts or J/s)
        h_avail = ( (ocean_hc*o_area) / step) * (ksi-to0)
        
        # convert snowfall into sea ice volume
        sf_vol = (snfl/ice_rho) * (o_area*si_frac0) 
        
        # energy needed to raise the ice temperature to freezing point (in J/s)
        h_need1 = ( (ksi-tsi0) * (ice_rho*sea_ice_sphc*si_ithick*(o_area*si_frac0)) ) / step  
        
        # energy needed to melt all the ice (in J/s)
        h_need2 = (sivol0*rholm) / step
        
        # if not enough energy to raise to melting point
        if (to0>ksi) and (sivol0>0.0) and (-(h_avail)<h_need1):
            
            # raise the temperature of the ice 
            tsi = tsi - (h_avail/(ice_rho*sea_ice_sphc*si_ithick*(o_area*si_frac0)))* step
            
            # set ocean temperature
            to = ksi
            
        # if energy exceeds melting point but can't melt all ice
        elif (to0>ksi) and (sivol0>0.0) and (-(h_avail)>h_need1) and (-(h_avail) < (h_need1 + h_need2)):
            
            # remove sea-ice due to surface melting- considering snowfall gains.
            sivol = sivol0 + (step*((h_avail+h_need1)/(rholm))+sf_vol) 
            
            # set surface ocean and sea ice temperature to freezing
            to  = ksi
            tsi = ksi
            
        # if there is enough energy available to melt the ice
        elif (to0>ksi) and (sivol0>0.0) and (-(h_avail)>h_need1) and (-(h_avail) > (h_need1 + h_need2)):
            
            # Remove sea-ice due to surface melting- considering snowfall gains.
            sivol = sivol0 + (step * ( (h_avail+h_need1)/(rholm) ) + sf_vol)
                             
            # convert excess melt into additional surface heating
            to = ksi + ( ( (0.0-sivol)*(rholm)) / o_area ) / ocean_hc
        
            # set ice volume to zero
            sivol = 0.0


    # --- calculate new sea-ice fraction ---
    # --------------------------------------
    
    # if sea ice cover is less than ocean surface area
    if sivol > 0.0:
        
        # set the new sea-ice fraction
        si_frac = sivol / (si_ithick*o_area)
        
        # set the sea ice thickness 
        si_thick = si_ithick
        
    else:
        si_frac = 0.0
        si_thick = 0.0
        
    # when sea-ice cover < 1.
    if (si_frac > 0.0) and (si_frac < 1.0):
        
        # set the sea-ice temperature to zero
        tsi = ksi
        
    # when sea-ice cover > 1.
    if si_frac > 1.0:
        
        # Let the sea-ice temperature drop below zero.
        tsi = tsi0 + ( (1-si_frac) * (lm/sea_ice_sphc) )
    
        # set the new sea-ice volume
        sivol = si_ithick * o_area
    
        # set new sea-ice fraction
        si_frac = 1.0
        
    
    si_frac = si_frac*omask
    si_thick = si_thick*omask
    tsi = tsi*omask
        
        
    # --- record melting and ice volume changes ---
    # ---------------------------------------------
        
    # Record changes in ice volume
    sivol_change = sivol - sivol0
    
    # Record changes in energy fluxes due to sea ice changes
    si_melt_flux = (sivol_change*rholm) / (o_area*step)
    si_melt_flux = si_melt_flux*omask
    
    return to, tsi, si_thick, sivol, si_frac, si_melt_flux

       

        
        





    
   



