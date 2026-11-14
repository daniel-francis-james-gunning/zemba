# -*- coding: utf-8 -*-

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange

@njit(nogil=True, cache=True)
def shortwave_radiation(clt, ta, alpha_nc, alpha_cl, S_u, z, S, clt_tau):
        
    """
    Calculate shortwave radiative fluxes using Bintanja (1996) parameterization.
    
    Parameters
    ----------
    clt : 
        Fractional cloud cover (0-1)
    ta :
        Near-surface air temperature (K) 
    alpha_nc :
        Clear-sky albedo (0-1) 
    alpha_cl :
        Cloudy albedo (0-1) 
    S_u : 
        Zenith angle of solar insolation 
    z : 
        Surface altitude (m)
    S : 
        Solar insolation (W/m²)
    clt_tau : 
        Cloud optical depth parameter 
  
    Returns
    -------
    
    sw_toa_up :
        Upward shortwave flux at the top of the atmosphere (W/m²)
    sw_boa_down :
        Downward shortwave flux at the surface (W/m²)
    sw_boa_up : 
        Upward shortwave flux at the surface (W/m²)
    """

    
    # --- Transmission as function of ozone ---
    # -----------------------------------------
    
    oz = 0.96586
    
    # --- Transmission as function of rayleigh scattering ---
    # -------------------------------------------------------
    
    ra = 0.92648
    
    # --- transmission as function of surface air temperature (water vapour) ---
    # --------------------------------------------------------------------------
    
    if ta < 180:
        ta_d = 1
        ta_u = 1.1807
    else:
        ta_d = -0.20518 + 0.016014 * ta - 6.6733e-5 * (ta**2) + 8.30352e-8 * (ta**3)
        ta_u = -0.29673 + 0.02004 * ta  -8.5214e-5 * (ta**2) + 1.08069e-7 * (ta**3)
        
    # --- Transmission as function of albedo---
    # -----------------------------------------
    
    # Clear sky (nc)
    alpha_nc_d = 0.9891 + 0.0512 * alpha_nc  + 0.0124 * (alpha_nc**2)
    alpha_nc_u = 0.3351 + 3.2821 * alpha_nc + 0.1969 * (alpha_nc**2)
    
    # Overcast (cl)
    alpha_cl_d = 0.9891 + 0.0512 * alpha_cl  + 0.0124 * (alpha_cl**2)
    alpha_cl_u = 0.8971 + 0.4316 * alpha_cl + 0.3643 * (alpha_cl**2)
    
    # ---Transmission as function of zenith angle---
    # ----------------------------------------------
    
    # Clear sky (nc)
    zn_nc_d = 0.6135 + 1.5377 * S_u - 1.9297 * (S_u**2) + 0.8602 * (S_u**3)
    zn_nc_u = 1.7286 - 2.8755 * S_u + 5.2534 * (S_u**2) - 4.7598 * (S_u**3) + 1.6551 * (S_u**4)
    
    # Overcast (cl)
    zn_cl_d = 0.500 + 1.1071 * S_u - 0.2088 * (S_u**2)
    zn_cl_u = 1.7501 - 0.08207 * S_u - 1.5343 * (S_u**2) + 0.8695 * (S_u**3)
    
    # ---Transmission as function of optical depth---
    # -----------------------------------------------
    
    tau = np.log(clt_tau) # cloud optical depth
    
    # downwards
    tau_d = 0.8447 - 0.1664 * tau - 0.0113 * (tau**2)
    delta_tau_d = -0.0503 - 0.02910 * tau - 0.0331 * (tau**2) + 0.0098 * (tau**3) 
    tau_alpha_d = tau_d + ((alpha_cl - 0.2) * (delta_tau_d))/0.5
   
    # Upward
    tau_u = 0.6422 + 0.1385 * tau + 0.0708 * (tau**2) - 0.0114 * (tau**3) 
    delta_tau_u = 0.32048 - 0.11872 * tau - 0.07024 * (tau**2) + 0.0123 * (tau**3)
    tau_alpha_u = tau_u + ((alpha_cl - 0.2)/0.6) * delta_tau_u
    
    # ---Total transmission---
    # ------------------------
    
    # Clear sky (nc)
    T_nc = oz * ra * ta_d * alpha_nc_d * zn_nc_d + 0.000022 * z
    
    # Overcast (cl)
    T_cl = oz * ra * ta_d * alpha_cl_d * zn_cl_d * tau_alpha_d * 0.628 + 0.000022 * z

    # --- Upwards solar radiation at top of the atmosphere---
    # -------------------------------------------------------
    
    # Clear sky (nc)
    sw_toa_up_nc = 0.18744 * S * ta_u * alpha_nc_u * zn_nc_u
    
    # Overcast (cl)
    sw_toa_up_cl = 0.32444 * S * ta_u * alpha_cl_u * zn_cl_u * tau_alpha_u
    
    # Weighted average
    sw_toa_up = sw_toa_up_nc * (1 - clt) + sw_toa_up_cl * clt

    # ---Downward solar radiation at surface---
    # -----------------------------------------
    
    # Clear sky (nc)
    sw_boa_down_nc = S * T_nc
    
    # Overcast (cl)
    sw_boa_down_cl = S * T_cl
    
    # Weighted average
    sw_boa_down = (1 - clt) * sw_boa_down_nc + clt * sw_boa_down_cl
    

    # ---Upwards solar radiation at surface---
    # ----------------------------------------
    
    # Clear sky (nc)
    sw_boa_up_nc = alpha_nc * sw_boa_down_nc
    
    # Overcast (cl)
    sw_boa_up_cl = alpha_cl * sw_boa_down_cl
    
    # Weighted average
    sw_boa_up = sw_boa_up_nc * (1 - clt) + sw_boa_up_cl * clt
    
    return sw_toa_up, sw_boa_down, sw_boa_up 





@njit(nogil=True, cache=True)
def longwave_radiation(sigma, co2, ta, ts, gmat, z, clt, e_cld, ghg_factor):
        
    """
    Calculate longwave radiative fluxes using Bintanja (1996) parameterization.
    
    Parameters
    ----------
    
    sigma : 
        Stefan-Boltzmann constant (W/m²/K⁴)
    co2 : 
        Atmospheric CO2 concentration (kg/kg)
    ta :
        Near-surface air temperature (K) 
    ts : 
        Surface temperature (K) 
    gmat : 
        Global mean air temperature (K)
    z : 
        Surface altitude (m)
    clt : 
        Fractional cloud cover (0-1)
    e_cld :
        Cloud emissivity (0-1)
    ghg_amp : 
        Greenhouse gas amplification factor
        
    Returns
    -------
    
    lw_boa_up :
        Upward longwave flux at the surface (W/m²)
    lw_boa_down :
        Downward longwave flux at the surface (W/m²)
    lw_toa_up : 
        Upward longwave flux at the top of atmosphere (W/m²)
    """
    
    # ---  pre-industrial global mean surface air temperature ---
    # -----------------------------------------------------------
    
    # from Stap et al. 2014 to adjust clear sky upwards radiation at the top of the atmosphere
    gmat_modern = 287.8 
    
    # --- standard longwave fluxes at the bottom (boa) and top (toa) of the atmosphere ---
    # ------------------------------------------------------------------------------------
    
    # Clear sky (nc)
    st_nc_d = 266.6 # boa downwards
    st_nc_u = 245.7 # toa upwards
    
    # Overcast (cl)
    st_cl_d = 330.8 # boa downwards
    st_cl_u = 215.7 # toa upwards
    
    # --- adjust temperature for altitude ---
    # ---------------------------------------
    
    taz = ta - 0.0065 * z
    tsz = ts - 0.0065 * z
    
    # --- deviations from standard due to air temperature ---
    # -------------------------------------------------------
    
    # Clear sky (nc)
    ta_nc_d = -851.1 + 10.622 * taz - 0.06435 * (taz**2) + 1.328e-4 * (taz**3)
    ta_nc_u = -268.0 - .0107 * taz + 0.0004903 *(taz**2) + 1.0204e-5 * (taz**3) - 1. * ((gmat - gmat_modern)**1.) 
 
    # Overcast (cl)
    ta_cl_d = -533.2 + 3.8795 * taz - 0.02621 * (taz**2) + 6.8068e-5 * (taz**3)
    ta_cl_u = -282.2 + 1.1775 * taz - 8.09e-3 * (taz**2) + 2.6393e-5 * (taz**3)
    
    # --- deviations from standard due to CO2 ---
    # -------------------------------------------

    # greenhouse gas amplification: radiative forcing per CO2 doubling multiplied by factor to account for non-CO2 GHGs
    ghg_amp = (ghg_factor * 3.7)/3.5

    # natural log of CO2 conc.
    ln_co2 = np.log(co2) 

    # Clear sky (nc)
    co2_nc_d = (3.7592 + 0.4950 * ln_co2) * 3. * ghg_amp
    co2_nc_u = (-14.77 - 1.9589 * ln_co2) * 3. * ghg_amp

    # Overcast (cl)
    co2_cl_d = (1.3064 + 0.1673 * ln_co2) * 3. * ghg_amp
    co2_cl_u = (-11.198 - 1.488 * ln_co2) * 3. * ghg_amp
    
    
    # --- deviations from standard due to cloud top altitude ---
    # ----------------------------------------------------------
    
    #  not included
    
    # cta_cl_u = 40.3 - 5.474 * cta - 0.518  * (cta**2)
    # cta_cl_d = 28.2 - 6.376 * cta + 0.1469 * (cta**2)
    
    # --- deviations from standard due to cloud emissivity ---
    # --------------------------------------------------------
    
    # Overcast (cl) 
    cem_cl_d = -54.1 + 54.1 * e_cld
    cem_cl_u = 79.9 - 79.9 * e_cld
    
    # --- surface upward longwave radiation ---
    # -----------------------------------------

    lw_boa_up = sigma * (tsz**4) # emissivity of surface assumed 1
    
    # --- downward longwave radiation at bottom of atmosphere ---
    # -----------------------------------------------------------
    
    # Clear sky (nc)
    lw_nc_d = st_nc_d + ta_nc_d + co2_nc_d 
    
    # Overcast (cl)
    lw_cl_d = st_cl_d + ta_cl_d + co2_cl_d + cem_cl_d #+ cta_cl_d
    
    # Weighted by cloud cover
    lw_boa_down = (1 - clt) * lw_nc_d + clt * lw_cl_d
    
    # --- upward longwave radiation at top of atmosphere ---
    # ------------------------------------------------------
    
    # Clear sky (nc)
    lw_nc_u = st_nc_u + ta_nc_u + co2_nc_u
    
    # Overcast
    lw_cl_u = st_cl_u + ta_cl_u + co2_cl_u + cem_cl_u# + cta_cl_u
    
    # Weighted by cloud cover
    lw_toa_up = (1 - clt) * lw_nc_u + clt * lw_cl_u
    
    return lw_boa_up, lw_boa_down, lw_toa_up