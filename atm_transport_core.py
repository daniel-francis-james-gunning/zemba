# -*- coding: utf-8 -*-

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange
from hydrology_core import saturation_specific_humidity

@njit(nogil=True)
def atm_transport(ta, q, cp, lv, mean_height, r, EQ, sz, G_face, D, w_face, gms_factor,
                  nlat, area, hydro, FS, FL, mse_north, dry_conv, lat_conv):
    """
    One sub-step of moist heat transport, two passes.
 
    Pass 1: read m/ms/ml, write face fluxes FS/FL/mse_north  (no Ta/Q writes)
    Pass 2: read fluxes, write Ta/Q                            (no flux writes)
 
    Each pass is prange-safe: within a pass, iteration k writes only to index
    it owns and reads only indices it does not write. Decorator parallel=False
    treats prange as range (serial). Flip parallel=True at large N for threads.
 
    FS, FL, mse_north : caller-supplied (sz,) scratch arrays (reused across
    calls to avoid per-call allocation). They are fully overwritten here.
    """
    
    
    # moist static energy
    #--------------------
    
    # dry static energy
    ms = cp * ta  
    
    # latent energy
    if hydro == 1:
        qa = q.copy()
    else:
        qa = saturation_specific_humidity(lv, ta, mean_height) * (r / 100.0)
    ml = lv * qa
    
    # moist static energy
    m=ms+ml
    
    # moist static energy at the equator
    meq = (m[EQ] + m[EQ - 1]) / 2.0
 
    # ---- Northward heat fluxes (prange safe; writes FS/FL/mse_north) ---
   
    # zero the polar boundaries
    FS[0] = 0.0; FL[0] = 0.0; mse_north[0] = 0.0
    FS[sz - 1] = 0.0; FL[sz - 1] = 0.0; mse_north[sz - 1] = 0.0
 
    for b in prange(1, sz - 1):
        
        pref = G_face[b] * D[b]
        
        if w_face.size > 1:
            wf = w_face[b]
        else:
            wf = 0.0
            
        # northward TOTAL
        F_tot = -pref * (m[b] - m[b - 1])
        
        # northward EDDIES
        FS_e  = -pref * (ms[b] - ms[b - 1]) * (1.0 - wf) # sensible
        FL_e  = -pref * (ml[b] - ml[b - 1]) * (1.0 - wf) # latent
        
        # northward HADLEY
        F_hc = F_tot - FS_e - FL_e
        gms = meq * gms_factor - (m[b] + m[b - 1]) / 2.0
        psi = F_hc/gms
        FL_hc = -lv * psi * (qa[b] + qa[b - 1]) / 2.0
        FS_hc = F_hc - FL_hc
        
        # assign
        FS[b] = FS_e + FS_hc
        FL[b] = FL_e + FL_hc
        mse_north[b] = F_tot
        
        
    # ---- Convergence (prange-safe) ----
    
    for i in prange(nlat):
        
        # dry static energy convergence
        dry_conv_i = -(FS[i + 1] - FS[i]) / area[i]
        
        # latent moisture convergence
        lat_conv_i = -(FL[i + 1] - FL[i]) / area[i]
        
        # assign
        dry_conv[i] = dry_conv_i
        lat_conv[i] = lat_conv_i
 
    return dry_conv, lat_conv


