# -*- coding: utf-8 -*-

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange

@njit(nogil = True, cache=False)
def ocean_fluxes(to, co, sbi, nbi, rhospc, oa, od, oddf, ow, rearthlat,
                 ww, u1, u2, dh, dz, do, nlat, nlatb, ADVF, HDIFFS,
                 HDIFFI, VDIFF, VADVF, ADVF_conv, HDIFFS_conv,
                 HDIFFI_conv, VADVF_conv, VDIFF_conv):
        
    
    '''
    Calculates meridional heat fluxes associated with ocean heat transport
    '''

    ADVF[:]=0.0; HDIFFS[:]=0.0; HDIFFI[:]=0.0
    VADVF[:]=0.0; VDIFF[:]=0.0
    ADVF_conv[:]=0.0; HDIFFS_conv[:]=0.0; HDIFFI_conv[:]=0.0
    VADVF_conv[:]=0.0; VDIFF_conv[:]=0.0
    
    for b in range(sbi+1, nbi):          # interior ocean faces, FULL-GRID index
    
        # --- advection (layers 0 and 5, co-split, upwind) ---
        # ----------------------------------------------------
        
        pref_top = ow[b]*od[0]*rhospc*u1[b]
        pref_bot = ow[b]*od[-1]*rhospc*u2[b]
        if b < co:                        # southern hemisphere
            ADVF[b] = pref_top * to[b]      # full-grid To index
            ADVF[5*nlatb + b] = pref_bot * to[5*nlat + (b-1)]
        elif b > co:                      # northern hemisphere
            ADVF[b] = pref_top * to[b-1]
            ADVF[5*nlatb + b] = pref_bot * to[5*nlat + b]
            
        # --- horizontal diffusion (all layers, gradient) ---
        # ---------------------------------------------------
        
        # surface layer 
        grad0 = -(to[b] - to[(b-1)]) / rearthlat
        HDIFFS[b] = ow[b]*od[0]*grad0*do*rhospc
        
        # interior layers
        for lyr in range(1, 6):
            base_to = lyr*nlat       # to: cells
            base_f  = lyr*nlatb      # HDIFFI: faces
            gradi = -(to[base_to + b] - to[base_to + (b-1)]) / rearthlat
            HDIFFI[base_f + b] = ow[b]*od[lyr]*gradi*dh*rhospc


    # --- COLUMN LOOP: horizontal convergence + vertical flux + vertical convergence ---
    
    for c in range(sbi, nbi):# for each column
             
       # convergence of horizontal advection (layers 0 and 5) ---
       ADVF_conv[c] = -(ADVF[c+1] - ADVF[c]) / oa[c]
       ADVF_conv[5*nlat + c] = -(ADVF[5*nlatb + (c+1)] - ADVF[5*nlatb + c]) / oa[c]
       
       # convergence of surface layer diffusion
       HDIFFS_conv[c] = -(HDIFFS[c+1] - HDIFFS[c]) / oa[c]
       
       for j in range(1, 6): # for each layer
       
           # convergence of interior layer diffusion
           base_c = j*nlat; base_f = j*nlatb     
           HDIFFI_conv[base_c + c] = -(HDIFFI[base_f + (c+1)] - HDIFFI[base_f + c]) / oa[c]
           
           # vertical advection
           if ww[c] > 0.0:
               VADVF[j*nlat + c] = oa[c]*ww[c]*rhospc * to[j*nlat + c]
           elif ww[c] < 0.0:
               VADVF[j*nlat + c] = oa[c]*ww[c]*rhospc * to[(j-1)*nlat + c]
               
           # vertical diffusion
           VDIFF[j*nlat + c] = oa[c]*(to[j*nlat+c]-to[(j-1)*nlat+c])/oddf[j-1]*dz*rhospc
            
           # vertical convergece of diffusion
           VDIFF_conv[(j-1)*nlat + c] = (VDIFF[(j)*nlat+c] - VDIFF[(j-1)*nlat+c]) / oa[c]
            
           # vertical convergence of advection
           VADVF_conv[(j-1)*nlat + c] = (VADVF[j*nlat+c] - VADVF[(j-1)*nlat+c]) / oa[c]
            
       # layer 5 convergence using bottom boundary interface 6 (=0, bottom boundary)
       VADVF_conv[5*nlat + c] = (0.0 - VADVF[5*nlat+c]) / oa[c]
       VDIFF_conv[5*nlat + c] = (0.0 - VDIFF[5*nlat+c]) / oa[c]
            
    return ADVF_conv, HDIFFS_conv, HDIFFI_conv, VADVF_conv, VDIFF_conv
   


@njit(nogil = True, cache=False)
def horizontal_ocean_velocities(olat, olatb, olatr, olatbr, dlatr, ocean_cntr, 
                                ocean_nbnd, ocean_sbnd, f, fb, r_earth, w0, odepth):
                                
    
    '''
    Calculates the meridional ocean velocities (in the top and bottom ocean
    layer) based on the prescribed vertical ocean velocities.
    '''
    
    w = w0.copy()
    
    # initialize meridional velocity arrays
    #--------------------------------------
    
    # velocity at surface layer
    u1 = np.zeros((olatb.size)) 
    
    # velocity at bottom layer
    u2 = np.zeros((olatb.size)) 
    
    # indexes for ocean boundaries
    #-----------------------------
    
    # index for centre of ocean circulation
    icoc = np.where(olatb == ocean_cntr)[0][0]
    
    # index for northern boundary
    inb = np.where(olatb == ocean_nbnd)[0][0]
    
    # index for southern boundary
    isb = np.where(olatb == ocean_sbnd)[0][0]
    
    
    # calculate upper velocities in the northern hemisphere
    #------------------------------------------------------
    
    for i in np.arange(icoc, inb-1):
        
        u1[i+1] = ((
            
                  (f[i]*np.cos(olatr[i])*r_earth*dlatr*w[i])
                  
                   +
                  
                   (fb[i]*np.cos(olatbr[i])*odepth[0]*u1[i])
            
                   )
            
                   /
                  
                   (fb[i]*np.cos(olatbr[i+1])*odepth[0])
                  
                   )
        
    # recalculate vertical velocity for last grid cell to conserve mass.
    w[inb-1] = -(
        
                (fb[i]*np.cos(olatbr[inb-1])*odepth[0]*u1[inb-1])
                
                /
                
                (f[i]*np.cos(olatr[inb-1])*r_earth*dlatr)
                
                )
                
                
    # calculate upper velocities in the southern hemisphere
    #------------------------------------------------------
        
    for i in np.arange(icoc, isb+1, -1):
        
        u1[i-1] = ((
            
            
                    -(f[i]*np.cos(olatr[i-1])*r_earth*dlatr*w[i-1])
                   
                    +
                   
                    (fb[i]*np.cos(olatbr[i])*odepth[0]*u1[i])
                   
                    )
            
                    /
                   
                    (fb[i]*np.cos(olatbr[i-1])*odepth[0])
                   
                    )
        
    # recalculate vertical velocity for last grid cell to conserve mass.
    w[isb] = (
        
                  (fb[i]*np.cos(olatbr[isb+1])*odepth[0]*u1[isb+1])
                 
                  /
                 
                  (f[i]*np.cos(olatr[isb])*r_earth*dlatr)
        
            )
    
        
    # calculate the bottom velocity 
    #------------------------------
    
    u2 = -u1*odepth[0]/odepth[5]    

    # return arrays
    #--------------
    
    return u1, u2, w
