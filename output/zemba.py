# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

ZEMBA model and sub-processes
"""

import numpy as np
from initialize_zemba import *
from numba import njit
from numba.typed import Dict
from numba.core import types
from utilities import *

@njit(nogil = True)
def mean_numba(a):
    
    'Returns average of numba array along 2nd axis...'

    res = []
    for i in prange(a.shape[0]):
        res.append(a[i, :].mean())

    return np.array(res)


#------------------------------------
# Energy balance model: sub-processes
#------------------------------------

@njit(nogil = True, cache=True)
def calculate_density(t):
    
    '''
    Calculates density of surface ocean water for 35 PSU using equation of state
    from Gill (1982) pp. 599-600
    '''
    # density of pure water
    pw = 999.842594 + 6.793952e-2*t - 9.095290e-3*t**2 + 1.001685e-4*t**3 - 1.120083e-6*t**4 + 6.536332e-9*t**5

    # density of water at 35 PSU
    s = 35
    pws = pw + s*(0.824493 - 4.0899e-3*t + 7.6438e-5*t**2 - 8.2467e-7*t**3 
              + 5.3875e-9*t**4) + s**(3/2) * (-5.72466e-3 + 1.0227e-4*t
                                              -1.6546e-6*t**2) + 4.8314e-4*s**2
                                                                                    
    return pws

@njit(nogil = True, cache=True)
def ocean_overturning_strength(to, lat, olat, oarea, dlat, idxcos, gamma, drn0,
                               drs0, zeta, tosN0, tosS0, w0, dz0, Var):
    
    '''
    Calculates changes in ocean overturning strength.
    '''
    
    # annual-mean surface ocean temperature
    tos = mean_numba(to)[Var["idxolyr1"].astype('i8')]

    # mean temperature from 60N to 80N.
    i60N = np.where(lat == 60. + dlat/2)[0][0]
    i80N = np.where(lat == 80. - dlat/2)[0][0]
    tosN = np.sum(tos[i60N:i80N+1]*oarea[i60N:i80N+1]) / np.sum(oarea[i60N:i80N+1])

    # mean temperature at equator (north of centre of ocean circulation)
    ieqN   = np.where(lat == idxcos + dlat/2)[0][0]
    toseqN = tos[ieqN]

    # mean temperature from 70S to 50S.
    i70S = np.where(lat == -70. + dlat/2)[0][0]
    i50S = np.where(lat == -50. - dlat/2)[0][0]
    tosS = np.sum(tos[i70S:i50S+1]*oarea[i70S:i50S+1]) / np.sum(oarea[i70S:i50S+1])

    # mean temperature at equator (south of centre of ocean circulation)
    ieqS = np.where(lat == idxcos - dlat/2)[0][0]
    toseqS = tos[ieqS]

    # initialize vertical ocean velocity
    ww     = np.zeros((olat.size))
    
    # initialize vertical ocean diffusion coefficient
    dz     = np.zeros((olat.size))
    
    # calculate density difference in the NH.
    rhonp  = calculate_density(tosN - 273.15)  # density at high latitudes
    rhoneq = calculate_density(toseqN- 273.15) # density at equator
    drn    = (rhonp - rhoneq) + gamma * (tosN - tosN0)
    
    # calculate density difference in the SH.
    rhosp  = calculate_density(tosS - 273.15)  # density at high latitudes
    rhoseq = calculate_density(toseqS- 273.15) # density at equator
    drs    = (rhosp - rhoseq) + gamma * (tosS - tosS0)
    
    # calculate change in overturning strength in the southern hemisphere
    i = np.where(olat==idxcos-dlat/2)[0][0]
    ww[0:i+1]  = w0[0:i+1]*(1.+zeta*(drs/drs0-1.))
    dz[0:i+1]  = dz0*(1.+zeta*(drs/drs0-1.))

    # calculate change in overturning strength in the northern hemisphere
    i = np.where(olat==idxcos+dlat/2)[0][0]
    ww[i:ww.size] = w0[i:ww.size]*(1.+zeta*(drn/drn0-1.))
    dz[i:ww.size] = dz0*(1.+zeta*(drn/drn0-1.))
    
    return ww, dz, drn, drs


@njit(nogil = True, cache=True)
def midpoint(to, lat, olat, oarea, dlat, idxcos, gamma, drn0,
                               drs0, zeta, ton0, tos0, w0, dz0, Var):
    
    '''
    Calculates changes in mid-point of ocean circulation
    '''
    
    # annual-mean surface ocean temperature
    tos = mean_numba(to)[Var["idxolyr1"].astype('i8')]

    # mean temperature from 60N to 80N.
    i60N = np.where(lat == 60. + dlat/2)[0][0]
    i80N = np.where(lat == 80. - dlat/2)[0][0]
    tosN = np.sum(tos[i60N:i80N+1]*oarea[i60N:i80N+1]) / np.sum(oarea[i60N:i80N+1])

    # mean temperature at equator (north of centre of ocean circulation)
    ieqN   = np.where(lat == idxcos + dlat/2)[0][0]
    toseqN = tos[ieqN]

    # mean temperature from 70S to 50S.
    i70S = np.where(lat == -70. + dlat/2)[0][0]
    i50S = np.where(lat == -50. - dlat/2)[0][0]
    tosS = np.sum(tos[i70S:i50S+1]*oarea[i70S:i50S+1]) / np.sum(oarea[i70S:i50S+1])

    # mean temperature at equator (south of centre of ocean circulation)
    ieqS = np.where(lat == idxcos - dlat/2)[0][0]
    toseqS = tos[ieqS]

    # density difference in the NH
    drn = 11.2003*(tosN-toseqN)-.034805*(tosN**2-toseqN**2)+3.46761e-5*(tosN**3-toseqN**3)+gamma*(tosN-ton0)

    # density difference in the SH
    drs = 11.2003*(tosS-toseqS)-.034805*(tosS**2-toseqS**2)+3.46761e-5*(tosS**3-toseqS**3)+gamma*(tosS-tos0)
    
    if (drn - drs < 0.51):
        
        idxcos = np.array([-15.])
        
    if (drn - drs >= 0.51) & (drn - drs <= 0.55):
        
        idxcos = np.array([-10.])
        
    if (drn - drs > 0.55):
        
        idxcos = np.array([-5.])
    
    return idxcos

@njit(nogil = True, cache=True)
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
            
                  (f[i]*np.cos(olatr[i])*r_earth[0]*dlatr[0]*w[i])
                  
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
                
                (f[i]*np.cos(olatr[inb-1])*r_earth[0]*dlatr[0])
                
                )
                
                
    # calculate upper velocities in the southern hemisphere
    #------------------------------------------------------
        
    for i in np.arange(icoc, isb+1, -1):
        
        u1[i-1] = ((
            
            
                    -(f[i]*np.cos(olatr[i-1])*r_earth[0]*dlatr[0]*w[i-1])
                   
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
                 
                  (f[i]*np.cos(olatr[isb])*r_earth[0]*dlatr[0])
        
            )
    
        
    # calculate the bottom velocity 
    #------------------------------
    
    u2 = -u1*odepth[0]/odepth[5]    

    # return arrays
    #--------------
    
    return u1, u2, w


@njit(nogil = True, cache=True)
def LWR_param(sigma, CO2, taz, tsz, tatot, z, cloud_cover, cloud_emissivity, GHG_amp):
    
    
    '''
    Calculates the longwave radiative fluxes using the radiation parameterization
    of Bintanja (1996).
    '''
    
    # pre-industrial surface air temperature
    #---------------------------------------
    
    tamodern = 287.8 # for adjustment of clear sky upwards radiation
                     # at the top of the atmosphere (Stap et al., 2014)
                        
    
    # standard longwave fluxes
    #-------------------------
    
    # standard longwave fluxes at the bottom of the atmosphere (BOA) and the 
    # top of the atmosphere (TOA) from Bintanja (1996)
    
    # Clear sky (nc)
    st_nc_d = 266.6 # BOA downwards
    st_nc_u = 245.7 # TOA upwards
    
    # Overcast (cl)
    st_cl_d = 330.8 # BOA downwards
    st_cl_u = 215.7 # TOA upwards
    
    # Modify air and surface temperature for elevation
    #-------------------------------------------------
    
    ta = taz  - 0.0065 * z # air
    ts = tsz  - 0.0065 * z # surface
    
    # ta: Devitation in longwave radiation from standard value due to changing
    #-------------------------------------------------------------------------
    # near-surface air temperatures (K)
    #----------------------------------
    
    # Clear sky (nc)
    ta_nc_d = -851.1 + 10.622 * ta - 0.06435 * (ta**2) + 1.328e-4 * (ta**3)
    ta_nc_u = -268.0 - .0107 * ta + 0.0004903 *(ta**2) + 1.0204e-5 * (ta**3) - 1. * ((tatot - tamodern)**1.) 
 
    # Overcast (cl)
    ta_cl_d = -533.2 + 3.8795 * ta - 0.02621 * (ta**2) + 6.8068e-5 * (ta**3)
    ta_cl_u = -282.2 + 1.1775 * ta - 8.09e-3 * (ta**2) + 2.6393e-5 * (ta**3)
    
    
    # CO2: Devitation in longwave radiation from standard value due to changing
    #--------------------------------------------------------------------------
    # CO2 concentrations (kg/kg)
    #---------------------------
    
    # ref CO2: 5.32e-4 (kg/kg)
    
    # greenhouse gas amplification factor: radiative forcing per CO2 doubling
    # multiplied by 1.3 to account for non-CO2 GHGs
    GHG_amplification = (GHG_amp[0] * 3.7)/3.5
    
    # natural log of CO2 conc.
    ln_co2 = np.log(CO2) 
    
    # Clear sky (nc)
    co2_nc_d = (3.7592 + 0.4950 * ln_co2) * 3. * GHG_amplification 
    co2_nc_u = (-14.77 - 1.9589 * ln_co2) * 3. * GHG_amplification 
    
    # Overcast (cl)
    co2_cl_d = (1.3064 + 0.1673 * ln_co2) * 3. * GHG_amplification 
    co2_cl_u = (-11.198 - 1.488 * ln_co2) * 3. * GHG_amplification 
    
    
    # CTA: Devitation in longwave radiation from standard value due to changing
    #--------------------------------------------------------------------------
    # cloud top altitude
    #-------------------
    
    #!!!!!! Not included !!!!!!!#
    
    # cta_cl_u = 40.3 - 5.474 * cta - 0.518  * (cta**2)
    # cta_cl_d = 28.2 - 6.376 * cta + 0.1469 * (cta**2)
    
    
    # Emissivity of clouds
    #---------------------
    
    # Emissivity of clouds assumed to be constant at value of 1.
    
    # Overcast (cl) 
    cem_cl_d = -54.1 + 54.1 * cloud_emissivity 
    cem_cl_u = 79.9 - 79.9 * cloud_emissivity 
        
    
    # Surface longwave radiation 
    #---------------------------
    
    # Emitted upwards longwave radiation at BOA based on the surface 
    # temperature.
    
    lwbu = sigma * (ts**4) # Emissivity of the surface assumed to be 1.
    
    
    # BOA downward longwave radiation
    #--------------------------------
    
    # Clear sky 
    lw_nc_d = st_nc_d + ta_nc_d + co2_nc_d 
    
    # Overcast
    lw_cl_d = st_cl_d + ta_cl_d + co2_cl_d + cem_cl_d #+ cta_cl_d
    
    # Weighted average by cloud cover
    lwbd = (1 - cloud_cover) * lw_nc_d + cloud_cover * lw_cl_d
    
   
    # TOA upward longwave radiation
    #------------------------------
    
    # Clear sky
    lw_nc_u = st_nc_u + ta_nc_u + co2_nc_u
    
    # Overcast
    lw_cl_u = st_cl_u + ta_cl_u + co2_cl_u + cem_cl_u# + cta_cl_u
    
    # Weighted average by cloud cover
    rlut = (1 - cloud_cover) * lw_nc_u + cloud_cover * lw_cl_u
    
    
    # Return TOA and surface fluxes
    #------------------------------

    # Return BOA upwards, BOA downwards and TOA upwards longwave fluxes.
    
    return lwbu, lwbd, rlut 


@njit(nogil = True, cache=True)
def SWR_param(cloud_cover, ta, alpha_nc, alpha_cl, u, z, s, tau):
    
    '''
    Calculates the shortwave radiative fluxes using the radiation parameterization
    of Bintanja (1996).
    '''
    
    # ta = ta  - 0.0065 * z
    
    # Transmission as function of ozone
    #----------------------------------
    
    oz = 0.96586
    
    # Transmission as function of rayleigh scattering
    #------------------------------------------------
    
    ra = 0.92648
    
    # Transmission as function of near-surface air temperature (water vapour)
    #------------------------------------------------------------------------
    
    ta_d = -0.20518 + 0.016014 * ta - 6.6733e-5 * (ta**2) + 8.30352e-8 * (ta**3)
    i = np.where(ta < 180)
    ta_d[i] = 1
            
    ta_u = -0.29673 + 0.02004 * ta  -8.5214e-5 * (ta**2) + 1.08069e-7 * (ta**3)
    i = np.where(ta < 180)
    ta_u[i] = 1.1807

    
    # Transmission as function of albedo
    #-----------------------------------
    
    # Clear sky (nc)
    alpha_nc_d = 0.9891 + 0.0512 * alpha_nc  + 0.0124 * (alpha_nc**2)
    alpha_nc_u = 0.3351 + 3.2821 * alpha_nc + 0.1969 * (alpha_nc**2)
    
    # Overcast (cl)
    alpha_cl_d = 0.9891 + 0.0512 * alpha_cl  + 0.0124 * (alpha_cl**2)
    alpha_cl_u = 0.8971 + 0.4316 * alpha_cl + 0.3643 * (alpha_cl**2)
    
    
    # Transmission as function of zenith angle
    #-----------------------------------------
    
    # Clear sky (nc)
    zn_nc_d = 0.6135 + 1.5377 * u - 1.9297 * (u**2) + 0.8602 * (u**3)
    zn_nc_u = 1.7286 - 2.8755 * u + 5.2534 * (u**2) - 4.7598 * (u**3) + 1.6551 * (u**4)
    
    
    # Overcast (cl)
    zn_cl_d = 0.500 + 1.1071 * u - 0.2088 * (u**2)
    zn_cl_u = 1.7501 - 0.08207 * u - 1.5343 * (u**2) + 0.8695 * (u**3)
    

    # Transmission as function of optical depth
    #------------------------------------------
    
    # cloud optical depth
    tau = np.log(tau)
    
    # downwards
    tau_d = 0.8447 - 0.1664 * tau - 0.0113 * (tau**2)
    delta_tau_d = -0.0503 - 0.02910 * tau - 0.0331 * (tau**2) + 0.0098 * (tau**3) 
    tau_alpha_d = tau_d + ((alpha_cl - 0.2) * (delta_tau_d))/0.5
   
    # Upward
    tau_u = 0.6422 + 0.1385 * tau + 0.0708 * (tau**2) - 0.0114 * (tau**3) 
    delta_tau_u = 0.32048 - 0.11872 * tau - 0.07024 * (tau**2) + 0.0123 * (tau**3)
    tau_alpha_u = tau_u + ((alpha_cl - 0.2)/0.6) * delta_tau_u
    
    
    # Total transmission
    #-------------------
    
    # Clear sky (nc)
    T_nc = oz * ra * ta_d * alpha_nc_d * zn_nc_d + 0.000022 * z
    
    # Overcast (cl)
    T_cl = oz * ra * ta_d * alpha_cl_d * zn_cl_d * tau_alpha_d * 0.628 + 0.000022 * z

    

    # Upwards solar radiation at TOA
    #-------------------------------
    
    # Clear sky (nc)
    swt_nc_u = 0.18744 * s * ta_u * alpha_nc_u * zn_nc_u
    
    # Overcast (cl)
    swt_cl_u = 0.32444 * s * ta_u * alpha_cl_u * zn_cl_u * tau_alpha_u
    
    # Weighted average
    swt_u = swt_nc_u * (1 - cloud_cover) + swt_cl_u * cloud_cover


    # Downward solar radiation at surface
    #------------------------------------
    
    # Clear sky (nc)
    swb_nc_d = s * T_nc
    
    # Overcast (cl)
    swb_cl_d = s * T_cl
    
    # Weighted average
    swb_d = (1 - cloud_cover) * swb_nc_d + cloud_cover * swb_cl_d
    

    # Upwards solar radiation at surface
    #-----------------------------------
    
    # Clear sky (nc)
    swb_nc_u = alpha_nc * swb_nc_d
    
    # Overcast (cl)
    swb_cl_u = alpha_cl * swb_cl_d
    
    # Weighted average
    swb_u = swb_nc_u * (1 - cloud_cover) + swb_cl_u * cloud_cover
    
    
    # Return the weighted average
    #----------------------------
    
    return swt_u, swb_d, swb_u 


@njit(nogil = True, cache=True)
def surface_albedo_land_v3(lat, tl, lr, land_height, alphag, alphai, alphas, alphaws,
                           ice_fraction, snow_fraction, cza, mask):
    
    '''
    Calculates the land surface albedo when the hydrological cycle is turned on.
    '''
    
    # lapse-rate modified land surface temperature
    #---------------------------------------------
    
    tlz = tl - (lr*land_height)
    
    #------------
    # Snow albedo
    #------------
    
    # initialise 
    snow_albedo_ground  = np.zeros((lat.size)) # snow albedo over bare ground
    snow_albedo_landice = np.zeros((lat.size)) # snow albedo over land-ice
    
    # temperature-dependant snow albedo over bare ground
    #---------------------------------------------------
    
    # cold snow
    j1=np.where((tlz<263.)) 
    snow_albedo_ground[j1] = alphas
    
    # mixed snow
    j2 = np.where(((tlz >= 263) & (tlz <= 273.))) 
    snow_albedo_ground[j2] = alphas + (alphaws -alphas) * ( (tlz[j2]-263.)/10)
    
    # warm snow
    j3 = np.where((tlz > 273.))
    snow_albedo_ground[j3] = alphaws
    
    # temperature-dependant snow albedo over land-ice
    #------------------------------------------------
    
    # cold snow
    j1=np.where((tlz<263.)) 
    snow_albedo_landice[j1] = alphas
    
    # mixed snow
    j2 = np.where(((tlz >= 263) & (tlz <= 273.))) 
    snow_albedo_landice[j2] = alphas + (alphaws -alphas) * ( (tlz[j2]-263.)/10)
    
    # warm snow
    j3 = np.where((tlz > 273.))
    snow_albedo_landice[j3] = alphaws
    
    
    # modify snow albedo for high zenith angle
    #-----------------------------------------
    
    '''
    Eq.5&6 of Lefebre et al., (2003): Modelling of snow and ice melt at ETH camp (West Greenalnd): A study of surface albedo
    '''
    
    # solar zenith angle (max = 80)
    cosz = np.minimum(cza, np.cos(np.deg2rad(80)))
    
    # where solar zenith angle > 60
    i = np.where(cza < np.cos(np.deg2rad(60)))
    
    # modify the snow albedo
    snow_albedo_ground[i]  = snow_albedo_ground[i]  + np.maximum(0, 0.032 * 1/2 * (3/(1+4*cosz[i])-1))
    snow_albedo_landice[i] = snow_albedo_landice[i] + np.maximum(0, 0.032 * 1/2 * (3/(1+4*cosz[i])-1))
    
    #----------------------------
    # Albedo of each surface type
    #----------------------------
    
    # average albedo over land-ice
    #-----------------------------
    
    albedo_landice =  (    # snow-free albedo               snow-covered albedo
                       ( alphai*(1-snow_fraction) ) + (snow_albedo_landice*snow_fraction )
                      )
    
    # average albedo over bare-ground
    #--------------------------------
    
    albedo_ground  =  (    # snow-free albedo             snow-covered albedo
                       ( alphag*(1-snow_fraction) ) + (snow_albedo_ground*snow_fraction)
                     )
    
    # average albedo for latitude band
    #---------------------------------
    
    albedo =     (     #   albedo of ground                albedo of land-ice
                   ( albedo_ground*(1-ice_fraction) ) + (albedo_landice*ice_fraction )
                 
                 ) * mask
    
    return albedo


@njit(nogil = True, cache=True)
def surface_albedo_land(lat, snow_thick, melt, alphag, alphai, alphas, alphaws,
                        ice_fraction, snow_fraction, cza, mask, snow):
    
    '''
    Calculates the surface albedo over land when the hydrological cycle is 
    turned on.
    '''
    
    # determine the snow albedo 
    #--------------------------
    
    # depends on whether there has been melting in the previous time-step.
    
    # initialize array
    snow_albedo = np.zeros((lat.size)) 
    
    # dry-snow albedo (no melting)  
    i = np.where(melt <= 0) 
    snow_albedo[i] = alphas 
    
    # wet-snow albedo (melting)
    i = np.where(melt > 0)   
    snow_albedo[i] = alphaws  
    

    # determine the albedo over ice and bare ground
    #----------------------------------------------
    
    # depends on the snowpack thickness.
    
    # critical snow depth (beyond which surface albedo = max snow albedo)
    dcrit = 0.1
    
    # albedo over ice
    
    if snow == 1: # uniform snow cover
        
        ice_albedo = alphai + snow_thick/dcrit * (snow_albedo - alphai)
        i = np.where(ice_albedo > snow_albedo) # where albedo exceeds max albedo
        ice_albedo[i] = snow_albedo[i]         # ... set to max albedo.
        
    if snow == 2: # partial snow cover
        
        # albedo over snow-covered ice
        snow_albedo_ice = alphai + snow_thick/dcrit * (snow_albedo - alphai)
        i = np.where(snow_albedo_ice > snow_albedo) # where albedo exceeds max albedo:
        snow_albedo_ice[i] = snow_albedo[i]         # ... set to max albedo.

        # average albedo over ice
        ice_albedo = snow_fraction * snow_albedo_ice + (1 - snow_fraction) * alphai
    
    
    # albedo over bare ground
    
    if snow == 1: # uniform snow cover
    
        ground_albedo = alphag + snow_thick/dcrit * (snow_albedo - alphag)
        i = np.where(ground_albedo > snow_albedo) # where albedo exceeds max albedo:
        ground_albedo[i] = snow_albedo[i]         # ... set to max albedo.
        
    if snow == 2: # partial snow cover
        
        # albedo over snow-covered ground
        snow_albedo_ground = alphag + snow_thick/dcrit * (snow_albedo - alphag)
        i = np.where(snow_albedo_ground > snow_albedo) # where albedo exceeds max albedo:
        snow_albedo_ground[i] = snow_albedo[i]         # ... set to max albedo.
        
        # average albedo over ice
        ground_albedo = snow_fraction * snow_albedo_ground + (1 - snow_fraction) * alphag
        
    
    # determine the average surface albedo
    #-------------------------------------
    
    # weighted mean of ice albedo and bare ground albedo, dependant on the ice 
    # fraction for a given latitude
    
    albedo = (1 - ice_fraction) * ground_albedo + ice_fraction * ice_albedo
    
    
    # modify albedo for high zenith angles
    #-------------------------------------
    
    # albedo value corrected by taking into account the solar zenith angle
    # dependancy- following equation 5 & 6 of Lefebre et al., (2003)
    
    cosz = np.minimum(cza, np.cos(np.deg2rad(80)))
    
    # where the zenith angle is greater than 60
    i = np.where(cza < np.cos(np.deg2rad(60)))
    
    # modify the albedo
    albedo[i] = albedo[i] + np.maximum(0, 0.032 * 1/2 * (3/(1+4*cosz[i])-1))
    

    # Mask
    #-----
    
    albedo = albedo * mask
    
    return albedo


@njit(nogil = True, cache=True)
def surface_albedo_land_v2(ebm_lat, land_height, lr, alphag, Tal, Tl,
                           alphas, alphaws, alphai, cza, land_mask,
                           ice_fraction):
    
    """
    Calculates the surface albedo over land when hydrological cycle is turned 
    off.
    """
    
    # snow fraction over each surface type (evenly distributed)
    #----------------------------------------------------------
    
    # initialize
    snow_fraction = np.zeros((ebm_lat.size))
    
    # elevation modified surface air temperature
    Talz = Tal - (lr*land_height)
    
    # completelty snow covered band
    i1 = np.where((Talz < 260.))
    snow_fraction[i1] = 1.
    
    # partially snow covered band
    i2 = np.where(((Talz >= 260.) & (Talz <= 280.)))
    snow_fraction[i2] = 0.05 * (280 - Talz[i2])
    
    # no snow covered band
    i3 = np.where((Talz > 280.))
    snow_fraction[i3] = 0.
    
    # snow albedo over each surface type
    #-----------------------------------
    
    # initialize
    alpha_snow = np.zeros((ebm_lat.size))
    
    # elevation modified surface temperature
    Tlz = Tl - (lr*land_height)
    
    # completely cold covered snow
    j1 = np.where((Tlz < 263.))
    alpha_snow[j1] = alphas
    
    # mixed snow albedo
    j2 = np.where(((Tlz >= 263) & (Tlz <= 273.)))
    alpha_snow[j2] = alphas + (alphaws -alphas) * ( (Tlz[j2]-263.)/10)
    
    # completely warm covered snow
    j3 = np.where((Tlz > 273.))
    alpha_snow[j3[0]] = alphaws
    
    # modify snow albedo for zenith angle
    #------------------------------------
    
    # Albedo value corrected by taking into account the solar zenith angle
    # dependancy- follwing equation 5 & 6 of Lefebre et al., (2003): 
    # Modelling od snow and ice melt at ETH camp (West Greenalnd): A study
    # of surface albedo
    cosz = np.minimum(cza, np.cos(np.deg2rad(80)))
    
    # where the zenith angle is greater than 60
    i = np.where(cza < np.cos(np.deg2rad(60)))
    
    # modify the albedo
    alpha_snow[i] = alpha_snow[i] + np.maximum(0, 0.032 * 1/2 * (3/(1+4*cosz[i])-1))
    
    
    # albedo over ice surfaces
    #-------------------------
    alpha_ice = (
                  # snow-free albedo
                 ( alphai * (1 - snow_fraction) )
                 
                 + 
                 # snow-covered albedo
                 ( alpha_snow * snow_fraction )
                 
                 )
    
    
    
    # albedo over bare ground surfaces
    #---------------------------------
    alpha_ground = (
                  # snow-free albedo
                 ( alphag * (1 - snow_fraction) )
                 
                 + 
                 # snow-covered albedo
                 ( alpha_snow * snow_fraction )
                 
                 )
    
    # total land albedo (area-weighted for ice fraction)
    #---------------------------------------------------
    
    alpha_land =  (
                  # bare ground albedo
                 ( alpha_ground * (1 - ice_fraction) )
                 
                 + 
                 # ice albedo
                 ( alpha_ice * ice_fraction )
                 
                 )
    
    # mask
    #----
    alpha_land = alpha_land * land_mask
    
    
    return alpha_land, snow_fraction
    

@njit(nogil = True, cache=True)
def surface_albedo_ocean(ebm_lat, si_fraction, si_melt_flux, alpha_si_min,
                         alpha_si_max, cza, ocean_mask):
    
    
    """
    Calculates the surface albedo over ocean
    """
                         
    # Sea-ice albedo
    #---------------
    
    si_albedo = np.zeros(ebm_lat.size)
    
    i = np.where((si_fraction > 0) & (si_melt_flux > 0))
    
    si_albedo[i] = alpha_si_min
    
    j = np.where((si_fraction > 0) & (si_melt_flux <= 0))
    
    si_albedo[j] = alpha_si_max
        
 
    # Modify albedo for high zenith angles
    #-------------------------------------
 
    # Albedo value corrected by taking into account the solar zenith angle
    # dependancy- follwing equation 5 & 6 of Lefebre et al., (2003)
    cosz = np.minimum(cza, np.cos(np.deg2rad(80)))
    
    # # where the zenith angle is greater than 60
    k = np.where((si_fraction > 0) & (cza < np.cos(np.deg2rad(60))))
    
    # # modify the albedo
    si_albedo[k] = si_albedo[k] + np.maximum(0, 0.032 * 1/2 * (3/(1+4*cosz[k])-1))
    
    
    # open ocean albedo
    #------------------  
  
    # Calculates the broadband surface ocean albedo as a function of the 
    # solar zenith angle, based on the parameterization of Taylor et al.,
    # (1996)
     
    o_albedo = 0.037 / ( (1.1 * cza**1.4) + 0.15)
     
    # weighted mean ocean albedo
    #---------------------------
    
    albedo = (si_albedo * si_fraction) + (
         o_albedo * (1 - si_fraction))
    
    # mask
    #------
    
    albedo = albedo*ocean_mask
     
    return albedo


@njit(nogil = True, cache=True)
def snow_edge(x, Var):
    
    '''
    Returns the equatorward edge of snow or sea ice extent in both NH and SH
    '''
    
    # northern hemisphere (NH)
    #-------------------------

    # slice for NH
    x_nh   = x[Var["idxnh"].astype('i8')]
    lat_nh = Var["lat"][Var["idxnh"].astype('i8')]
    
    # check if there is any snow...
    i = np.where(x_nh > 0.)[0]
    
    # if no snow
    if i.any()==False:
        latidx_nh = np.NaN
    
    # if snow
    elif i.any()==True:
        
        # find snow edge
        idx_nh = np.argmax(x_nh > 0.)
        
        # find lat of snow edge
        latidx_nh = lat_nh[idx_nh]

    
    # southern hemisphere (SH)
    #-------------------------

    # slice for SH
    x_sh   = x[Var["idxsh"].astype('i8')]
    lat_sh = Var["lat"][Var["idxsh"].astype('i8')]
    
    # reverse order
    x_sh   = np.flip(x_sh )
    lat_sh = np.flip(lat_sh )
    
    # check if there is any snow...
    i = np.where(x_sh > 0.)[0]
    
    # if no snow
    if i.any()==False:
        latidx_sh = np.NaN
    
    # if snow
    elif i.any()==True:
        
        # find snow edge
        idx_sh = np.argmax(x_sh > 0.)
        
        # find lat of snow edge
        latidx_sh = lat_sh[idx_sh]

    
    # return 
    #-------
    
    return np.array([latidx_nh]), np.array([latidx_sh])
   
        
@njit(nogil = True, cache=True)
def surface_pressure(h):
    
    """
    Calculates the surface pressure as it varies with height (in Pa)
    """
    
    # define constants 
    #-----------------
    
    temp_r = 288.15       # reference temperature (K)
    R      = 8.314        # universal gas constant (J/(mol.K))
    M      = 0.029        # molar mass of Earth's air (kg/mol)
    p_0    = 101325       # mean atmopsheric surface pressure (Pa) 
    g      = 9.81         # gravitational acceleration (m s^-2)
    
    
    # calculate pressure at new altitude 
    #-----------------------------------

    p = p_0  * ((temp_r + h * -0.0065 )/(temp_r)) **((-g * M)/(R*-0.0065))
    
    return p 


@njit(nogil = True, cache=True)
def saturation_specific_humidity(Lv, T, h):
    
    '''
    Uses a given temperature and pressure to calculate the saturation
    specific humidity-- the mass of water vapor per mass of moist air, 
    expressed as kg/kg. 
    '''

    # define some constants
    #----------------------
    
    e0  = 610.78  # Reference vapor pressure for CC equation (Pa)
    Rv  = 461.52  # Specific gas constant of water vapor (J kg^-1 K^-1)
    Rd  = 287     # Specific gas constant of dry air (J kg^-1 K^-1)
    T_0 = 273.16  # Reference temperature for CC equation (K)
    

    # calculate the saturation vapor pressure (e_s) from the Clasius -
    #-----------------------------------------------------------------
    # Clapeyron relation (in Pa)
    #---------------------------
    
    e_s = e0 * np.exp(-(Lv/Rv) * ((1/T) - (1/T_0)))
    
    
    # calculate the surface pressure as it changes with height
    #---------------------------------------------------------
    p = surface_pressure(h)
    

    # calculate the saturation specific humidity, q_s, in kg/kg.
    #-----------------------------------------------------------
    
    q_s =  (Rd / Rv ) * (e_s / p)

    
    return q_s 


@njit(nogil = True, cache=True)
def surface_evaporation(ts, ta, qa, surface_height, Lv, r, atm_rho, lhfr, w, hydro, height):
    
    '''
    Calculates the surface evaporation flux (in kg m^-2 s^-1), rate (in m/yr) 
    and associated heat flux (in W m^-2) for land or ocean, depending on 
    the "surface_type" keyword. Evaporation flux calculated using the 
    traditional bulk aerodynamical formula. Positive is upwards.
    '''
    

    # Saturation specific humidity of the surface
    #--------------------------------------------
    
    if height == 0:
    
        # lapse-rate modified surface temperature 
        tsz = ts
        
        # saturation specific humidity of surface
        qss = saturation_specific_humidity(Lv, tsz, 0.)
    
    if height == 1:
    
        # lapse-rate modified surface temperature 
        tsz = ts - (0.0065 * surface_height) 
        
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
            
        if height == 1:
            
            # lapse-rate modified surface air temperature   
            taz = ta - (0.0065 * surface_height) 
            
            # saturation specific humidity of near-surface air
            qsa = saturation_specific_humidity(Lv, taz, surface_height)
        
        # specific humidity of near-surface air
        q = qsa * (r/100)
        
    if hydro==1: # if hydrological cycle is turned on
    
        # specific humidity of near-surface air (as determined by hydrological cycle)
        q = qa 
        

    # upwards evaporative flux (kg m^-2 s^-1)
    #----------------------------------------
    
    evaporation_flux = atm_rho * lhfr * w * (qss - q) 
    

    # upwards latent heat flux loss (W m^-2)
    #---------------------------------------
    
    evaporation_lhf = evaporation_flux * Lv
    
    
    return evaporation_flux, evaporation_lhf



@njit(nogil = True, cache=True)
def precipitation(lat, ta, qa, surface_height, Lv, r, atm_rho, atm_depth, secs_in_day, 
                  mask, water_rho, height):
    
    '''
    Calculates the precipitation flux (in kg m^-2 s^-1), rate (in m/yr)
    and associated heat flux (in W m^-2). 
    '''

    # Initialize the precipitation flux array
    #----------------------------------------
    
    precipitation_flux = np.zeros((lat.size))
    
    # Saturation specific humidity of the atmosphere
    #-----------------------------------------------
    
    if height == 0:
        
        # surface air temperature 
        taz = ta
        
        # saturation specific humidity of near-surface air
        qsa = saturation_specific_humidity(Lv, taz, 0.)
        
    if height == 1:
        
        # surface air temperature 
        taz = ta - (0.0065 * surface_height)
        
        # saturation specific humidity of near-surface air
        qsa = saturation_specific_humidity(Lv, taz, surface_height)
    

    # Relative humidity of near-surface air
    #--------------------------------------
    
    rs = (qa / qsa) * 100
    

    # Precipitation flux (kg / m^-2 s^-1)
    #------------------------------------
    
    # relative humidity exceeding the maximum relative humidity 
    i = np.where(rs > r)
    
    # precipitation flux
    precipitation_flux[i] = (atm_rho * atm_depth) / (3*secs_in_day) * (qa[i] - ( (r/100) * qsa[i]) )
    qa[i] = qa[i] - precipitation_flux[i]/(atm_rho*atm_depth)*secs_in_day
    
    
    # modified near-surface humidity
    # qa[i] = (r/100) * qsa[i]
    
    # mask
    precipitation_flux = precipitation_flux*mask
    
    # Precipitation rate (m/day)
    #---------------------------
    
    precipitation_rate = (precipitation_flux / water_rho) * secs_in_day
    
    # Latent heat flux of precipitation (W m^-2)
    #-------------------------------------------
    
    precipitation_heat_flux = precipitation_flux * Lv
    
    # Return precipitation arrays
    #----------------------------

    return precipitation_flux, precipitation_rate, precipitation_heat_flux, qa



@njit(nogil = True, cache=True)
def precipitation_v2(lat, ta, qa, surface_height, Lv, r, atm_rho, atm_depth, secs_in_day, 
                  mask, water_rho, height):
    
    '''
    Calculates the precipitation flux (in kg m^-2 s^-1), rate (in m/yr)
    and associated heat flux (in W m^-2). 
    '''

    # Initialize the precipitation flux array
    #----------------------------------------
    
    precipitation_flux  = np.zeros((lat.size))
    precipitation_extra = np.zeros((lat.size))
    
    # Saturation specific humidity of the atmosphere
    #-----------------------------------------------
    
    if height == 0:
        
        # surface air temperature 
        taz = ta
        
        # saturation specific humidity of near-surface air
        qsa = saturation_specific_humidity(Lv, taz, 0.)
        
    if height == 1:
        
        # surface air temperature 
        taz = ta - (0.0065 * surface_height)
        
        # saturation specific humidity of near-surface air
        qsa = saturation_specific_humidity(Lv, taz, surface_height)
    

    # Relative humidity of near-surface air
    #--------------------------------------
    
    rs = (qa / qsa) * 100
    

    # Precipitation flux (kg / m^-2 s^-1)
    #------------------------------------
    
    precipitation_flux=(atm_rho*atm_depth) / (5*secs_in_day) * (qa)
    
    # relative humidity exceeding the maximum relative humidity 
    i = np.where(rs > r)
    precipitation_extra[i] = (atm_rho*atm_depth) / secs_in_day * (qa[i] - ( (r/100) * qsa[i]) )
    qa[i] = (r/100) * qsa[i]
    
    # mask
    precipitation_flux = precipitation_flux*mask
    
    # Precipitation rate (m/day)
    #---------------------------
    
    precipitation_rate = (precipitation_flux / water_rho) * secs_in_day
    
    # Latent heat flux of precipitation (W m^-2)
    #-------------------------------------------
    
    precipitation_heat_flux = precipitation_flux * Lv
    
    # Return precipitation arrays
    #----------------------------

    return precipitation_flux, precipitation_rate, precipitation_heat_flux, qa

    

@njit(nogil = True, cache=True)
def sensible_heat_flux(ts, tas, atm_sphc, atm_rho, shfr, mask):
    
    '''
    Determines the upwards sensible heat flux using the traditional bulk 
    parameterization.
    '''
    
    # sensible heat flux
    #-------------------
    
    shf = atm_sphc * atm_rho * shfr * (ts - tas)
    
    # mask
    #-----
    shf = shf*mask
    
    return shf

@njit(nogil = True, cache=True)
def dry_heat_transport(Ta, mean_height, ebm_diff, s_width, atm_hc,
                              D_T, r_earth, s_area):
    
    # Calculates the northward heat transport (in J/s) and heat convergence
    # (in W m^-2) for each latitudinal band, based on the parameterization
    # of Bintanja (1997). 
    
    #--------------------------------------------------------------------------
    # Step 1: Calculate the atmospheric temperature gradient in radians and
    # sets heat transport to zero at the poles. 
    #--------------------------------------------------------------------------
    
    taz = Ta - (0.0065 * 
                         mean_height) # Modity atm temp for lapse rate
    
    T_grad = np.diff(taz)/ebm_diff
    T_grads = np.concatenate((np.zeros(1),T_grad,np.zeros(1)))
    
    #--------------------------------------------------------------------------
    # Step 2: Calculate the the northward heat transport (in J/s)
    #--------------------------------------------------------------------------
    
    ntransport = -s_width * atm_hc * D_T  * (T_grads/r_earth)
    
    #--------------------------------------------------------------------------
    # Step 3: Calculate the area of each latitudinal band 
    #--------------------------------------------------------------------------
    
    area = np.copy(s_area)
    
    #--------------------------------------------------------------------------
    # Step 4: Calculate the heat convergence (in W m^-2) of each latitudinal band 
    #--------------------------------------------------------------------------
    
    convergence = -np.diff(ntransport)/area
    
    #--------------------------------------------------------------------------
    # Step 5: Return northward heat transport and convergence arrays 
    #--------------------------------------------------------------------------
    
    return ntransport, convergence




@njit(nogil = True, cache=True)
def moist_heat_transport_v5(Var, State, INPUT, hadley_cell, hydro, height=0):
        
    
    '''
    Atmospheric heat transport represented as down-gradient diffusion of 
    near-surface moist static energy.
    
    Option to include Hadley cell parameterization to simulate latent and 
    dry static energy transport (from Siler et al., 2018)...

    '''
    
    # declare variables
    #------------------
    
    T=State['Ta'] 
    cp=Var["atm_sphc"][0]
    rho=Var['atm_rho'][0]
    dpth=Var['atm_depth'][0]
    
    lv=Var['Lv'][0]
    r=INPUT['r'][0]
    qa=State['Q']
    
    lat=Var["lat"]
    latb=Var["latb"]
    latr=Var["latr"]
    latbr=Var["latbr"]
    EQ=int(Var['idxeq'][0])
    dlatr=Var['dlatr'][0]
    sz=Var['latb'].size
    area=Var['sarea']
      
    gms_factor=INPUT["hadley_constant"][0]
    D=INPUT['dt']
    
    # moist static energy
    #--------------------
    
    # dry static energy
    ms=cp*T
    
    # latent energy
    if hydro==0:
        q=saturation_specific_humidity(lv,T,np.zeros((lat.size) ))*(r/100)
    if hydro==1:
        q=qa.copy()
    ml=lv*q
    
    # moist static energy
    m=ms+ml
    
    # hadley cell weighting
    #----------------------
    
    if hadley_cell==1: # ON
        w = np.exp( (-np.sin(latbr)**2) / (0.3**2) ) 
    if hadley_cell==0: # OFF
        w = np.array([0.])
        
    # solve atm. heat transport
    #--------------------------
    
    F=np.zeros((sz))
    FL_EDDIES=np.zeros((sz))
    FS_EDDIES=np.zeros((sz))
    F_HC=np.zeros((sz))
    FS_HC=np.zeros((sz))
    FL_HC=np.zeros((sz))
    
    MEQ=(m[EQ]+m[EQ-1])/2
    for i in np.arange(1,sz-1):
        
        # northward - TOTAL
        F[i] = -2*np.pi*np.cos(latbr[i])*D[i]*rho*dpth*(m[i]-m[i-1])/dlatr
        
        # northward - EDDIES
        FS_EDDIES[i] = (-2*np.pi*np.cos(latbr[i])*D[i]*rho*dpth*(ms[i]-ms[i-1])/dlatr)*(1-w[i])
        FL_EDDIES[i] = (-2*np.pi*np.cos(latbr[i])*D[i]*rho*dpth*(ml[i]-ml[i-1])/dlatr)*(1-w[i])
        
        
        # northward - HADLEY
        F_HC[i]  = F[i]-FL_EDDIES[i]-FS_EDDIES[i]
        gms      = MEQ*gms_factor - (m[i]+m[i-1])/2
        psi      = F_HC[i]/gms
        FL_HC[i] = -lv*psi*(q[i]+q[i-1])/2
        FS_HC[i] = F_HC[i] - FL_HC[i]
    
    # total latent + dry static transport
    FS=FS_EDDIES+FS_HC
    FL=FL_EDDIES+FL_HC
    
    # convergence
    F_conv  = -np.diff(F)/area
    FS_conv = -np.diff(FS)/area
    FL_conv = -np.diff(FL)/area
    
    return F, F_conv, FL_HC, FS_HC, FL_EDDIES, FS_EDDIES, FS, FS_conv, FL, FL_conv


@njit(nogil = True, cache=True)
def moist_heat_transport_v2(Var, State, INPUT, hadley_cell, hydro, height=0):
        
    
    '''
    Northward heat transport (in J/s) and heat convergence (in W m^-2) 
    based on meridional gradients in moist static energy. 
    
    Includes total, latent heat and dry static contributions.
    '''
    
    #--------------------------------------------------------------------------------------
    # Meridional gradients in surface temperature, surface humidity and moist static energy
    #--------------------------------------------------------------------------------------
    
    # declare variables
    #------------------
    
    Ta = State['Ta']
    mean_height = Var['mean_height']
    Lv = Var['Lv']
    r = INPUT['r']
    qa = State['Q']
    dlatr = Var['dlatr']
    rearth = Var['r_earth']
    atm_sphc = Var["atm_sphc"]
    swidth = Var['swidth']
    atm_rho = Var['atm_rho']
    atm_depth = Var['atm_depth']
    DT = INPUT['dt']
    area = Var['sarea']
    lat = Var["lat"]
    latb = Var["latb"]
    latr = Var["latr"]
    latbr = Var["latbr"]
    idx_eq = int(Var['idxeq'][0])
    hadley_cell_constant = INPUT["hadley_constant"]


    # Near-surface temperature
    #-------------------------
    
    if height == 0: # no dependance of height on meridional heat transport
    
        # modify temperature for lapse-rate
        taz = Ta.copy()
    
    elif height == 1: # dependance of height on meridional heat transport
    
        # modify temperature for lapse-rate
        taz = Ta - (0.0065 * mean_height)
        
    # Near-surface humidity
    #----------------------
    
    if hydro==0: # if hydrological cycle is turned off
    
        # specific humidity of near-surface air
        q = (saturation_specific_humidity(Lv, taz, mean_height)) * (r/100)
     
    elif hydro==1: # if hydrological cycle is turned on
    
        # specific humidity of near-surface air (as determined by hydrological cycle)
        q = qa.copy()
        
    # Temperature and humidity gradients
    #-----------------------------------
    
    # temperature
    taz_grads = np.concatenate((np.zeros(1),
                                np.diff(taz)/(dlatr*rearth),
                                np.zeros(1)))
    
    # humidity
    q_grads = np.concatenate((np.zeros(1),
                              np.diff(q)/(dlatr*rearth),
                              np.zeros(1))) 
        
    #---------------------
    # Total heat transport
    #---------------------
    
    # moist static energy transport
    #------------------------------
    
    # mse
    mse = (atm_sphc*taz)+(Lv*q)
    
    # transport 
    mse_transport = (-swidth * 
                     atm_rho * 
                     atm_depth * 
                     DT  * 
                     np.concatenate((np.zeros(1),  # Moist static energy gradient
                                     np.diff(mse)/(dlatr*rearth),
                                     np.zeros(1)))
                     
                     )

    # heat convergence (in W m^-2)
    #-----------------------------
    
    mse_conv = -np.diff(mse_transport)/area
    
    #----------------------------------------------
    # Partition into hadley cell and eddy transport
    #----------------------------------------------
    
    # interpolated fields (needed for calculations)
    #----------------------------------------------
    
    # moist static energy
    mse_b = np.interp(latb, lat, mse)
    
    # humidity
    q_b = np.interp(latb, lat, q)
    
    
    # weighting function for partition between Hadley cell and eddy components
    #-------------------------------------------------------------------------
    
    if hadley_cell==1: # hadley cell turned on
    
        wf = np.exp( (-np.sin(latbr)**2) / (0.3**2) )
        
    if hadley_cell==0: # hadley cell turned off
        wf = np.array([0.])
    
    # gross moist stability of the atmosphere
    #----------------------------------------
    
    # gross moist stability
    g = (hadley_cell_constant * mse_b[idx_eq]) - mse_b
    
    
    # solve for Hadley cell mass transport
    #-------------------------------------
   
    # psi = (mse_transport * wf) / g
    
    # Heat transport by Hadley cell
    #------------------------------
    
    # latent heat (equatorward)
    hadley_latent = -((mse_transport * wf) / g) * Lv * q_b
    # hadley_latent = -(psi) * Lv * q_b
    
    # dry heat (poleward)
    hadley_dry   = ((mse_transport * wf) / g) * (g + (Lv * q_b) )
    # hadley_dry = (psi) * (g + (Lv * q_b) )
    
    # Heat transport by Eddies
    #-------------------------
    
    # latent heat (poleward)
    eddy_latent = (-swidth*atm_depth*atm_rho*Lv*DT*q_grads) * (1-wf)  
    
    # dry heat (poleward)
    eddy_dry = (-swidth*atm_sphc*atm_depth*atm_rho*DT*taz_grads) * (1-wf)
    
    #--------------------------------------------------
    # Partition into total dry static and latent energy
    #--------------------------------------------------
    
    # Dry static
    #-----------
    
    # northward transport
    dry_northward = hadley_dry + eddy_dry
    
    # heat convergence
    dry_conv = -np.diff(dry_northward)/area
    
        
    # Latent
    #-------
    
    # northward transport
    latent_northward = hadley_latent + eddy_latent
    
    # heat convergence
    latent_conv = -np.diff(latent_northward)/area
    
    
    # Return variables
    #-----------------
    
    return mse_transport, mse_conv, hadley_latent, hadley_dry, eddy_latent, eddy_dry, dry_northward, dry_conv, latent_northward, latent_conv



@njit(nogil = True, cache=True)
def snowfall(lat, tas, precip_flux, height, lr, Lm, ice_rho):
    
    '''
    Determines the snowfall fraction, average snowfall flux over each 
    latitudinal band and the associated latent heat flux.
    '''
    
    # Snowfall fraction
    #------------------
    
    # modify surface temperature for elevation-lapse rate effect
    tasz = tas - (lr*height)
    
    # initialize snowfall fraction array
    snowfall_fraction = np.zeros((lat.size))
    
    # complete snow
    i1 = np.where((tasz < 260.))
    snowfall_fraction[i1] = 1.
    
    # partial snow
    i2 = np.where(((tasz >= 260.) & (tasz <= 280.)))
    snowfall_fraction[i2] = 0.05 * (280 - tasz[i2])
    
    # no snow
    i3 = np.where((tasz > 280.))
    snowfall_fraction[i3] = 0.
    
    #----------------------
    # Average snowfall flux
    #----------------------
    
    # snowfall flux (kg m^-2 s^-1)
    #-----------------------------
    snowfall_flux = precip_flux * snowfall_fraction
    
    # snowfall rate (m per day)
    #-----------------------------
    snowfall_rate = (snowfall_flux / ice_rho) * (60*60*24) 
    
    # latent heat flux of snowfall (W m^-2) 
    #--------------------------------------
    snowfall_heat_flux = snowfall_flux * Lm
    
    return snowfall_fraction, snowfall_flux, snowfall_rate, snowfall_heat_flux



@njit(nogil = True, cache=True)
def snow_accumulation(snowfall_fraction, snow_fraction, snow_thick, snowfall_rate, precip_flux, 
                      sarea, ice_rho, secs_in_day, snow):
    
   '''
   Determines changes in snowpack thickness and snowpack fraction of the land
   surface during accumulation
       
       Optional:
           
           snow = 1: Snowfall is evenly distributed over latitude band
           
           snow = 2: Snowfall covers area of latitude band according to the 
                     snowfall fraction
   '''
    
   #-------------------------------------------
   # Snowpack variables from previous time step
   #-------------------------------------------
   
   # land snow fraction
   snow_fraction0 = snow_fraction.copy()
   
   # snow thickness
   snow_thick0 = snow_thick.copy()
   
   # snow area
   snow_area0 = sarea*snow_fraction0
   
   # snow volume
   snow_volume0 = snow_thick0 * snow_area0
   
   #--------------------------------------
   # Snowpack variables from this time step
   #---------------------------------------
   
   # land snow fraction
   snow_fraction1 = snow_fraction0.copy()
   
   # snow thickness
   snow_thick1 = snow_thick0.copy()
   
   # snow area
   snow_area1 = snow_area0.copy()
   
   # snow volume
   snow_volume1 = snow_volume0.copy()
   
   #--------------------------------------------
   # Modification to snow fraction and thickness
   #--------------------------------------------
        
   # if snowfall fraction is evenly distributed
   #-------------------------------------------
       
   if snow == 1:
       
       # new snow thickness
       #-------------------
       
       snow_thick1 = (
                      # snow thickness of previous time step
                      snow_thick0 
                      
                      +
                      
                      # additional snow thickness
                      ( (snowfall_rate) )
                      
                      )
       
       # new snow fraction
       #------------------
       
       snow_fraction1 = snow_fraction0.copy()
       
       i = np.where(snow_thick1 > 0.)
       j = np.where(snow_thick1 == 0.)
       
       snow_fraction1[i] = 1.
       snow_fraction1[j] = 0.
       
       k = np.where(snow_thick1 > 500.)
       snow_thick1[k] = 500.
       
       
       
   # if snowfall fraction is unevenly distributed
   #---------------------------------------------
    
   if snow == 2:
       
       # if snowfall fraction is greater than snowpack fraction
       i = np.where( (snowfall_fraction > 0.) & (snowfall_fraction >= snow_fraction0) ) 
    
       # new snow fraction
       snow_fraction1[i] = snowfall_fraction[i]
         
       # new snow area
       snow_area1[i] = snow_fraction1[i] * sarea[i]
         
       # new snow thickness
       snow_thick1[i] = (
                       
                       # snow thickness adjusted for new snow area 
                       (snow_volume0[i] / snow_area1[i])
                       
                       +
                       
                       # additional snow thickness from snowfall
                       ( (precip_flux[i]/ice_rho)*60*60*24)
                       
                       )
       
       # if snowfall fraction is less than snowpack fraction
       j = np.where( (snowfall_fraction > 0.) & (snowfall_fraction < snow_fraction0) )
       
       # new snow fraction
       snow_fraction1[j] = snow_fraction0[j]
         
       # new snow area
       snow_area1[j] = snow_area0[j]
         
       # additonal snow volume (m^3)
       additional_snow = np.zeros((snow_fraction0.size))
       additional_snow[j] = (
                         
                            # area of snowfall
                            (snowfall_fraction[j] * sarea[j])
                            
                            *
                            
                            # additional snow thickness from snowfall
                            ( (precip_flux[j]/ice_rho)*60*60*24)
                            
                            )
         
       # new snow thickness
       snow_thick1[j] = (snow_thick0[j] + (additional_snow[j]/snow_area1[j]))
       
       k = np.where(snow_thick1 > 500.)
       snow_thick1[k] = 500.
    
   return snow_fraction1, snow_thick1
    

@njit(nogil = True, cache=True)
def sea_ice(to, tsi, sivol, snfl, sifraction, ebm_lat, ls, ice_rho, Lm,
            Ksi, ocean_hc, o_area, ebm_step, sea_ice_sphc, si_ithick):   
    
    '''
    Computes sea-ice volume and the fraction of ocean surface
    area based on the surface ocean temperature. Based on the 
    parameterizations of Gildor and Tziperman (2001).
    '''
    
    # Set the ice volume of the previous time-step.
    #----------------------------------------------
    
    #----> sivol_t1 = sea ice volume in previous time step
    sivol_t1 = sivol.copy() # m^3
    
    #----> sivol_t2 = sea ice volume in current time step (after this function)
    sivol_t2 = np.zeros((ebm_lat.size)) # m^3
    sivol_t2[ls] = np.NaN
    
  
    # Set the surface ocean temperature of the previous time-step.
    #------------------------------------------------------------
    
    #----> so_t1 = surface ocean temp in previous time step
    so_t1 = to.copy() # m^3
    
    #----> so_t2 = surfa ocean temp in current time step (after this function)
    so_t2 = so_t1.copy()
    
 
    # Set the sea-ice temperature of the previous time-step.
    #-------------------------------------------------------
    
    #----> tsi_t1 = sea ice temperature in previous time step
    tsi_t1 = tsi.copy()
    
    #----> tsi_t2 = sea ice temperature in current time step
    tsi_t2 = np.zeros((ebm_lat.size)) # m^3
    tsi_t2[ls] = np.NaN
    

    # Set the sea-ice fraction of the previous time-step.
    #----------------------------------------------------\
    
    #----> si_fraction_t1 = sea ice fraction in previous time step
    si_fraction_t1 = sifraction.copy()
    
    #----> si_fraction_t1 = sea ice fraction in current time step
    si_fraction_t2 = np.zeros((ebm_lat.size)) # m^3
    tsi_t2[ls] = np.NaN
    

    # Initialize arrays for storing some ice volume related variables
    #----------------------------------------------------------------
    
    h_avail          = np.zeros((ebm_lat.size))   # Energy available for sea-ice formation (W)
    h_need1          = np.zeros((ebm_lat.size))   # Energy needed to melt all sea-ice (W)
    h_need2          = np.zeros((ebm_lat.size))   # Energy needed to melt all sea-ice (W)
    si_melt_flux     = np.zeros((ebm_lat.size))   # Sea ice melt flux (in W/m2)
    sf_vol           = np.zeros((ebm_lat.size))   # snowfall in m^3
    si_thick         = np.zeros((ebm_lat.size))   # Sea ice thickness (m)
    si_thick[ls]     = np.NaN
    exc_melt         = np.zeros((ebm_lat.size))   # Excess melt ()
    exc_melt[ls]     = np.NaN
    

    # Some constants
    #---------------
    
    rho_lm = ice_rho * Lm  # density x latent heat of fusion.
    

    # Sea ice formation
    #------------------
    
    # locate zonal bands where surface ocean temperature is below freezing point
    i = np.where(so_t1 < Ksi)
    
    
    # calculate energy available for surface freezing (in Watts or J/s)
    h_avail[i] = ( ( (ocean_hc*o_area[i]) 
                  
                  / ebm_step) 
    
                  * (Ksi - so_t1[i]) )
    
    
    # Convert surface snowfall into volume of ice addition
    sf_vol[i] = ((snfl[i] / ice_rho) # in m/s
              * (o_area[i] * si_fraction_t1[i]))  # to ---> m^3/s
                  #|
                  #------> multiply by surface of sea-ice to get volume
    
    # Add sea ice from surface water freezing and snowfall
    sivol_t2[i] = ((ebm_step * # in s 
                    
                    ((h_avail[i] / (rho_lm)) # m^3/s sea-ice formation
                     
                     + sf_vol[i])) # in m^3/s snowfall gain
                    
                     + sivol_t1[i]) # + sea ice volume of previous timestep
    
    so_t2[i] = Ksi

    
    #----------------
    # Sea ice melting
    #----------------
    
    # locate zonal bands where surface ocean temperature is above freezing point
    j = np.where((so_t1 > Ksi) & (sivol_t1 > 0))
    
    
    # calculate energy available for surface melting (in Watts or J/s)
    h_avail[j] = ( ( (ocean_hc*o_area[j]) 
                  
                  / ebm_step) 
    
                  * (Ksi - so_t1[j]) )
    
    # Convert surface snowfall into volume of ice addition
    sf_vol[j] = ((snfl[j] / ice_rho) # in m/s
              * (o_area[j] * si_fraction_t1[j]))  
                  #|
                  #------> multiply by surface of sea-ice to get volume
                  

    # Energy need to raise the ice temperature to freezing point 
    h_need1[j] = ( (Ksi - tsi_t1[j]) *
                  
                  (ice_rho * sea_ice_sphc * si_ithick[j] 
                   * (o_area[j] * si_fraction_t1[j]) )
                  
                  )  / ebm_step   # in J/s 
    

    # Energy needed to melt all the ice
    h_need2[j] = ((sivol_t1[j]) * (rho_lm) # Joules
                 / ebm_step) # J/s
    
    
    
    # not enough energy to raise to melting point
    #--------------------------------------------
    
    # locate zonal bands where not enough energy to raise temperature of ice
    j1 = np.where((so_t1 > Ksi) 
                  & (sivol_t1 > 0) 
                  & (-(h_avail) < h_need1) )
    
    # raise the temperature of the ice
    tsi_t2[j1] = ( tsi_t1[j1] -
                 
                 ( h_avail[j1] / 
                     
                 ( ice_rho * sea_ice_sphc * si_ithick[j1]
                       * (o_area[j1] * si_fraction_t1[j1]) ) )
                                                
                  * ebm_step )  # in J/s 
    
    # set ice volume
    sivol_t2[j1] = np.copy(sivol_t1[j1])
    
    # set ocean temperature
    so_t2[j1] = Ksi
    
       
    # Exceeds melting point but can't melt all ice
    #---------------------------------------------
    
    # # where energy can melt raise to freezing point but not melt all ice
    j2 = np.where((so_t1 > Ksi) 
                  & (sivol_t1 > 0) 
                  & (-(h_avail) > h_need1)
                  & (-(h_avail) < (h_need1 + h_need2)))
    
    
    # Remove sea-ice due to surface melting- considering snowfall gains.
    sivol_t2[j2] = ( (ebm_step * # in s
                          
                     ((h_avail[j2] + h_need1[j2]) /(rho_lm)) # in m^3/s sea-ice melting 
                     
                     + sf_vol[j2]) # in m^3/s snowfall gain
                     
                     + sivol_t1[j2]) # + ice volume of previous time step.
                     
                          
    # where sea-ice is melting, but not all ice has melted, set surface ocean
    # temperature to freezing
    so_t2[j2] = Ksi 
    
    # where sea-ice is melting, but not all ice has melted, set ice
    # temperature to freezing
    tsi_t2[j2] = Ksi
    
    
    # Raise ice temperature and melt all ice
    #---------------------------------------
    
    # locate zonal bands where there is enough energy available to melt the ice
    j3 = np.where((so_t1 > Ksi) 
                  & (sivol_t1 > 0) 
                  & (-(h_avail) > h_need1)
                  & (-(h_avail) > (h_need1 + h_need2)))
    
    # Remove sea-ice due to surface melting- considering snowfall gains.
    sivol_t2[j3] = ( (ebm_step * # in s
                          
                     ((h_avail[j3] + h_need1[j3]) /(rho_lm))# in m^3/s sea-ice melting 
                     
                     + sf_vol[j3]) # in m^3/s snowfall gain
                     
                     + sivol_t1[j3]) # + ice volume of previous time step.

    
    # convert excess melt into additional surface heating
    so_t2[j3] = Ksi + ( ((0 - sivol_t2[j3]) * (rho_lm))
                              / o_area[j3] ) / ocean_hc
    
    # set ice volume to zero
    sivol_t2[j3] = 0

    #-------------------------------
    # Calculate new sea-ice fraction
    #-------------------------------
    
    # calculate the sea-ice cover
    #----------------------------
    
    # where sea ice cover is less than ocean surface area
    k = np.where((sivol_t2 > 0))
    
    # set the new sea-ice fraction
    si_fraction_t2[k] = sivol_t2[k] / (si_ithick[k] * o_area[k])
    
    # set the sea ice thickness to the standard value
    si_thick[k] = si_ithick[k] 
    
    # when sea-ice cover < 1.
    #------------------------
    k1 = np.where((si_fraction_t2 > 0.) & (si_fraction_t2 < 1.))
    
    # set the sea-ice temperature to zero
    tsi_t2[k1] = Ksi
    
    # when sea-ice cover > 1.
    #------------------------
    k2 = np.where(si_fraction_t2 > 1.) 

    # Let the sea-ice temperature drop below zero.
    
    # Derivation:
        
        # State["Tsi"] = Var["Ksi"] + [ ( (1 - State["si_fraction"]) * Var["oarea"]
        # * State["si_thick"] * Var["rho_ice"] * Var["Lm"] ) 
        
        #   / (Var["oarea"]*State["si_thick"] * Var["rho_ice"] * Var["sea_ice_sphc"])
        
        # )
        
        # ======
        
        # State["Tsi"] = Var["Ksi"] + 
        
        # [ ( (1 - State["si_fraction"]) * Var["Lm"]/Var["sea_ice_sphc"] ) 
        
    tsi_t2[k2] = tsi_t1[k2] + ( (1-si_fraction_t2[k2]) * 
                                 (Lm/sea_ice_sphc) )
    
    # set the new sea-ice volume
    sivol_t2[k2] = si_ithick[k2] * o_area[k2]
    
    # set new sea-ice fraction
    si_fraction_t2[k2] = 1.
    
    si_fraction_t2[ls] = np.NaN
    
    
    #--------------------------------------
    # Record melting and ice volume changes
    #--------------------------------------
    
    # Record changes in ice volume
    sivol_change = sivol_t2 - sivol_t1
    
    # Record changes in energy fluxes due to sea ice changes
    si_melt_flux = ((sivol_change * (rho_lm)) # in Joules
                        / (o_area * ebm_step)) # to W/m^2
    
    si_melt_flux[ls] = np.NaN

   
    return so_t2, tsi_t2, si_thick, sivol_t2, si_fraction_t2, si_melt_flux


@njit(nogil = True, cache=True)
def snow_melt(lat, snow_thick, snow_fraction, ts, lr, height, K, land_hc, Lm, 
              ice_rho, mask, secs_in_day, sarea, snow):
   
    '''
    Determines changes in snowpack thickness and snowpack fraction of the land
    surface during melting:
        
        Optional:
            
            snow = 1: Snowfall is evenly distributed over latitude band
            
            snow = 2: Snowfall covers area of latitude band according to the 
                      snowfall fraction
    '''
   
    #-------------------------------------------
    # Snowpack variables from previous time step
    #-------------------------------------------
   
    # land snow fraction
    snow_fraction0 = snow_fraction.copy()
   
    # snow thickness
    snow_thick0 = snow_thick.copy()
   
    #-----------------------
    # Initialize melt arrays
    #-----------------------
    
    melt_rate     = np.zeros((lat.size))     # melt rate (m/day).
    melt_flux     = np.zeros((lat.size))     # melt energy flux (W/m2)
   
    #--------------------
    # Surface temperature
    #--------------------
    
    # modified for lapse rate effect
    tsz = ts - (0.0065 * height)
    
    #--------------------------------------
    # Average snow melt rate over snow area
    #--------------------------------------
    
    if snow == 1:
        
        # locate snow-covered land with temperatures exceeding 0C.
        i = np.where( ( snow_thick > 0) & (tsz > K))
        
        # melt snow with excess temperature
        melt_rate[i] = (tsz[i] - K) * ((land_hc[i])/(Lm * ice_rho))
        
        # reset temperatures of snow-covered land to 0C.
        tsz[i] = K
        
    if snow == 2:
        
        # locate snow-covered land with temperatures exceeding 0C.
        i = np.where( (snow_thick0 > 0) & (tsz > K) )
        
        # melt snow with excess temperature
        melt_rate[i] = (tsz[i] - K) * (land_hc[i]/(Lm * ice_rho)) * (1/snow_fraction0[i])
        
        # reset temperatures of snow-covered land to 0C.
        tsz[i] = K
    
    
    #--------------------------------------
    # Modify snow thickness due to ablation
    #--------------------------------------
    
    # if snowfall fraction is evenly distributed
    #-------------------------------------------
        
    if snow == 1:
        
        # new snow thickness
        snow_thick1 = snow_thick0 - melt_rate
        
        # new snow fraction
        snow_fraction1 = snow_fraction0.copy()
        
    # if snowfall fraction is unevenly distributed
    #---------------------------------------------
     
    if snow == 2:
        
        # equally partition average snow melt into vertical and lateral melting
        
        # new snow thickness
        snow_thick1 = snow_thick0 - ( (melt_rate*0.5) ) 
        
        # new snow fraction
        snow_fraction1 = (snow_fraction0) - ( ( (melt_rate*0.5) * snow_fraction0 ) / snow_thick1)
        
        # convert nan's to zeros
        i = np.where(np.isnan(snow_fraction1) == True)
        snow_fraction1[i] = 0.
  
        
    #---------------------------------------
    # Convert excess melt to surface heating
    #---------------------------------------
    
    # Where the amount of surface melting exceeds the actual volume of the
    # snowpack, the excess energy is used to heat the surface.
    
    # locate areas of excess melt
    i = np.where( (melt_rate > 0) & (melt_rate >= snow_thick0))
    
    # amount of excess melt
    excess_melt    = np.zeros((lat.size))
    excess_melt[i] = melt_rate[i] - snow_thick0[i]
    
    if snow == 1:
    
        # correct excess melt 
        melt_rate[i] = snow_thick0[i]
        
        # additional surface heating
        tsz[i] = tsz[i] + ( excess_melt[i] * ice_rho * Lm ) / land_hc[i]
        
        # correct snow thickness and snow fraction
        snow_thick1[i]    = 0
        snow_fraction1[i] = 0.
        
    if snow == 2:
        
        # correct excess melt 
        melt_rate[i] = snow_thick0[i]
        
        # additional surface heating
        tsz[i] = tsz[i] + ( excess_melt[i] * snow_fraction0[i] * ice_rho * Lm ) / land_hc[i]
        
        # correct snow thickness and snow fraction
        snow_thick1[i]    = 0
        snow_fraction1[i] = 0.
        
    #----------------------------------------------------
    # Latent heat flux of melt (averaged over total area)
    #----------------------------------------------------
    
    melt_flux = ((melt_rate * snow_fraction0 * ice_rho)/secs_in_day) * Lm
    
    #----------------------------------------------
    # "Re-modify" temperature for lapse rate effect
    #----------------------------------------------
    
    tsc = tsz + (0.0065 * height)
    
    #-----
    # Mask
    #-----
    
    snow_thick1    = snow_thick1 * mask
    snow_fraction1 = snow_fraction1 * mask
    tsc           = tsc * mask
    melt_rate     = melt_rate * mask
    melt_flux     = melt_flux * mask
        
    return snow_thick1, snow_fraction1, tsc, melt_rate, melt_flux 


@njit(nogil = True, cache=True)
def ocean_fluxes(State, Var, INPUT):
    
    '''
    Calculates meridional heat fluxes associated with ocean heat transport
    '''
    
    
    # indexes for ocean boundaries
    #-----------------------------
    
    # ocean centre of circulation
    co = np.where(Var["olatb"] == State["idxcos"])[0][0]
    
    # boundary for ocean south
    sbi = np.where(Var["latb"] == -70.)[0][0]
    
    # boundary for ocean north
    nbi = np.where(Var["latb"] == 80.)[0][0]
    
    # reshape 
    To = State["To"].copy().reshape(6, Var["lat"].size)[:,sbi:nbi]
    
    
    # some constants
    #---------------
    
    rhospc = (Var["ocean_rho"]*Var["ocean_sphc"])[0] # specific heat * denisty of seawater
    oa = Var["oa"]
    odepth = Var["odepth"]
    odepth_diff = Var["odepth_diff"]
    ow = Var["ow"]
    olat = Var["olat"]
    olatb = Var["olatb"]
    rearthlat = Var["r_earth"] * Var["dlatr"]
    ww = State["ww"]
    u1 = State["u1"]
    u2 = State["u2"]
    dh = INPUT["dh"]
    dz = State["dz"]
    do = INPUT["do"]
    
 
    # horizontal advective fluxes (surface)
    #--------------------------------------
    
    # initialize
    advfs      = np.zeros((6, olatb.size))
    advfs_conv = np.zeros((6, olat.size))
          
    # northern hemisphere flux (in Watts)
    advfs[0, co+1:advfs.shape[1]-1] = (
                      ow[co+1:advfs.shape[1]-1]        # ocean width
                      * odepth[0]                      # ocean odepth
                      * u1[co+1:advfs.shape[1]-1]      # surface velocity
                      * To[0, co:advfs.shape[1]-2]           # surface temperature
                      * (rhospc) )               # specific heat and denisty of water
    
    # southern hemisphere flux (in Watts)
    advfs[0, 1:co] = (
                      ow[1:co]      # ocean width
                      * odepth[0]                  # ocean odepth
                      * u1[1:co]    # surface velocity
                      * To[0, 1:co]         # surface temperature
                      * (rhospc) )           # specific heat and denisty of water
                       
    # energy convergence (in Watts per m^2)
    advfs_conv[0, :] = (
                  -np.diff(advfs[0,:]) # convergence
                  
                  /
                  
                  (oa) )  # surface ocean area
    

    # horizontal advective fluxes (bottom)
    #--------------------------------------

    # initialize
    # advfs      = np.zeros((6, olatb.size)) 
    # advfs_conv = np.zeros((6, olat.size))
    
    # northern hemisphere flux (in Watts)
    advfs[5, co+1:advfs.shape[1]-1] = (
                        ow[co+1:advfs.shape[1]-1]       # ocean width
                        * odepth[5]            # ocean odepth
                        * u2[co+1:advfs.shape[1]-1]     # surface velocity
                        * To[5, co+1:advfs.shape[1]-1]    # surface temperature
                        * (rhospc) )        # specific heat and denisty of water
    
    # southern hemisphere flux (in Watts)
    advfs[5, 1:co] = (
                      ow[1:co]         # ocean width
                      * odepth[5]                  # ocean odepth
                      * u2[1:co]       # surface velocity
                      * To[5, 0:co-1]            # surface temperature
                      * (rhospc) )              # specific heat and denisty of water
    
    
    # energy convergence (in Watts per m^2)
    advfs_conv[5,:] =  (
                    -np.diff(advfs[5,:])
                   
                    /
                    
                    (oa))
    
    # Vertical advective fluxes 
    #--------------------------
    
    # initialize
    vadvf      = np.zeros((7, olat.size))
    vadvf_conv = np.zeros((6, olat.size))
    
    # calculate vertical fluxes (W)
    for i in prange(0, vadvf.shape[1]): # for each column

        for j in np.arange(1, vadvf.shape[0]-1): # for each layer (except top-bottom)
        
            if ww[i] > 0.: # if upwelling
            
                vadvf[j,i]= oa[i]*ww[i]*rhospc*To[j,i] 
                
            elif ww[i]<0.: # if downwelling
                
                vadvf[j,i]= oa[i]*ww[i]*rhospc*To[j-1,i]
                
    # calculate vertical divergence (W m^2)
    for j in np.arange(0, vadvf_conv.shape[0]): # for each layer
    
        vadvf_conv[j,:] = (vadvf[j+1,:] - vadvf[j,:])/ oa
    
    
    # horizontal diffusion (eddy/gyre transport)
    #-------------------------------------------

    # intialize
    hdiffs      = np.zeros((6, olatb.size))
    hdiffs_conv = np.zeros((6, olat.size))
    
    # northward ocean heat transport (W)
    for i in np.arange(0, 1): # for each layer
    
        # southern hemisphere
        hdiffs[i,:] = ( 
            
            ow *
            odepth[0] *
            np.concatenate(( np.zeros(1), (-np.diff(To[i,:].copy()))/(rearthlat), np.zeros(1) )) *
            do *
            rhospc
                        )
    
    # convergence of surface heat diffusion (W/m2) 
    for i in np.arange(0, 1): # for each layer
    
    
        hdiffs_conv[i,:] = (
            
                        -np.diff(hdiffs[i,:].copy())  # divergence of surface heat diffusion
                       
                        /  # divide by surface area
                       
                        (oa))

    
    # horizontal diffusion (interior ocean)
    #--------------------------------------
    
    # initialize
    hdiffi = np.zeros((6,olatb.size))
    hdiffi_conv = np.zeros((6,olat.size))
    
    # northward ocean heat transport (W)
    for i in np.arange(1, hdiffi.shape[0]): # for each layer
    
        hdiffi[i,:] = ( 
            
            ow *
            odepth[i] *
            np.concatenate(( np.zeros(1), (-np.diff(To[i,:].copy()))/(rearthlat), np.zeros(1) )) *
            dh[0] *
            rhospc
                                         )
        
    # convergence of interior horizontal heat diffusion (W/m2)  
    for i in np.arange(1, hdiffi.shape[0]): # for each layer
    
        hdiffi_conv[i,:] = ( -np.diff(hdiffi[i,:].copy()) / oa )
        
     
    # vertical diffusion 
    #-------------------
    
    # initialize
    vdiff      = np.zeros((7, olat.size))
    vdiff_conv = np.zeros((6, olat.size))
    
    # vertical diffusive fluxes (W)
    for i in np.arange(1, vdiff.shape[0]-1): # for each layer
    
        vdiff[i,:] = (
            
            oa *
            (To[i,:].copy() - To[i-1,:].copy()) / odepth_diff[i-1] *
            dz *
            rhospc
            
            )
        
    # vertical diffusion divergence (W/m^2)
    for j in np.arange(0, vdiff_conv.shape[0]): # for each layer
    
        vdiff_conv[j,:] = (vdiff[j+1,:] - vdiff[j,:])/ oa
    
    
    # return advfs, advfs_conv, advfb, advfb_conv, vadvf, vadvf_conv, hdiffi, hdiffi_conv, vdiff, vdiff_conv, hdiffs, hdiffs_conv


    return vadvf.flatten(), vadvf_conv.flatten(), hdiffi.flatten(), hdiffi_conv.flatten(), vdiff.flatten(), vdiff_conv.flatten(), hdiffs.flatten(), hdiffs_conv.flatten(), advfs.flatten(), advfs_conv.flatten()
    # return vadvf, vadvf_conv, hdiffi, hdiffi_conv, vdiff, vdiff_conv, hdiffs, hdiffs_conv, advfs, advfs_conv
    
    
@njit(nogil = True, cache=True)
def land_sea_dry_mixing(Var, State, INPUT):
    
    '''
    Calculates the east-west exchange of heat between land and ocean in a 
    given zone.
    '''
    
    # initialize arrays
    land_flux  = np.zeros((Var["lat"].size))
    ocean_flux = np.zeros((Var["lat"].size))
    
    # index for where zonal heat exchange occurs
    i = np.where((INPUT['land_fraction'] > 0.) & (INPUT['land_fraction'] < 1.))
    
    # east-west heat fluxes (based on eq. 18 from Harvey, 1988)
    #----------------------------------------------------------
    
    # heat flux over land
    land_flux[i] = ( ((2*Var["dlatr"]*INPUT["dew"]*Var["atm_hc"]) / (Var["larea"][i]*np.pi*np.cos(Var["latr"][i])))
                      *
                      ((State["Tal"][i] - State["Tao"][i]) * (-1))
                      )
    
    # heat flux over ocean
    ocean_flux[i] = ( ((2*Var["dlatr"]*INPUT["dew"]*Var["atm_hc"]) / (Var["oarea"][i]*np.pi*np.cos(Var["latr"][i])))
                      *
                      ((State["Tal"][i] - State["Tao"][i]) * (-1)**2)
                      )
    
    return land_flux, ocean_flux


@njit(nogil = True, cache=True)
def land_sea_moist_mixing(Var, State, INPUT):
    
    '''
    Calculates the east-west exchange of moisture between land and ocean in a 
    given zone.
    '''
    
    # initialize arrays
    land_flux  = np.zeros((Var["lat"].size))
    ocean_flux = np.zeros((Var["lat"].size))
    
    # index for where zonal heat exchange occurs
    i = np.where((INPUT['land_fraction'] > 0.) & (INPUT['land_fraction'] < 1.))
    
    # east-west heat fluxes (based on eq. 18 from Harvey, 1988)
    #----------------------------------------------------------
    
    # heat flux over land
    land_flux[i] = ( ((2*Var["dlatr"]*INPUT["dew"]*Var["atm_rho"]*Var["atm_depth"]) / (Var["larea"][i]*np.pi*np.cos(Var["latr"][i])))
                      *
                      ((State["Q_land"][i] - State["Q_ocean"][i]) * (-1))
                      )
    
    # heat flux over ocean
    ocean_flux[i] = ( ((2*Var["dlatr"]*INPUT["dew"]*Var["atm_rho"]*Var["atm_depth"]) / (Var["oarea"][i]*np.pi*np.cos(Var["latr"][i])))
                      *
                      ((State["Q_land"][i] - State["Q_ocean"][i]) * (-1)**2)
                      )
    
    return land_flux, ocean_flux
    
    
    
#------------------------------------
# Energy balance model: main function
#------------------------------------

# types must be defined outside the numba function
float_array_2d = types.float64[:,:]
float_array_3d = types.float64[:,:,:]

@njit(nogil = True, cache=True)
def ebm(Var, State, INPUT, I, znth_dw, settings, key_variables, 
        
        # fixed land albedo
        fixed_land_albedo=None,
        
        # fixed ocean albedo
        fixed_ocean_albedo=None,
        
        # fixed land snow 
        fixed_snow_fraction=None,
        fixed_snow_thick=None,
        fixed_snow_melt=None,
        
        # fixed sea ice
        fixed_si_fraction=None,
        fixed_si_thick=None,
        fixed_si_melt=None,
        
        # fixed ocean transport
        fixed_ocean_transport=None,
        
        # fixed atm. transport
        fixed_atm_transport=None):
    
    '''
    Main energy balance model function. 
    '''
    
    #---------------------
    # Define some settings
    #---------------------
    
    snow= settings["snow"][0] # 0 = no snowfall (albedo function of temperature), 1 = snowfall (uniform coverage), 2 = snowfall (fractional coverage) 
    
    hydro= settings["hydro"][0]  # 0 = no hydrological cycle, 1 = hydrological cycle
    
    seaice=settings["seaice"][0]  # 0 = no sea ice, 1 = sea ice
    
    transport=settings["transport"][0]  # 0 = no meridional heat transport, 1 = transport
    
    hadley_cell=settings["hadley_cell"][0]  # 0 =  no hadley cell parameterization, 1 = hadley cell
    
    atm_transport=settings["atm_transport"][0]  # 0 = no atmospheric heat transport, 1 = atm. heat transport
    
    ocn_transport=settings["ocn_transport"][0]  # 0 = no ocean heat transport, 1 = ocn. heat transport
    
    height=settings["height"][0]  # 0 = evaporation, precipitation and transport dependant on surface elevation, 1 = they are not. 
    
    version=settings["version"][0]  # 0 = classic 'dry' EBM, 1 = moist EBM
     
    #--------------------------------
    # Initialize StateYear dictionary
    #--------------------------------

    # StateYear: Dictionary containing information on model variables for the
    # last year of the model run.

    StateYear = Dict.empty(key_type=types.unicode_type, value_type=float_array_2d)
    
    # Fill StateYear with State variables
    for keys, values in State.items(): # for every variable in State
        StateYear[keys] = np.zeros(( len(values), 365 ))
    
    # add insolation
    StateYear['I'] = I.copy()
        
    # Initialize StateAnnual dictionary
    #----------------------------------
    
    # StateAnnual: Dictionary containing information on annual mean for key state variables for all model years.
    StateAnnual = Dict.empty(key_type=types.unicode_type, value_type=float_array_3d)
   
    for keys in key_variables:
        StateAnnual[keys] = np.zeros((INPUT['nyrs'].astype('i8')[0], Var['ndays'].size, State[keys].shape[0]), dtype="f8")

    #-----------------------
    # Execute the main loop.
    #-----------------------
   
    # prescribe initial ocean overturning rate
    #-----------------------------------------
    State["u1"], State["u2"], State["ww"] = horizontal_ocean_velocities(Var["olat"], Var["olatb"], Var["olatr"], Var["olatbr"], Var["dlatr"], State["idxcos"], np.array([80.]), np.array([-70.]), np.zeros((Var["olat"].size))+0.7, np.zeros((Var["olatb"].size))+0.7, Var["r_earth"], State["ww"], Var["odepth"])

    
    for year in np.arange(0, INPUT['nyrs'].astype('i8')[0]): # for every year

    
        for day in np.arange(0, 365): # for every day
        
        
        
            #-----------------------------
            # Calculate the surface albedo
            #-----------------------------
            
            # Over land
            #----------
            
            if fixed_land_albedo is None: # albedo calculated internally
            
                if snow == 0: # as a function of surface temperature
            
                    State["alpha_land"], State["snow_fraction_land"] = surface_albedo_land_v2(Var["lat"], INPUT["land_height"], Var["lr"], INPUT["alphag"], State["Tal"], State["Tl"], INPUT["alphas"], INPUT["alphaws"], INPUT["alphai"], znth_dw[:,day], Var["land_mask"], INPUT['ice_fraction'])
                
                elif snow != 0: # as a function of snowfal/melt
                
                    State["alpha_land"] = surface_albedo_land_v3(Var["lat"], State["Tl"], Var["lr"], INPUT['land_height'], INPUT["alphag"], INPUT["alphai"], INPUT["alphas"], INPUT["alphaws"], INPUT['ice_fraction'], State["snow_fraction_land"], znth_dw[:,day], Var["land_mask"])
        
            
            else: # albedo prescribed internally
            
                State["alpha_land"] = fixed_land_albedo[:,day]
            
                
            # Over ocean
            #-----------
            
            if fixed_ocean_albedo is None: # albedo calculated internally
            
                State["alpha_ocean"] = surface_albedo_ocean(Var["lat"], State["si_fraction"], State["si_melt_flux"], INPUT["alphasimn"], INPUT["alphasimx"], znth_dw[:,day], Var["ocean_mask"])
                
            else: # albedo prescribed internally
            
                State["alpha_ocean"] = fixed_ocean_albedo[:,day]
                
        
            #-----------------------------------------------------
            # Calculate the shortwave radiative fluxes (in W m^-2)
            #-----------------------------------------------------
    
            # Shortwave radiative fluxes
            State["rsut_land"], State["rsds_land"], State["rsus_land"]    =  SWR_param(INPUT["ccl"], State["Tal"], State["alpha_land"], State["alpha_land"], znth_dw[:,day], INPUT["land_height"], I[:,day], INPUT["tau"])
            State["rsut_ocean"], State["rsds_ocean"], State["rsus_ocean"] =  SWR_param(INPUT["cco"], State["Tao"], State["alpha_ocean"], State["alpha_ocean"], znth_dw[:,day], Var["ocean_height"], I[:,day], INPUT["tau"])
    
            # Net downwards shortwave radiation at TOA
            State["rsdtnet_land"]  = I[:,day] - State["rsut_land"]
            State["rsdtnet_ocean"] = I[:,day] - State["rsut_ocean"] 
            State["rsdtnet"]       = weighted_average(INPUT, State["rsdtnet_land"], State["rsdtnet_ocean"])
            
            # Net downwards shortwave radiation at surface
            State["rsdsnet_land"]  = State["rsds_land"] - State["rsus_land"]
            State["rsdsnet_ocean"] = State["rsds_ocean"] - State["rsus_ocean"]
    
            # Shortwave radiation absorbed by atmosphere
            State["rsatmnet_land"]  = State["rsdtnet_land"]  - State["rsdsnet_land"]
            State["rsatmnet_ocean"] = State["rsdtnet_ocean"] - State["rsdsnet_ocean"]
            
            if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
    
                # save weighted averages for land and ocean components
                State["rsds"]      = weighted_average(INPUT, State["rsds_land"], State["rsds_ocean"])
                State["rsus"]      = weighted_average(INPUT, State["rsus_land"], State["rsus_ocean"])
                State["rsut"]      = weighted_average(INPUT, State["rsut_land"], State["rsut_ocean"])
                State["rsdsnet"]   = weighted_average(INPUT, State["rsdsnet_land"], State["rsdsnet_ocean"])
                State["rsatmnet"]  = weighted_average(INPUT, State["rsatmnet_land"], State["rsatmnet_ocean"])
    
            #----------------------------------------------------
            # Calculate the longwave radiative fluxes (in W m^-2)
            #----------------------------------------------------
    
            # Longwave radiation fluxes
            State["rlus_land"], State["rlds_land"], State["rlut_land"]    = LWR_param(Var["sigma"], INPUT["co2"], State["Tal"], State["Tl"], State["Tagx"], INPUT["land_height"], INPUT["ccl"], Var["cloud_emissivity"], INPUT['GHG_amp'])
            State["rlus_ocean"], State["rlds_ocean"], State["rlut_ocean"] = LWR_param(Var["sigma"], INPUT["co2"], State["Tao"], State["Tos"], State["Tagx"], Var["ocean_height"], INPUT["cco"], Var["cloud_emissivity"], INPUT['GHG_amp'])
            
            # Weighted average of upwards longwave radiation at TOA
            State["rlut"] = weighted_average(INPUT, State["rlut_land"], State["rlut_ocean"])
            
            # Net downwards longwave radiation at surface
            State["rldsnet_land"]  = State["rlds_land"]  - State["rlus_land"]
            State["rldsnet_ocean"] = State["rlds_ocean"] - State["rlus_ocean"]
            
            # Surface absorbed longwave radiation
            State["rldsnet_land"]  = State["rlds_land"]  - State["rlus_land"]
            State["rldsnet_ocean"] = State["rlds_ocean"] - State["rlus_ocean"]
    
            # Longwave radiation absorbed by atmosphere
            State["rlatmnet_land"]  = State["rlus_land"]  - (State["rlds_land"]  + State["rlut_land"])
            State["rlatmnet_ocean"] = State["rlus_ocean"] - (State["rlds_ocean"] + State["rlut_ocean"])
            
            if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
    
                # save weighted averages for land and ocean components
                State["rlus"]      = weighted_average(INPUT, State["rlus_land"], State["rlus_ocean"])
                State["rlds"]      = weighted_average(INPUT, State["rlds_land"], State["rlds_ocean"])
                State["rldsnet"]   = weighted_average(INPUT, State["rldsnet_land"], State["rldsnet_ocean"])
                State["rlatmnet"]  = weighted_average(INPUT, State["rlatmnet_land"], State["rlatmnet_ocean"])
            
            
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
            
            if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
            
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
            
            
            if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
    
                # Save weighted average
                #----------------------
                
                State["shf"] = weighted_average(INPUT, State["shf_land"], State["shf_ocean"])
    
            #-----------------------------------------------------
            # Calculate the evaporation flux (in kg m^-2 s^-1) and
            # associated latent heat flux (in W m^-2)
            #-----------------------------------------------------
    
            # Over land
            #----------
            
            State["evap_flux_land"], State["lhf_evap_land"] = surface_evaporation(State["Tl"], State["Tal"], State["Q_land"], INPUT["land_height"], Var["Lv"], INPUT["r"], Var["atm_rho"], INPUT["tbhfcl"], Var["W_land"], hydro, height)
    
            # Over ocean
            #-----------
            
            State["evap_flux_ocean"], State["lhf_evap_ocean"] = surface_evaporation(State["Tos"], State["Tao"], State["Q_ocean"], Var["ocean_height"], Var["Lv"], INPUT["r"], Var["atm_rho"], INPUT["tbhfco"], Var["W_ocean"], hydro, height)
            
            if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
    
                # Save weighted average
                #----------------------
                
                State["evap_flux"] = weighted_average(INPUT, State["evap_flux_land"], State["evap_flux_ocean"])
                State["lhf_evap"]  = weighted_average(INPUT,  State["lhf_evap_land"], State["lhf_evap_ocean"])
            
            #-----------------
            # Moisture Balance
            #-----------------
    
            if (hydro == 1): # if hydrological cycle is on.
    
                # moisture balance gain due to evaporation
                State["Q_land"]  = State["Q_land"]  + (Var["secs_in_day"]/(Var["atm_rho"] * Var["atm_depth"])) * (State["evap_flux_land"])
                State["Q_ocean"] = State["Q_ocean"] + (Var["secs_in_day"]/(Var["atm_rho"] * Var["atm_depth"])) * (State["evap_flux_ocean"])
                
            
            #----------------------------------------------------------------
            # Precipitation flux (kg m^-2 s^-1), precipitation rate (in m/yr)
            #----------------------------------------------------------------
            #-------------------------------------
            # and associated heat flux (in W m^-2)
            #-------------------------------------
    
            if (hydro == 1): # if hydrological cycle is on.
                
                # Over land
                #----------
                
                State["precip_flux_land"], State["precip_rate_land"], State["lhf_precip_land"], State['Q_land'] = precipitation(Var["lat"], State["Tal"], State["Q_land"], INPUT["land_height"], Var["Lv"], INPUT["r"], Var["atm_rho"], Var["atm_depth"], Var["secs_in_day"], Var["land_mask"], Var["water_rho"], height)
                
                # Over ocean
                #-----------
                
                State["precip_flux_ocean"], State["precip_rate_ocean"], State["lhf_precip_ocean"], State['Q_ocean'] = precipitation(Var["lat"], State["Tao"], State["Q_ocean"], Var["ocean_height"], Var["Lv"], INPUT["r"], Var["atm_rho"], Var["atm_depth"], Var["secs_in_day"], Var["ocean_mask"], Var["water_rho"], height)
    
                if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
                
                    # Save weighted average
                    #----------------------
                    
                    State["precip_flux"] = weighted_average(INPUT, State["precip_flux_land"], State["precip_flux_ocean"])
                    State["precip_rate"] = weighted_average(INPUT, State["precip_rate_land"], State["precip_rate_ocean"])
                    State["lhf_precip"]  = weighted_average(INPUT, State["lhf_precip_land"], State["lhf_precip_ocean"])
                
             #-----------------
             # Moisture Balance
             #-----------------
     
            # if (hydro == 1): # if hydrological cycle is on.
     
            #      # moisture balance gain due to evaporation
            #     State["Q_land"]  = State["Q_land"]  + (Var["secs_in_day"]/(Var["atm_rho"] * Var["atm_depth"])) * (State["evap_flux_land"]-State["precip_flux_land"])
            #     State["Q_ocean"] = State["Q_ocean"] + (Var["secs_in_day"]/(Var["atm_rho"] * Var["atm_depth"])) * (State["evap_flux_ocean"]-State["precip_flux_ocean"])
                 
                 
                 
            #---------
            # Snowfall
            #---------
            
            if ( (hydro == 1) & (snow != 0) ): # if hydrological cycle is on.
                
                # Over land
                #----------
                
                State["snowfall_fraction_land"], State["snowfall_flux_land"], State['snowfall_rate_land'], State["lhf_snowfall_land"] = snowfall(Var["lat"], State["Tal"], State["precip_flux_land"], INPUT["land_height"], Var["lr"], Var["Lm"], Var['ice_rho'])
                
                # Over ocean
                #-----------
                
                State["snowfall_fraction_ocean"], State["snowfall_flux_ocean"], State['snowfall_rate_ocean'], State["lhf_snowfall_ocean"] = snowfall(Var["lat"], State["Tao"], State["precip_flux_ocean"], Var["ocean_height"], Var["lr"], Var["Lm"], Var['ice_rho'])
                
                if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
            
                    # save weighted average
                    State["snowfall_fraction"] = weighted_average(INPUT, State["snowfall_fraction_land"], State["snowfall_fraction_ocean"])
                    State["snowfall_flux"]     = weighted_average(INPUT, State["snowfall_flux_land"], State["snowfall_flux_ocean"])
                    State["snowfall_rate"]     = weighted_average(INPUT, State["snowfall_rate_land"], State["snowfall_rate_ocean"])
                    State["lhf_snowfall"]      = weighted_average(INPUT, State["lhf_snowfall_land"], State["lhf_snowfall_ocean"])
             
            #----------------------------
            # Snow accumulation over land
            #----------------------------
            
            if fixed_snow_fraction is None and fixed_snow_thick is None and fixed_snow_melt is None: # snow is calculated internally...
            
                if ( (hydro == 1) & (snow != 0) ): # if hydrological cycle is on.
                    
                    State["snow_fraction_land"], State["snow_thick"] = snow_accumulation(State["snowfall_fraction_land"], State["snow_fraction_land"], State["snow_thick"], State["snowfall_rate_land"], State["precip_flux_land"], Var["sarea"], Var['ice_rho'], Var["secs_in_day"], snow)
                
                    # Calculate equatorward edge of snow in each hemisphere
                    State["sc_edge_nh"], State["sc_edge_sh"] = snow_edge(State["snow_fraction_land"], Var)
                    
            else: # snow is prescribed externally (for sensitivity experiments)
            
                State['snow_fraction_land'] = fixed_snow_fraction[:,day]
                
                State['snow_thick'] = fixed_snow_thick[:,day]
                
                State['melt'] = fixed_snow_melt[:,day]
                
                
            #--------------------
            # Atmospheric Heating
            #--------------------
    
            if (hydro == 1): # if hydrological cycle is on.
    
                State["Tal"] = State["Tal"] + Var["secs_in_day"] / Var["atm_hc"]  * (State["ratmnet_land"] + State["shf_land"] + State["lhf_precip_land"] + State["lhf_snowfall_land"])
    
                State["Tao"] = State["Tao"] + Var["secs_in_day"] / Var["atm_hc"]  * (State["ratmnet_ocean"] + State["shf_ocean"] + State["lhf_precip_ocean"] + State["lhf_snowfall_ocean"])
    
    
            if (hydro == 0): # if hydrological cycle is off.
    
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
    
            #-----------------------------------------------------------------------
            # Surface melting and surface heat modifications (for snow covered land)
            #-----------------------------------------------------------------------
            
            if fixed_snow_fraction is None and fixed_snow_thick is None and fixed_snow_melt is None: # snow is calculated internally...
            
                if ( (hydro == 1) & (snow != 0) ): # if hydrological cycle is on.
                    
                    State["snow_thick"], State["snow_fraction_land"], State["Tl"], State["melt"], State["melt_flux"]  = snow_melt(Var["lat"], State["snow_thick"], State["snow_fraction_land"], State["Tl"], Var["lr"], INPUT["land_height"], Var["K"], Var["land_hc"], Var["Lm"], 
                                  Var["ice_rho"], Var["land_mask"], Var["secs_in_day"], Var["sarea"], snow)
                    
            else: # snow is prescribed externally (for sensitivity experiments)
            
                pass
    
            
            #-----------------------------------------------------------------------.
            # Weighted average atmospheric temperature and humidities (for transport)
            #-----------------------------------------------------------------------.
            
            State["Ta"]  = weighted_average(INPUT, State["Tal"], State["Tao"])
            State['Tao'] = State["Ta"].copy()
            State['Tal'] = State["Ta"].copy()   
                
            if (hydro == 1): # if hydrological cycle is on.
                
                State["Q"] = weighted_average(INPUT, State["Q_land"], State["Q_ocean"])
                State['Q_land']  = State["Q"].copy()
                State['Q_ocean'] = State["Q"].copy()
                    
            #---------------------------------------
            # Meridional heat and moisture transport
            #---------------------------------------
                
            if transport==1: # if transport is on.
            
                # define time step
                if INPUT['res']==1.:
                    small_step = 50
                    
                if INPUT['res']==2.5:
                    small_step = 3
                    
                if INPUT['res']==5.:
                    small_step = 1
                    
                # initialize arrays for heat transport
                if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
                
                    if version == 0: # if classic 'dry' version of EBM 
                    
                        # initialize atmopsheric transport arrays
                        State["mse_north"]         = np.zeros((Var["latb"].size))
                        State["mse_conv"]          = np.zeros((Var["lat"].size))
                    
                    if version == 1: # if moist version of EBM
                
                        # initialize atmopsheric transport arrays
                        State["mse_north"]         = np.zeros((Var["latb"].size))
                        State["mse_conv"]          = np.zeros((Var["lat"].size))
                        State["dry_north"]         = np.zeros((Var["latb"].size))
                        State["hadley_dry"]        = np.zeros((Var["latb"].size))
                        State["hadley_latent"]     = np.zeros((Var["latb"].size))
                        State["eddy_dry"]          = np.zeros((Var["latb"].size))
                        State["eddy_latent"]       = np.zeros((Var["latb"].size))
                        State["dry_north"]         = np.zeros((Var["latb"].size))
                        State["dry_conv"]          = np.zeros((Var["lat"].size))
                        State["latent_north"]      = np.zeros((Var["latb"].size))
                        State["latent_conv"]       = np.zeros((Var["lat"].size))
             
                    # intialize ocean transport arrays
                    State["advf"]       = np.zeros((Var["olatb"].size*6)) 
                    State["advf_conv"]  = np.zeros((Var["olat"].size*6))
                    State["hdiffs"]      = np.zeros((Var["olatb"].size*6))
                    State["hdiffs_conv"] = np.zeros((Var["olat"].size*6)) 
                    State["hdiffi"]      = np.zeros((Var["olatb"].size*6)) 
                    State["hdiffi_conv"] = np.zeros((Var["olat"].size*6)) 
                    State["vdiff"]       = np.zeros((Var["olat"].size*7)) 
                    State["vdiff_conv"]  = np.zeros((Var["olat"].size*6)) 
                    State["vadvf"]       = np.zeros((Var["olat"].size*7)) 
                    State["vadvf_conv"]  = np.zeros((Var["olat"].size*6)) 
                
        
                for n in np.arange(0, small_step): 
                    
                    # atmospheric heat transport
                    #---------------------------
            
                    if atm_transport==1: # if atmospheric heat transport turned on.
                    
                        if version == 0: # if classic 'dry' version of EBM 
                        
                            mse_north, mse_conv = dry_heat_transport(State['Ta'], np.zeros((Var["lat"].size)), Var['dlatr'], Var['swidth'], Var['atm_hc'], INPUT['dt'], Var['r_earth'], Var['sarea'])
                            
                            # atmospheric heating/cooling
                            State["Ta"] = State["Ta"] + (Var["secs_in_day"]/small_step) / Var["atm_hc"]  * (mse_conv)
                            
                        if version == 1: # if moist version of EBM   
                    
                            mse_north, mse_conv, hadley_latent, hadley_dry, eddy_latent, eddy_dry, dry_north, dry_conv, latent_north, latent_conv = moist_heat_transport_v5(Var, State, INPUT, hadley_cell, hydro, height)
                        
                            if hydro==0: # if hydrological cycle turned off.
                   
                                # atmospheric heating/cooling
                                State["Ta"] = State["Ta"] + (Var["secs_in_day"]/small_step) / Var["atm_hc"]  * (mse_conv)
                            
                            if hydro==1: # if hydrological cycle turned on.
                        
                                # atmospheric heating/cooling                        
                                State["Ta"] = State["Ta"] + (Var["secs_in_day"]/small_step) / Var["atm_hc"]  * (dry_conv)
                                    
                                # atmospheric moistening/drying
                                State["Q"]  = State["Q"]  + (Var["secs_in_day"]/small_step) / (Var["atm_rho"] * Var["atm_depth"]) * (latent_conv/Var["Lv"])
                                    
        
                    # ocean heat transport
                    #---------------------
                    
                    if ocn_transport==1: # if ocean heat transport turned on.
                    
                        # ocean (advective and diffusive) heat transport                   
                        vadvf, vadvf_conv, hdiffi, hdiffi_conv, vdiff, vdiff_conv, hdiffs, hdiffs_conv, advf, advf_conv = ocean_fluxes(State, Var, INPUT)
                    
                        if fixed_ocean_transport is None: # ocean heat transport calculated internally
                        
                            # ocean heating/cooling 
                            State["To"][Var["idxoc"].astype('int')] = State["To"][Var["idxoc"].astype('int')] + ( (Var["secs_in_day"]/small_step)/Var["ocean_hc"] * (advf_conv + vadvf_conv + hdiffs_conv + hdiffi_conv + vdiff_conv) )
                        
                        else: # ocean heat transport prescribed externally
                            
                            # ocean heating/cooling 
                            State["To"][Var["idxoc"].astype('int')] = State["To"][Var["idxoc"].astype('int')] + ( (Var["secs_in_day"]/small_step)/Var["ocean_hc"] * (fixed_ocean_transport[:,day]) )
                        
                    
                    if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
                        
                        if atm_transport==1: # if atmospheric heat transport turned on.
                        
                            if version == 0: # if classic 'dry' version of EBM 
                            
                                # Sum atmospheric transport values over small time step
                                State["mse_north"]  = State["mse_north"] + mse_north
                                State["mse_conv"]   = State["mse_conv"] + mse_conv
                            
                            if version == 1: # if moist version of EBM 
                    
                                # Sum atmospheric transport values over small time step
                                State["mse_north"]         = State["mse_north"] + mse_north
                                State["mse_conv"]          = State["mse_conv"] + mse_conv
                                State["hadley_dry"]        = State["hadley_dry"] + hadley_dry
                                State["hadley_latent"]     = State["hadley_latent"] + hadley_latent
                                State["eddy_dry"]          = State["eddy_dry"] + eddy_dry
                                State["eddy_latent"]       = State["eddy_latent"] + eddy_latent
                                State["dry_north"]         = State["dry_north"] + dry_north
                                State["dry_conv"]          = State["dry_conv"] + dry_conv
                                State["latent_north"]      = State["latent_north"] + latent_north
                                State["latent_conv"]       = State["latent_conv"] + latent_conv
                    
                        if ocn_transport==1: # if ocean heat transport turned on.
                        
                            # Sum ocean transport values over small time step
                            State["advf"]        = State["advf"]  + advf
                            State["advf_conv"]   = State["advf_conv"] + advf_conv
                            State["vadvf"]       = State["vadvf"] + vadvf
                            State["vadvf_conv"]  = State["vadvf_conv"] + vadvf_conv
                            State["hdiffi"]      = State["hdiffi"] + hdiffi
                            State["hdiffi_conv"] = State["hdiffi_conv"] + hdiffi_conv
                            State["vdiff"]       = State["vdiff"] + vdiff
                            State["vdiff_conv"]  = State["vdiff_conv"] + vdiff_conv
                            State["hdiffs"]      = State["hdiffs"] + hdiffs
                            State["hdiffs_conv"] = State["hdiffs_conv"] + hdiffs_conv
                            
                        
                if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
                    
                    if atm_transport==1: # if atmospheric heat transport turned on.
                    
                        if version == 0: # if classic 'dry' version of EBM 
                        
                            # Average atmospheric transport values over small time step
                            State["mse_north"]         = State["mse_north"]        /  small_step
                            State["mse_conv"]          = State["mse_conv"]         /  small_step
                        
                        if version == 1: # if moist version of EBM 
        
                            # Average atmospheric transport values over small time step
                            State["mse_north"]         = State["mse_north"]        /  small_step
                            State["mse_conv"]          = State["mse_conv"]         /  small_step
                            State["hadley_dry"]        = State["hadley_dry"]       /  small_step
                            State["hadley_latent"]     = State["hadley_latent"]    /  small_step
                            State["eddy_dry"]          = State["eddy_dry"]         /  small_step
                            State["eddy_latent"]       = State["eddy_latent"]      /  small_step
                            State["dry_north"]         = State["dry_north"]        /  small_step
                            State["dry_conv"]          = State["dry_conv"]         /  small_step
                            State["latent_north"]      = State["latent_north"]     /  small_step
                            State["latent_conv"]       = State["latent_conv"]      /  small_step
                    
                    if ocn_transport==1: # if ocean heat transport turned on.
                    
                        # Average ocean transport values over small time step
                        State["advf"]       = State["advf"]  /  small_step
                        State["advf_conv"]  = State["advf_conv"] /  small_step
                        State["vadvf"]       = State["vadvf"] /  small_step
                        State["vadvf_conv"]  = State["vadvf_conv"] /  small_step
                        State["hdiffi"]      = State["hdiffi"] /  small_step
                        State["hdiffi_conv"] = State["hdiffi_conv"] /  small_step
                        State["vdiff"]       = State["vdiff"] /  small_step
                        State["vdiff_conv"]  = State["vdiff_conv"] /  small_step
                        State["hdiffs"]      = State["hdiffs"] /  small_step
                        State["hdiffs_conv"] = State["hdiffs_conv"] /  small_step
                        State["ocean_conv"]  = (
                                                State["advf_conv"] +
                                                State["vadvf_conv"] +
                                                State["hdiffi_conv"] +
                                                State["vdiff_conv"] +
                                                State["hdiffs_conv"]
                                                )
                    
            #---------------------------------
            # Calculate mean ocean temperature
            #---------------------------------
            
            State["mot"] = np.array([mean_ocean_temperature(State["To"], Var)])

            #----------------------------------------------
            # Zonal exchange of heat between land and ocean
            #----------------------------------------------
            
            State["Tao"] = State["Ta"].copy()
            State["Tal"] = State["Ta"].copy()  
                    
            if hydro == 1:
                State["Q_ocean"] = State["Q"].copy()
                State["Q_land"]  = State["Q"].copy()
                    

            #------------------------------
            # Sea ice formation and melting
            #------------------------------
            
            if fixed_si_fraction is None and fixed_si_thick is None and fixed_si_melt is None: # snow is calculated internally...
            
                if (seaice == 1): # if sea ice is on.
                    
                    # calculate changes in sea-ice cover
                    State["To"][Var["idxolyr1"].astype('i8')], State["Tsi"], State["si_thick"], State["si_volume"], State["si_fraction"], State["si_melt_flux"] = sea_ice(State["To"][Var["idxolyr1"].astype('i8')], State["Tsi"], State["si_volume"], State["snowfall_flux_ocean"], State["si_fraction"], Var["lat"], Var["idxal"].astype('i8'), Var["ice_rho"], Var["Lm"],
                                Var["Ksi"], Var["ocean_hc1"], Var["oarea"], Var["secs_in_day"], Var["sea_ice_sphc"], Var["si_ithick"])
        
                    # calculate areal extent of sea-ice in each hemisphere
                    State["si_area_nh"][0] = np.nansum(State["si_fraction"][Var["idxnh"].astype('i8')]*Var["oarea"][Var["idxnh"].astype('i8')])
                    State["si_area_sh"][0] = np.nansum(State["si_fraction"][Var["idxsh"].astype('i8')]*Var["oarea"][Var["idxsh"].astype('i8')])
                    
                    # Calculate equatorward edge of sea ice in each hemisphere
                    State["si_edge_nh"], State["si_edge_sh"] = snow_edge(State["si_fraction"], Var)
                    
            else: # snow is prescribed externally (for sensitivity experiments)
            
                State['si_fraction'] = fixed_si_fraction[:,day]
                
                State['si_thick'] = fixed_si_thick[:,day]
                
                State['si_melt_flux'] = fixed_si_melt[:,day]
            
    
            #--------------------------
            # Surface ocean temperature
            #--------------------------
    
            # Set ocean surface temperatures
            State["Tos"] = np.copy(State["To"][Var["idxolyr1"].astype('i8')])
    
            if (seaice == 1): # if sea ice is on.
    
                # Set to sea-ice temperature when ocean totally covered by sea-ice.
                i = np.where(State["si_fraction"] == 1)
                State["Tos"][i] = np.copy(State["Tsi"][i])
                
            #------------------------------------------
            # Calculate the average surface temperature
            #------------------------------------------
            
            State["Ts"] = weighted_average(INPUT, State["Tl"], State["Tos"])
            
            #------------------------
            # Global mean temperature
            #------------------------
            
            State['Tagx'][0] = global_nanmean(State["Ta"] - (Var["lr"] * Var["mean_height"]), Var)
            State['Tsgx'][0] = global_nanmean(State["Ts"] - (Var["lr"] * Var["mean_height"]), Var)
            State['Tosg'][0] = global_nanmean(State["Tos"], Var)
            
            #-------------------------
            # Lapse rate modifications
            #-------------------------
            
            State['Tax']  = State["Ta"]   - (Var["lr"] * Var["mean_height"])
            State['Tlx']  = State["Tl"]  - (Var["lr"] * INPUT["land_height"])
           
            
            
            """
            Save key state variables for every model year.
            """
            
            for keys in key_variables:
                
                StateAnnual[keys][year,day,:] = State[keys]
                
            
            """
            Save key state variables for last year.
            """
            if year == INPUT['nyrs'].astype('i8')[0] - 1: # if last year of model run.
            
            
                # save every model variable
                #--------------------------
                
                for keys, values in State.items(): # for every variable in State
                
                    try:
                
                        StateYear[keys][:,day] = State[keys]
                    
                    except:
                        
                        pass
                
            
            
        # changes in ocean circulation
        #-----------------------------
    
        # if variable_overturning == 1: # if ocean overturning changes
            
        #     # calculate change in ocean overturning
        #     State["ww"], State["dz"], State["drn"], State["drs"] = ocean_overturning_strength(StateYear["To"], Var["lat"], Var["olat"], Var["oarea"], Var["dlat"], State["idxcos"], Var["gamma"], Var["drn0"],
        #                                     Var["drs0"], Var["zeta"], Var["ton0"], Var["tos0"], Var["w0"], INPUT["dz0"], Var)
     
        #     # calculate subsequent changes in horizontal velcoities
        #     State["u1"], State["u2"], State["ww"] = horizontal_ocean_velocities(Var["olat"], Var["olatb"], Var["olatr"], Var["olatbr"], Var["dlatr"], State["idxcos"], np.array([80.]), np.array([-70.]), np.zeros((Var["olat"].size))+0.7, np.zeros((Var["olatb"].size))+0.7, Var["r_earth"], State["ww"], Var["odepth"])

            
        # if variable_midpoint == 1: # if changes in centre of ocean circulation
            
        #     State["idxcos"] = midpoint(StateYear["To"], Var["lat"], Var["olat"], Var["oarea"], Var["dlat"], State["idxcos"], Var["gamma"], Var["drn0"],
        #                                     Var["drs0"], Var["zeta"], Var["ton0"], Var["tos0"], Var["w0"], INPUT["dz0"], Var)
                    
                    
        
    return State, StateYear, StateAnnual

    
         