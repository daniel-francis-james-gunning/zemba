# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

Calculation  of standard pre-industrial surface water density difference 
between the equatorial and polar (60–80◦ N and 50–70◦ S) regions. 

Used for varying the strength of thermohaline circulation (see Bintanja 1996 
and Stap 2014). 
"""

import numpy as np
import xarray as xr
import os
from numba import njit
from numba.typed import Dict
from numba.core import types

# load NorESM2 surface ocean temperature
#---------------------------------------

# load noresm
cdir    = os.getcwd()
pdir    = os.path.dirname(cdir)
os.chdir(pdir)
NorESM2 = xr.open_dataset(pdir+"/comparative_data/noresm2/monthly/noresm2_annual.nc")

# load surface ocean temperature
lat = NorESM2["lat"].to_numpy()  # latitude
tos = (NorESM2["ts"]*NorESM2["ocean_mask"]).mean(dim="lon").to_numpy() # surface ocean temp

# correct for sea-ice effect
i = np.where(tos < 271.15)
tos[i] = 271.15 # set temp below freezing to freezing point


# calculate surface ocean area
#-----------------------------

# total surface area
sarea =  (
                # east-west width of latitude band
                (2*np.pi*6371000.*np.cos(np.deg2rad(lat))) 
                * 
                # north-south length of latitude band 
                (6371000.*np.deg2rad(1))
                ) 

# land fraction (from noresm)
lf = np.asarray( NorESM2["land_mask"].sum(dim="lon").to_numpy() / (NorESM2["lon"].size), dtype = 'f8' )

# surface ocean area
oarea = sarea * (1 - lf)

# average surface ocean temperature in NH polar water (60-80N)
#-------------------------------------------------------------

# index for 60N
i60N  = np.where(lat == 60.5)[0][0]

# index for 80N
i80N  = np.where(lat == 79.5)[0][0] 

# area-weighted average temperature
tosN = np.sum(tos[i60N:i80N+1]*oarea[i60N:i80N+1]) / np.sum(oarea[i60N:i80N+1])


# average surface ocean temperature in SH polar water (70-50S)
#-------------------------------------------------------------

# index for 70S
i70S  = np.where(lat == -69.5)[0][0]

# index for 50S
i50S  = np.where(lat == -50.5)[0][0] 

# area-weighted average temperature
tosS  = np.sum(tos[i70S:i50S+1]*oarea[i70S:i50S+1]) / np.sum(oarea[i70S:i50S+1])


# average surface ocean temperature at equator (north of 5S)
#-----------------------------------------------------------

# index for 4S
i4S    = np.where(lat == -4.5)[0][0] 

# area-weighted average temperature
toseqN = tos[i4S]

# average surface ocean temperature at equator (south of 5S)
#-----------------------------------------------------------

i6S    = np.where(lat == -5.5)[0][0] # index for 4S
toseqS = tos[i6S]

# function to calculate surface ocean density at 35 PSU (from Gill, 1982)
#------------------------------------------------------------------------

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

# calculate density
#------------------

# northern hemisphere polar water
rho_n = calculate_density(tosN - 273.15)

# northern hemisphere equator
rho_eqn = calculate_density(toseqN - 273.15)

# southern hemisphere polar water
rho_s = calculate_density(tosS - 273.15)

# southern hemisphere equator
rho_eqs = calculate_density(toseqS - 273.15)

# density difference in NH
drn0 = rho_n - rho_eqn

# density difference in SH
drs0 = rho_s - rho_eqs





@njit(nogil = True, cache=True)
def ocean_overturning_strength(to, lat, olat, oarea, dlat, idxcos, gamma, drn0,
                               drs0, zeta, tosN0, tosS0, w0, dz0, Var):
    
    
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






# # calculate change in ocean overturning during colder conditions
# #---------------------------------------------------------------

# # load modules
# from initialize_test2 import *
# from ebm_test2 import ocean_overturning_strength
# Var, State = initialize(topography = "PI", ocean_centre=-5., resolution = 1., nyrs= 2000.) # set to PI topography

# # assume cooling in northern hemisphere polar 
# tosN_LGM   = 271.15

# # assume same in equator
# toseqN_LGM = toseqN

# # calculate density
# rho_n_LGM  = calculate_density(tosN_LGM - 273.15) # just temperature gradient
# rho_n_LGM2 = rho_n_LGM - 0.09 * (tosN_LGM-tosN)

# # calculate density difference in the NH
# drn =  rho_n_LGM - rho_eqn
# drn2 = rho_n_LGM2 - rho_eqn

# # calculate change in overturning rate -- just temperature gradient
# ww1 = np.zeros((Var["olat"].size))
# i = np.where(Var["olat"]==-0.5)[0][0]
# zeta = 6.
# ww1[i:ww1.size] = Var["w0"][i:ww1.size]*(1.+zeta*(drn/drn0-1.))

# # calculate change in overturning rate -- just temperature gradient and freshwater term
# ww2             = np.zeros((Var["olat"].size))
# i               = np.where(Var["olat"]==-0.5)[0][0]
# zeta            = 6.
# ww2[i:ww2.size] = Var["w0"][i:ww2.size]*(1.+zeta*(drn2/drn0-1.))



# drn=drn+(11.2003*(tsnoord-to1(m))-.034805*(tsnoord**2&
#      	-to1(m)**2)+3.46761e-5*(tsnoord**3-to1(m)**3)&
#      	+gamma*(tsnoord-tsnoordstan))



# np.sum(Var["w0"][20:69+1]*Var["sarea"][40:89+1])/np.sum(Var["sarea"][40:89+1]) * (60*60*24*365)


# np.sum(Var["w0"][0:19+1]*Var["sarea"][20:39+1])/np.sum(Var["sarea"][20:39+1]) * (60*60*24*365)


# np.sum(Var["w0"][130:149]*Var["sarea"][150:169])/np.sum(Var["sarea"][150:169]) * (60*60*24*365)




# lf = Var["ocean_fraction_bounds"][20:171]

# ow = (2*np.pi*Var["r_earth"]*(lf)*np.cos(Var["olatbr"]))

# np.sum(Var["w0"][0:19+1]*Var["sarea"][20:39+1]*lf[0:19+1]) / 1e6
# np.sum(Var["w0"][0:19+1]*Var["sarea"][20:39+1]*0.7) / 1e6

# np.sum(Var["w0"][130:149]*Var["sarea"][150:169]*lf[130:149]) / 1e6
# np.sum(Var["w0"][130:149]*Var["sarea"][150:169]*0.7) / 1e6

# ow = (2*np.pi*Var["r_earth"]*f*np.cos(Var["olatbr"]))

# Var["ow"]




# f = 0.7
# r_earth = Var["r_earth"]
# dlatr = Var["dlatr"]
# w = Var["w0"]
# olatbr = Var["olatbr"]
# olatr = Var["olatr"]
# odepth = Var["odepth"]

# # index for centre of ocean circulation
# icoc = np.where(olatb == -5.)[0][0]

# # index for northern boundary
# inb = np.where(olatb == 80.)[0][0]

# # index for southern boundary
# isb = np.where(olatb == -70.)[0][0]

# u2 = np.zeros((olatb.size))
# u3 = np.zeros((olatb.size))

# for i in np.arange(icoc, inb-1):
    
#     u2[i+1] = ((
        
#               (f*np.cos(olatr[i])*r_earth[0]*dlatr[0]*(w[i]/odepth[0]))
              
#                 +
              
#                 (f*np.cos(olatbr[i])*u2[i])
        
#                 )
        
#                 /
              
#                 (f*np.cos(olatbr[i+1]))
              
#                 )
    
#     u3[i+1] = ((
        
#                 (f*np.cos(olatbr[i])*odepth[0]*u3[i])
                
#                 +
                
#                 (f*np.cos(olatr[i])*r_earth*dlatr[0]*w[i])
                
#                 )
        
#                 /
                
#                 (f*np.cos(olatbr[i+1])*odepth[0])
                
#                 )
        
        
        
        
        
                








