# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

Python script containing some useful non-specific functions
"""

import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange

#------------------------------------------------------------------------------
# Function which returns the RMSE between the energy balance model and the 
# NorESM piControl simulation
#------------------------------------------------------------------------------ 
    
@njit(nogil = True)
def RMSE(predicted, observed):
        
    N    = predicted.size # Number of data points
    
    MSE  = np.sum((observed - predicted)**2)/N # Mean square error
    
    RMSE = np.sqrt(MSE) # Root mean sqaure error
    
    return RMSE

def global_pymean(variable, Var):
    
    #--------------------------------------------------------------------------
    # Calculates the global average of a variable by weighing for the area of 
    # the latitudinal band
    #--------------------------------------------------------------------------
    
    # Area of each latitude band
    area = Var["sarea"]
    
    # Total surface area
    total_area = np.sum(area)
    
    # Weighted average of variable
    variable_weighted = np.sum(variable*area)/total_area
    
    return variable_weighted 

@njit(nogil = True)
def global_mean(variable, Var):
    
    #--------------------------------------------------------------------------
    # Calculates the global average of a variable by weighing for the area of 
    # the latitudinal band
    #--------------------------------------------------------------------------
    
    # Area of each latitude band
    area = Var["sarea"]
    
    # Total surface area
    total_area = np.sum(area)
    
    # Weighted average of variable
    variable_weighted = np.sum(variable*area)/total_area
    
    return variable_weighted 

@njit(nogil = True)
def global_mean2(variable, lat, dlat):
    
    #--------------------------------------------------------------------------
    # Calculates the global average of a variable by weighing for the area of 
    # the latitudinal band
    #--------------------------------------------------------------------------
    
    # latitude in radians
    latr = np.deg2rad(lat)
    dlatr = np.deg2rad(dlat)
    
    # surface area for latitude 'band'
    area = 2*np.pi*(6371000**2)*np.cos(latr)*dlatr
    
    # Total surface area
    total_area = np.sum(area)
    
    # Weighted average of variable
    variable_weighted = np.sum(variable*area)/total_area
    
    return variable_weighted 

@njit(nogil = True)
def mean_numba(a):

    res = []
    for i in prange(a.shape[0]):
        res.append(a[i, :].mean())

    return np.array(res)


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


def calculate_monthlies(x, lat):
    
    '''
    Returns monthly values for StateYear variable.
    '''

    # initialize monthly array
    monthly_array = np.zeros((lat, 12))

    # days in months
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # months
    monthly_array[:,0]  = x[:, 0:31].mean(axis=1)       # jan
    monthly_array[:,1]  = x[:, 31:59].mean(axis=1)      # feb
    monthly_array[:,2]  = x[:, 59:90].mean(axis=1)      # mar
    monthly_array[:,3]  = x[:, 90:120].mean(axis=1)     # apr
    monthly_array[:,4]  = x[:, 120:151].mean(axis=1)    # may
    monthly_array[:,5]  = x[:, 151:181].mean(axis=1)    # jun
    monthly_array[:,6]  = x[:, 181:212].mean(axis=1)    # jul
    monthly_array[:,7]  = x[:, 212:243].mean(axis=1)    # aug
    monthly_array[:,8]  = x[:, 243:273].mean(axis=1)    # sep
    monthly_array[:,9]  = x[:, 273:304].mean(axis=1)    # oct
    monthly_array[:,10] = x[:, 304:334].mean(axis=1)    # nov
    monthly_array[:,11] = x[:, 334:365].mean(axis=1)    # dec

    return monthly_array


def spatial_average(data, xlatitude, xlatitude_spacing, fraction):
    
    '''
    '''
    
    # constants
    #----------
    
    xlatitude_radians = np.deg2rad(xlatitude)
    
    xlatitude_radians_spacing = np.deg2rad(xlatitude_spacing)
    
    radius_of_earth = 6371000.
    
    surface_area =  (
                    # east-west width of latitude band
                    (2*np.pi*radius_of_earth*fraction*np.cos(xlatitude_radians)) 
                    * 
                    # north-south length of latitude band 
                    (radius_of_earth*xlatitude_radians_spacing)
                    ) 
    
    # indexes
    #--------
    
    i90S  = np.where(xlatitude == -90. + xlatitude_spacing/2)[0][0]
    i60Ss = np.where(xlatitude == -60. - xlatitude_spacing/2)[0][0]
    i60Sn = np.where(xlatitude == -60. + xlatitude_spacing/2)[0][0]
    i30Ss = np.where(xlatitude == -30. - xlatitude_spacing/2)[0][0]
    i30Sn = np.where(xlatitude == -30. + xlatitude_spacing/2)[0][0]
    i0s   = np.where(xlatitude == 0. - xlatitude_spacing/2)[0][0]
    i0n   = np.where(xlatitude == 0. + xlatitude_spacing/2)[0][0]
    i30Ns = np.where(xlatitude == 30. - xlatitude_spacing/2)[0][0]
    i30Nn = np.where(xlatitude == 30. + xlatitude_spacing/2)[0][0]
    i60Ns = np.where(xlatitude == 60. - xlatitude_spacing/2)[0][0]
    i60Nn = np.where(xlatitude == 60. + xlatitude_spacing/2)[0][0]
    i90N  = np.where(xlatitude == 90. - xlatitude_spacing/2)[0][0]
    
    # averaging
    #----------
    
    zones = []
    
    # 90S to 60S
    zones.append(np.round(np.nansum(data[i90S:i60Ss+1]*surface_area[i90S:i60Ss+1])/np.nansum(surface_area[i90S:i60Ss+1]),2))
    
    # 60S to 30S
    zones.append(np.round(np.nansum(data[i60Sn:i30Ss+1]*surface_area[i60Sn:i30Ss+1])/np.nansum(surface_area[i60Sn:i30Ss+1]),2))
    
    # 30S to 0
    zones.append(np.round(np.nansum(data[i30Sn:i0s+1]*surface_area[i30Sn:i0s+1])/np.nansum(surface_area[i30Sn:i0s+1]),2))
    
    # 0 to 30N
    zones.append(np.round(np.nansum(data[i0n:i30Ns+1]*surface_area[i0n:i30Ns+1])/np.nansum(surface_area[i0n:i30Ns+1]),2))
    
    # 30N to 60N
    zones.append(np.round(np.nansum(data[i30Nn:i60Ns+1]*surface_area[i30Nn:i60Ns+1])/np.nansum(surface_area[i30Nn:i60Ns+1]),2))
    
    # 60N to 90N
    zones.append(np.round(np.nansum(data[i60Nn:i90N+1]*surface_area[i60Nn:i90N+1])/np.nansum(surface_area[i60Nn:i90N+1]),2))
    
    return np.array(zones)
    
    







    
