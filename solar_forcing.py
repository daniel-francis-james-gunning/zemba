# -*- coding: utf-8 -*-

'''
@author: Daniel Gunning (University of Bergen)

Python script for importing orbital parameters from the last 5 million years
based on the solution of Laskar et al., (2004).

Main reference:
Laskar, J., Gastineau, M., Joutel, F., Robutel, P., Levrard, B., Correia, A.,: 
2004, A long term numerical solution for the insolation quantities of Earth.
{\it in preparation}

*******************************************************************************
*  Authors: J. Laskar, M. Gastineau, F. Joutel                                *
*  (c) Astronomie et Systemes Dynamiques, Institut de Mecanique Celeste,      *
*      Paris (2004)                                                           *
*                                                                             *
*                               Jacques Laskar                                *
*                               Astronomie et Systemes Dynamiques,            *
*                               Institut de Mecanique Celeste                 *
*                               77 av. Denfert-Rochereau                      *
*                               75014 Paris                                   *
*                       email:  laskar@imcce.fr                               *
*                                                                             *
*******************************************************************************

Daily insolation values calculated based on the matlab code of 
daily_insolation.m from Ian Eisenman and Peter Huybers, Harvard University, 
August 2006:

Available online at http://eisenman.ucsd.edu/code
'''

#------------------------------------------------------------------------------
# Load modules
#------------------------------------------------------------------------------

import numpy as np
import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
# Function to retrieve orbital parameters from the Laskar 2004 text file 
# containing eccentricity, longitude of perihelion and obliquity
#------------------------------------------------------------------------------

def import_orbital_parameters(kyear):
    
    if (kyear < 0 | kyear > 51000):
        
        raise Exception("Kyear must reside between 0 and 51,000")
    
    else:
        
        
        # thousand of years before present
        kyr_data          = np.loadtxt('laskar_2006.txt', 
                                       dtype = str, 
                                       usecols = 0)
        
        # eccentricity
        eccentricity_data = np.loadtxt('laskar_2006.txt', 
                                       dtype = str, 
                                       usecols = 1)
        
        # obliquity 
        obliquity_data    = np.loadtxt('laskar_2006.txt', 
                                       dtype = str, 
                                       usecols = 2)
        
        # longitude of perihelion
        omega_data        = np.loadtxt('laskar_2006.txt', 
                                       dtype = str, 
                                       usecols = 3)
        
        # Covert fortran scientific notation (e.g., D-01) to python scientific
        # notation (e.g.,E-01 )
        
        eccentricity = float(eccentricity_data[kyear].replace('D', 'E'))
        
        obliquity    = float(obliquity_data[kyear].replace('D', 'E'))
        
        omega        = float(omega_data[kyear].replace('D', 'E'))
        
        # Add 180 degrees (pi) to omega (see lambda definition, Berger 1978 Appendix)
        omega = omega + np.pi
        
        # return dictionary of orbital parameters
        orb = {}
        orb["ecc"] = eccentricity
        orb["obl"] = obliquity
        orb["pre"] = omega
        
        return orb 
    
    
def import_orbital_parameters_v2(kyr):
    

        
    if kyr.is_integer() == True:
    
        # thousand of years before present
        kyr_data          = np.loadtxt('laskar_2006.txt', 
                                       dtype = str, 
                                       usecols = 0)
        
        # eccentricity
        eccentricity_data = np.loadtxt('laskar_2006.txt', 
                                       dtype = str, 
                                       usecols = 1)
        
        # obliquity 
        obliquity_data    = np.loadtxt('laskar_2006.txt', 
                                       dtype = str, 
                                       usecols = 2)
        
        # longitude of perihelion
        omega_data        = np.loadtxt('laskar_2006.txt', 
                                       dtype = str, 
                                       usecols = 3)
        
        # Covert fortran scientific notation (e.g., D-01) to python scientific
        # notation (e.g.,E-01 )
        
        eccentricity = float(eccentricity_data[int(kyr)].replace('D', 'E'))
        
        obliquity    = float(obliquity_data[int(kyr)].replace('D', 'E'))
        
        omega        = float(omega_data[int(kyr)].replace('D', 'E'))
        
        # Add 180 degrees (pi) to omega (see lambda definition, Berger 1978 Appendix)
        omega = omega + np.pi
        
        # return dictionary of orbital parameters
        orb = {}
        orb["ecc"] = eccentricity
        orb["obl"] = obliquity
        orb["pre"] = omega
        
        return orb 
        
    elif kyr.is_integer() == False:
        
        kyr_bf = kyr - 0.5
        kyr_af = kyr + 0.5
        
        # eccentricity
        eccentricity_data = np.loadtxt('laskar_2006.txt', dtype = str, usecols = 1)
        obliquity_data    = np.loadtxt('laskar_2006.txt', dtype = str, usecols = 2)
        omega_data        = np.loadtxt('laskar_2006.txt', dtype = str, usecols = 3)
        
        # Covert fortran scientific notation (e.g., D-01) to python scientific
        # notation (e.g.,E-01 )
        
        eccentricity_bf = float(eccentricity_data[int(kyr_bf)].replace('D', 'E'))
        obliquity_bf    = float(obliquity_data[int(kyr_bf)].replace('D', 'E'))
        omega_bf        = float(omega_data[int(kyr_bf)].replace('D', 'E'))
        omega_bf = omega_bf + np.pi
        
        eccentricity_af = float(eccentricity_data[int(kyr_af)].replace('D', 'E'))
        obliquity_af    = float(obliquity_data[int(kyr_af)].replace('D', 'E'))
        omega_af        = float(omega_data[int(kyr_af)].replace('D', 'E'))
        omega_af = omega_af + np.pi
        
        eccentricity   = (eccentricity_bf + eccentricity_af)/2
        obliquity      = (obliquity_bf + obliquity_af)/2
        omega          = (omega_bf + omega_af)/2
        
        # Add to table
        orb = {}
        orb["ecc"] = eccentricity
        orb["obl"] = obliquity
        orb["pre"] = omega
        
        return orb 
    
#------------------------------------------------------------------------------
# Function to calculate the daily average solar radiation (W/m2) and the cosine
# of the zenith angle
#------------------------------------------------------------------------------


def calculate_daily_insolation(orb, lat, days, day_type):
    
    
    # For output of orbital parameters
    ecc_output       = orb["ecc"]
    obliquity_output = orb["obl"] * (180/np.pi)
    long_perh_output = orb["pre"] * (180/np.pi)
    
    # 2-D grids for days and latitudes (in radians)
    day_grid, lat_grid = np.meshgrid(days, lat)
    
    # Orbital parameter arrays
    ecc_array = np.zeros((lat.size)) + orb["ecc"]
    obl_array = np.zeros((lat.size)) + orb["obl"]
    prh_array = np.zeros((lat.size)) + orb["pre"]
    
    # 2-D grids for orbital parameters (fill to be ignored)
    fill, ecc_grid = np.meshgrid(days, ecc_array)
    fill, obl_grid = np.meshgrid(days, obl_array)
    fill, prh_grid = np.meshgrid(days, prh_array)
    
    # lambda (or solar longitude)- angular distance along Earth's orbit measured 
    # from spring equinox (21 March)

    if np.absolute(day_type) == 1: # calendar days

        # Estimate lambda from calendar day using an approximation from Berger 1978 Section 3
        delta_lambda_m = (day_grid - 80) * (2 * np.pi/365.2422)
        beta = (1 - ecc_grid**2)**(1/2)
        lambda_m0 = -2 * ( (1/2 * ecc_grid + 1/8 * ecc_grid**3) * ( 1 + beta) * np.sin(-prh_grid) - 1/4 * ecc_grid**2 * (1/2 + beta) * np.sin(-2 * prh_grid) + 1/8 * ecc_grid**3 * (1/3 + beta) * (np.sin(-3 * prh_grid)) )
        lambda_m = lambda_m0 + delta_lambda_m
        lambda_ = lambda_m + (2 * ecc_grid - 1/4 * ecc_grid**3) * np.sin(lambda_m - prh_grid) + (5/4) * ecc_grid**2 * np.sin(2 * (lambda_m - prh_grid)) + (13/12) * ecc_grid**3 * np.sin(3 * (lambda_m - prh_grid))
    
    elif np.absolute(day_type) == 2: # solar longitude (1-360)
    
        lambda_ = days * 2 * np.pi/360 # lambda=0 for spring equinox
        
    else:
        
        raise Exception("Error: invalid day_type")
        
    # Solar constant (W/m^2)
    So = 1365.2
    
    # Declination of the sun
    delta = np.arcsin(np.sin(obl_grid) * np.sin(lambda_))
    
    H0 = np.arccos(-np.tan(lat_grid) * np.tan(delta)) # hour angle at sunrise/sunset
    
    # no sunrise or no sunset: Berger 1978 eqn (8),(9)
    warnings.filterwarnings("ignore")
    H0[(( np.abs(lat_grid) >= np.pi/2 - np.abs(delta) ) & (lat_grid*delta > 0))]  = np.pi  #case no sunset
    H0[(( np.abs(lat_grid) >= np.pi/2 - np.abs(delta) ) & (lat_grid*delta <= 0))] = 0  #case no sunrise
    
    # Insolation: Berger 1978 eq (10)
    Fsw = So/np.pi * (1 + ecc_grid * np.cos(lambda_- prh_grid))**2 /(1 - ecc_grid**2)**2 * ( H0 * np.sin(lat_grid) * np.sin(delta) + np.cos(lat_grid) * np.cos(delta) * np.sin(H0) )
    
    # Solar weighted average cosine zenith angle: Balmes and Fu (2020) eqn (7)
    zav_sw = np.sin(lat_grid)**2 * np.sin(delta)**2 * H0 + 2 * np.sin(lat_grid) * np.sin(delta) * np.cos(lat_grid) * np.cos(delta) * np.sin(H0) + np.cos(lat_grid)**2 * np.cos(delta)**2 * (H0/2 + np.sin(2*H0)/4)
    zav_swd = np.rad2deg(np.arccos(zav_sw)) # in degrees
    
    # Day time average cosine zenith angle: Balmes and Fu (2020) eqn (5)
    zav_dm = (H0 * np.sin(lat_grid) * np.sin(delta) + np.cos(lat_grid) * np.cos(delta) * np.sin(H0))/H0
    i = np.where(np.isnan(zav_dm) == True)
    zav_dm[i] = 0
    zav_dmd = np.rad2deg(np.arccos(zav_dm)) # in degrees

    return ecc_output, obliquity_output, long_perh_output, Fsw, zav_sw, zav_dm


# #------------------------------------------------------------------------------
# # Plot the present-day seasonal cycle and annual mean insolation
# #------------------------------------------------------------------------------

# import proplot as pplt
# import matplotlib.colors as colors
# import os

# # define some constants
# #----------------------

# # latitude
# lat = np.linspace(-89.5,89.5,180)

# # latitude (in radians)
# latr = np.deg2rad(lat)

# # number of days
# nsteps = np.linspace(0.,365.2422,365) 

# # retrieve insolation forcing
# #----------------------------

# kyear = 0. # present-day
# orb = import_orbital_parameters_v2(kyear) 
# ecc, obl, pre, I, znth_sw, znth_dw = calculate_daily_insolation(orb, latr, nsteps, day_type = 1)

# # define some constants
# #----------------------

# # latitude
# lat = np.linspace(-89.5,89.5,180)

# # number of days
# nsteps = np.linspace(0.,365.2422,365) 


# # initialize figure
# #------------------

# fig, axs = pplt.subplots(figsize = (13.5,6), nrows = 1, ncols = 2, sharey = False, sharex = False)

# # format seasonal cycle plot
# #--------------------------

# # create meshgrid
# days, lats = np.meshgrid(np.arange(1, 366, 1), lat)

# axs.format(abc="A.", abcloc='ul')

# axs[1].format(
    
# # xaxis
# xlabel = 'Month',
# xlocator = np.linspace(30,365,12)-15,
# xminorlocator = np.linspace(30,365,12)-15,
# xticklabels = ["Ja", "Fe", "Ma", "Ap", "Ma", "Ju", "Ju", "Au", "Se", "Oc", "No", "De"],

# # yaxis
# ylocator=[-90, -60, -30, 0, 30, 60, 90],
# yminorlocator=[-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
# yformatter='deglat',
# ylabel = "Latitude",

# # axis boundaries
# ylim = (-90, 90),
# xlim = (0, 365),

# # fonts and weights
# ticklabelsize = 10, ticklabelweight = "normal", labelsize = 12, 
# labelweight = "bold", titleweight='bold', titlesize = 22)


# # format annual-mean plot
# #------------------------

# axs[0].format(
    
# # xaxis
# xlabel = 'Latitude',
# xformatter = 'deglat',
# xlocator = [-90, -60, -30, 0, 30, 60, 90],
# xminorlocator = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90],


# # yaxis
# ylabel = r'$W \ m^{-2}$',

# # axis boundaries
# xlim = (-90, 90),
# # xlim = (0, 365),

# # fonts and weights
# ticklabelsize = 10, ticklabelweight = "normal", labelsize = 12, 
# labelweight = "bold", titleweight='bold', titlesize = 22)


# # plot seasonal cycle
# #--------------------

# axs[1].contourf(days, lats, I, 
#              cmap='BuRd', colorbar='r', colorbar_kw={'label': r'W $m^{-2}$',
#                                             'ticklabelsize': 10,
#                                             'labelsize': 12,
#                                             'labelweight': 'normal',})


# # plot annual mean
# #-----------------

# # annual mean
# axs[0].plot(lat, I.mean(axis=1), linewidth = 2, color = "black", label = "Annual Average")

# # June 21 (summer solstice)
# axs[0].plot(lat, I[:,172], linewidth = 2, color = "red9", label = "Summer Solstice (Jun 21)")

# # Dec 21 (winter solstice)
# axs[0].plot(lat, I[:,355], linewidth = 2, color = "blue9", label = "Winter Solstice (Dec 21)")


# # figure legend
# #--------------

# axs.legend(loc = "ur", fontsize = 15, 
#            fontweight = "heavy", frame=False, ncols = 1)

# # save figure
# #------------

# fig.save(os.getcwd()+"/plots/inso/present_day.png", dpi = 400)




