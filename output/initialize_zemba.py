# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

Initialization of ZEMBA model state and model constants.
"""

import numpy as np
import xarray as xr
from numba.core import types
from numba.typed import Dict
from numba import njit
import os

#----------------------
# Some utility function
#----------------------

@njit(nogil = True)
def weighted_average(INPUT, land_flux, ocean_flux):
    
    '''
    Calculates the weighted average between land and ocean zones.
    '''
    
    # initialize
    landflux  = land_flux.copy()
    oceanflux = ocean_flux.copy()
    
    # convert NaNs to zeros
    for i in np.arange(0, landflux.size):
        
        if np.isnan(landflux[i]) == True:
            
            landflux[i] = 0.
            
        if np.isnan(oceanflux[i]) == True:
            
            oceanflux[i] = 0.

    # calculate weighted average
    average_flux = (INPUT["land_fraction"] * landflux + (1 - INPUT["land_fraction"]) * oceanflux)
    
    return average_flux 


@njit(nogil = True)
def global_mean(variable, Var):
    
    '''
    Calculates global average of a variable by weighing for the area of 
    the latitudinal band
    '''
    
    # area of latitudinal bands
    area = Var["sarea"]
    
    # total surface area
    total_area = np.sum(area)
    
    # weighted average of variable
    variable_weighted = np.sum(variable*area)/total_area
    
    return variable_weighted 

@njit(nogil = True)
def global_nanmean(variable, Var):
    
    '''
    Calculates global average of a variable by weighing for the area of 
    the latitudinal band
    
    Accounts for NaNs (by setting to zero).
    '''
    
    # area of latitudinal band
    area = Var["sarea"]
    
    # list for Nan values
    nanlist = []
    
    # find Nans
    for i in np.arange(0, variable.size):
        
        if np.isnan(variable[i]) == True:
            
            nanlist.append(i)
            
    # remove Nans
    variable = np.delete(variable, nanlist) 
    area     = np.delete(area, nanlist) 
     
    # total surface area
    total_area = np.sum(area) 
    
    # weighted average of variable
    variable_weighted = np.sum(variable*area)/total_area
    
    return variable_weighted 


def inferred_heat_transport(energy_in, lat):
    
    '''
    Infers northward heat transport from energy imbalance.
    '''
    
    # load modules
    from scipy import integrate
    
    # latitude in radians
    latr = np.deg2rad(lat)
    
    # cosine of latitude
    coslat = np.cos(latr)
    
    # weighted-mean energy flux
    field = coslat*energy_in
    
    # integral of energy flux for each latitude
    integral = integrate.cumtrapz(field, x=latr, initial=0)
    
    # scale
    result = (1E-15 * 2 * np.math.pi * 6371e3**2 * integral)
    
    return result


def inferred_heat_transport2(energy, lat, lon):
    
    '''
    Infers northward heat transport from energy imbalance.
    '''
    
    # load modules
    from scipy import integrate
    
    # energy integral over longitude
    energy_lat = np.zeros((lat.size))
    
    # latitude in radiation
    latr = np.deg2rad(lat)
    
    # longitude in radiation
    lonr = np.deg2rad(lon)
    
    # integrate over longitude
    for i in np.arange(0, lat.size):
        
        field = energy[i,:] * 6371e3**2 * np.cos(latr[i])
        
        energy_lat[i] = integrate.trapz(field, x=lonr)
       
        
    # integrate over latitude
    
    result = integrate.cumtrapz(energy_lat, x=latr, initial=0)
        
    return result


def get_constants(INPUT):
    
    '''
    Function that retrieves model constants and dimensions for ZEMBA
    '''
    
    Var = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:],)
    
    #-------------------
    # earth's radius (m)
    #-------------------
    
    Var["r_earth"] = np.array([6371e3]) 

    #--------------------------------
    # model dimensions and timesteps.
    #--------------------------------

    # no. of days                                       
    Var["ndays"] = np.linspace(1., 365.2422, 365) 
   
    # seconds in day
    Var["secs_in_day"] = np.array([86400.])
 
    # model grid
    if INPUT["res"]==1.:        
        # latitude at grid centres 
        Var["lat"]   = np.arange(-89.5, 89.5+1., 1.)  # in degrees
        Var["latr"]  = np.deg2rad(Var["lat"])         # in radians
        # latitude at grid boundaries
        Var["latb"]   = np.arange(-90., 90.+1., 1.)  # in degrees
        Var["latbr"]  = np.deg2rad(Var["latb"])      # in radians
        
    if INPUT['res']==2.5:        
        # latitude at grid centres 
        Var["lat"]   = np.arange(-88.75, 88.75+2.5, 2.5)  # in degrees
        Var["latr"]  = np.deg2rad(Var["lat"])             # in radians
        # latitude at grid boundaries
        Var["latb"]   = np.arange(-90., 90.+2.5, 2.5)  # in degrees
        Var["latbr"]  = np.deg2rad(Var["latb"])        # in radians
    
    if INPUT['res']==5.:        
        # latitude at grid centres 
        Var["lat"]   = np.arange(-87.5, 87.5+5., 5.)  # in degrees
        Var["latr"]  = np.deg2rad(Var["lat"])         # in radians
        # latitude at grid boundaries
        Var["latb"]   = np.arange(-90., 90.+5., 5.)  # in degrees
        Var["latbr"]  = np.deg2rad(Var["latb"])      # in radians
        
    # grid spacing (in degrees)
    Var["dlat"] = np.array([np.diff(Var["lat"])[0]])
    
    # grid spacing (in radians)
    Var["dlatr"] = np.array([np.diff(Var["latr"])[0]])
    
    # no. of days in each months 
    Var["days_in_months"] = np.array([31., 28., 31., 30., 31., 30., 31., 31., 30., 31., 30., 31.])

    # index for location of equator
    idx = np.where(Var["latb"] == 0.)
    Var["idxeq"] = np.asarray(idx[0], dtype = 'f8')

    #------------------------
    # ocean model dimensions.
    #------------------------
    
    # ocean grid
    #-----------
    
    if INPUT['res']==1.:
        # latitude at grid centres 
        Var["olat"]  = np.arange(-69.5, 79.5 + 1., 1.)  # in degrees
        Var["olatr"] = np.deg2rad(Var["olat"])          # in radians
        # latitude at grid boundaries
        Var["olatb"]  = np.arange(-70., 80. + 1., 1.)  # in degrees
        Var["olatbr"] = np.deg2rad(Var["olatb"])       # in radians 
        
    if INPUT['res']==2.5:
        # latitude at grid centres 
        Var["olat"]  = np.arange(-68.75, 78.75 + 2.5, 2.5)  # in degrees
        Var["olatr"] = np.deg2rad(Var["olat"])              # in radians
        # latitude at grid boundaries
        Var["olatb"]  = np.arange(-70., 80.+ 2.5, 2.5)  # in degrees
        Var["olatbr"] = np.deg2rad(Var["olatb"])        # in radians
        
    if INPUT['res']==5.:
        # latitude at grid centres 
        Var["olat"]  = np.arange(-67.5, 77.5 + 5., 5.)  # in degrees
        Var["olatr"] = np.deg2rad(Var["olat"])          # in radians
        # latitude at grid boundaries
        Var["olatb"]  = np.arange(-70., 80. + 5., 5.)  # in degrees
        Var["olatbr"] = np.deg2rad(Var["olatb"])       # in radians
    
    # Index for ocean circulation boundaries within ZEMBA grid
    #---------------------------------------------------------
    
    if INPUT['res'] == 1.:
        
        Var["idxoc"] = np.concatenate((
            np.arange(20. + (Var["lat"].size*0), 170. + (Var["lat"].size*0)),
            np.arange(20. + (Var["lat"].size*1), 170. + (Var["lat"].size*1)),
            np.arange(20. + (Var["lat"].size*2), 170. + (Var["lat"].size*2)),
            np.arange(20. + (Var["lat"].size*3), 170. + (Var["lat"].size*3)),
            np.arange(20. + (Var["lat"].size*4), 170. + (Var["lat"].size*4)),
            np.arange(20. + (Var["lat"].size*5), 170. + (Var["lat"].size*5))
            ))
        
    if INPUT['res'] == 2.5:
        
        Var["idxoc"] = np.concatenate((
            np.arange(8. + (Var["lat"].size*0), 68. + (Var["lat"].size*0)),
            np.arange(8. + (Var["lat"].size*1), 68. + (Var["lat"].size*1)),
            np.arange(8. + (Var["lat"].size*2), 68. + (Var["lat"].size*2)),
            np.arange(8. + (Var["lat"].size*3), 68. + (Var["lat"].size*3)),
            np.arange(8. + (Var["lat"].size*4), 68. + (Var["lat"].size*4)),
            np.arange(8. + (Var["lat"].size*5), 68. + (Var["lat"].size*5))
            ))
        
    if INPUT['res'] == 5:
        
        Var["idxoc"] = np.concatenate((
            np.arange(4. + (Var["lat"].size*0), 34. + (Var["lat"].size*0)),
            np.arange(4. + (Var["lat"].size*1), 34. + (Var["lat"].size*1)),
            np.arange(4. + (Var["lat"].size*2), 34. + (Var["lat"].size*2)),
            np.arange(4. + (Var["lat"].size*3), 34. + (Var["lat"].size*3)),
            np.arange(4. + (Var["lat"].size*4), 34. + (Var["lat"].size*4)),
            np.arange(4. + (Var["lat"].size*5), 34. + (Var["lat"].size*5))
            ))
        
    # Index for no ocean circulation boundaries (deep ocean) 
    #-------------------------------------------------------
    
    # index for no ocean circulation boundaries (deep ocean) in the south
    if INPUT['res']==1.:
        Var["idxnocs"] = np.asarray(np.arange(0, 20), dtype = 'f8')  
    if INPUT['res']==2.5:
        Var["idxnocs"] = np.asarray(np.arange(0, 8), dtype = 'f8')
    if INPUT['res']==5.:
        Var["idxnocs"] = np.asarray(np.arange(0, 4), dtype = 'f8')

    # index for no ocean circulation boundaries (deep ocean) in the north
    if INPUT['res']==1.:
        Var["idxnocn"] = np.asarray(np.arange(170, Var["lat"].size), dtype = 'f8') 
    if INPUT['res']==2.5:
        Var["idxnocn"] = np.asarray(np.arange(68, Var["lat"].size), dtype = 'f8') 
    if INPUT['res']==5.:
        Var["idxnocn"] = np.asarray(np.arange(34, Var["lat"].size), dtype = 'f8') 

    #---------------------
    # Index for land cover
    #---------------------
    
    # Index for completely land covered grids.
    i = np.where(INPUT["land_fraction"] == 1.)
    Var["idxal"] = np.asarray(i[0], dtype = 'f8')
    
    # Index for completly ocean covered latitdudes
    i = np.where(INPUT["land_fraction"] == 0.)
    Var["idxao"] = np.asarray(i[0], dtype = 'f8')
    
    #-------------------
    # Surface areas (m2)
    #-------------------
    
    # surface area
    Var["sarea"] =  (
                    # east-west width of latitude band
                    (2*np.pi*Var["r_earth"]*np.cos(Var["latr"])) 
                    * 
                    # north-south length of latitude band 
                    (Var["r_earth"]*Var["dlatr"])
                    ) 
    
    # width of latitudinal band
    Var["swidth"]  = 2*np.pi*Var["r_earth"]*np.cos(Var["latbr"])
    
    # surface area of land
    Var["larea"] = Var["sarea"]*INPUT["land_fraction"]
    
    # surface area of ocean
    Var["oarea"] = Var["sarea"]*(1-INPUT["land_fraction"])
    
    #----------------------
    # Depth of ocean layers
    #----------------------
    
    # ocean depth at grid centres (m)
    Var["odepth"]  = np.array([100, 316.6, 543.5, 775.8, 1012.3, 1251.8])
    
    # ocean depth at grid boundaries (m)
    Var["odepth_diff"] = np.array([208.3, 430.05, 659.65, 894.05, 1132.05])
    
    #--------------------------------
    # Vertical ocean velocities (m/s)
    #--------------------------------
    
    # Ensures mean upwelling rate of 4 m/yr from 50S to 60N, with a global mean 
    # vertical velocity of 0 m/yr (following Bintanja, 1997). Code that 
    # calculates velocity field in miscellaneous folder. 
    
    # Four velocities fields for each resolution, corresponding to different 
    # centres of ocean circulation. Ensures each hemispheres upwelling zone
    # has the same average upwelling rate.
    
    if INPUT['res']==1.:
        
        # with centre of ocean circulation at equator
        Var["w0"] = np.array([ -37.1774302 , -35.31855869, -33.45968718, -31.60081567,
                               -29.74194416, -27.88307265, -26.02420114, -24.16532963,
                               -22.30645812, -20.44758661, -18.5887151 , -16.72984359,
                               -14.87097208, -13.01210057, -11.15322906,  -9.29435755,
                                -7.43548604,  -5.57661453,  -3.71774302,  -1.85887151,
                                 0.        ,   0.15257402,   0.30514805,   0.45772207,
                                 0.61029609,   0.76287012,   0.91544414,   1.06801817,
                                 1.22059219,   1.37316621,   1.52574024,   1.67831426,
                                 1.83088828,   1.98346231,   2.13603633,   2.28861035,
                                 2.44118438,   2.5937584 ,   2.74633243,   2.89890645,
                                 3.05148047,   3.2040545 ,   3.35662852,   3.50920254,
                                 3.66177657,   3.81435059,   3.96692461,   4.11949864,
                                 4.27207266,   4.42464669,   4.57722071,   4.72979473,
                                 4.88236876,   5.03494278,   5.1875168 ,   5.34009083,
                                 5.49266485,   5.64523887,   5.7978129 ,   5.95038692,
                                 6.10296095,   6.25553497,   6.40810899,   6.56068302,
                                 6.71325704,   6.86583106,   7.01840509,   7.17097911,
                                 7.32355313,   7.47612716,   7.14734678,   7.02822434,
                                 6.90910189,   6.78997944,   6.670857  ,   6.55173455,
                                 6.4326121 ,   6.31348966,   6.19436721,   6.07524477,
                                 5.95612232,   5.83699987,   5.71787743,   5.59875498,
                                 5.47963253,   5.36051009,   5.24138764,   5.12226519,
                                 5.00314275,   4.8840203 ,   4.76489786,   4.64577541,
                                 4.52665296,   4.40753052,   4.28840807,   4.16928562,
                                 4.05016318,   3.93104073,   3.81191828,   3.69279584,
                                 3.57367339,   3.45455095,   3.3354285 ,   3.21630605,
                                 3.09718361,   2.97806116,   2.85893871,   2.73981627,
                                 2.62069382,   2.50157137,   2.38244893,   2.26332648,
                                 2.14420403,   2.02508159,   1.90595914,   1.7868367 ,
                                 1.66771425,   1.5485918 ,   1.42946936,   1.31034691,
                                 1.19122446,   1.07210202,   0.95297957,   0.83385712,
                                 0.71473468,   0.59561223,   0.47648979,   0.35736734,
                                 0.23824489,   0.11912245,  -0.        ,  -3.69048917,
                                -7.38097834, -11.07146751, -14.76195668, -18.45244585,
                               -22.14293501, -25.83342418, -29.52391335, -33.21440252,
                               -36.90489169, -40.59538086, -44.28587003, -47.9763592 ,
                               -51.66684837, -55.35733754, -59.04782671, -62.73831587,
                               -66.42880504, -70.11929421]) /(365*24*60*60) # in m/s
        
        # with centre of ocean circulation at 5S
        Var["w5s"] = np.array([-32.94761486, -31.30023411, -29.65285337, -28.00547263,
                               -26.35809189, -24.71071114, -23.0633304 , -21.41594966,
                               -19.76856891, -18.12118817, -16.47380743, -14.82642669,
                               -13.17904594, -11.5316652 ,  -9.88428446,  -8.23690371,
                                -6.58952297,  -4.94214223,  -3.29476149,  -1.64738074,
                                 0.        ,   0.16986265,   0.33972531,   0.50958796,
                                 0.67945061,   0.84931327,   1.01917592,   1.18903857,
                                 1.35890123,   1.52876388,   1.69862653,   1.86848919,
                                 2.03835184,   2.20821449,   2.37807715,   2.5479398 ,
                                 2.71780245,   2.88766511,   3.05752776,   3.22739041,
                                 3.39725307,   3.56711572,   3.73697837,   3.90684103,
                                 4.07670368,   4.24656633,   4.41642899,   4.58629164,
                                 4.75615429,   4.92601695,   5.0958796 ,   5.26574225,
                                 5.43560491,   5.60546756,   5.77533021,   5.94519287,
                                 6.11505552,   6.28491817,   6.45478083,   6.62464348,
                                 6.79450613,   6.96436879,   7.13423144,   7.30409409,
                                 7.47395675,   7.16867153,   7.05838427,   6.94809702,
                                 6.83780977,   6.72752251,   6.61723526,   6.506948  ,
                                 6.39666075,   6.28637349,   6.17608624,   6.06579899,
                                 5.95551173,   5.84522448,   5.73493722,   5.62464997,
                                 5.51436271,   5.40407546,   5.29378821,   5.18350095,
                                 5.0732137 ,   4.96292644,   4.85263919,   4.74235193,
                                 4.63206468,   4.52177743,   4.41149017,   4.30120292,
                                 4.19091566,   4.08062841,   3.97034115,   3.8600539 ,
                                 3.74976665,   3.63947939,   3.52919214,   3.41890488,
                                 3.30861763,   3.19833037,   3.08804312,   2.97775587,
                                 2.86746861,   2.75718136,   2.6468941 ,   2.53660685,
                                 2.42631959,   2.31603234,   2.20574509,   2.09545783,
                                 1.98517058,   1.87488332,   1.76459607,   1.65430881,
                                 1.54402156,   1.43373431,   1.32344705,   1.2131598 ,
                                 1.10287254,   0.99258529,   0.88229803,   0.77201078,
                                 0.66172353,   0.55143627,   0.44114902,   0.33086176,
                                 0.22057451,   0.11028725,  -0.        ,  -4.06189551,
                                -8.12379102, -12.18568652, -16.24758203, -20.30947754,
                               -24.37137305, -28.43326856, -32.49516406, -36.55705957,
                               -40.61895508, -44.68085059, -48.7427461 , -52.8046416 ,
                               -56.86653711, -60.92843262, -64.99032813, -69.05222364,
                               -73.11411914, -77.17601465]) /(365*24*60*60) # in m/s
        
        # with centre of ocean circulation at 10S
        Var["w10s"] = np.array([-28.74999096, -27.31249141, -25.87499186, -24.43749231,
                                -22.99999277, -21.56249322, -20.12499367, -18.68749412,
                                -17.24999457, -15.81249503, -14.37499548, -12.93749593,
                                -11.49999638, -10.06249683,  -8.62499729,  -7.18749774,
                                 -5.74999819,  -4.31249864,  -2.8749991 ,  -1.43749955,
                                  0.        ,   0.19181196,   0.38362391,   0.57543587,
                                  0.76724782,   0.95905978,   1.15087173,   1.34268369,
                                  1.53449565,   1.7263076 ,   1.91811956,   2.10993151,
                                  2.30174347,   2.49355542,   2.68536738,   2.87717934,
                                  3.06899129,   3.26080325,   3.4526152 ,   3.64442716,
                                  3.83623911,   4.02805107,   4.21986303,   4.41167498,
                                  4.60348694,   4.79529889,   4.98711085,   5.1789228 ,
                                  5.37073476,   5.56254672,   5.75435867,   5.94617063,
                                  6.13798258,   6.32979454,   6.5216065 ,   6.71341845,
                                  6.90523041,   7.09704236,   7.28885432,   7.48066627,
                                  7.19640608,   7.09360028,   6.99079447,   6.88798867,
                                  6.78518287,   6.68237707,   6.57957127,   6.47676547,
                                  6.37395967,   6.27115387,   6.16834807,   6.06554226,
                                  5.96273646,   5.85993066,   5.75712486,   5.65431906,
                                  5.55151326,   5.44870746,   5.34590166,   5.24309586,
                                  5.14029005,   5.03748425,   4.93467845,   4.83187265,
                                  4.72906685,   4.62626105,   4.52345525,   4.42064945,
                                  4.31784365,   4.21503784,   4.11223204,   4.00942624,
                                  3.90662044,   3.80381464,   3.70100884,   3.59820304,
                                  3.49539724,   3.39259144,   3.28978563,   3.18697983,
                                  3.08417403,   2.98136823,   2.87856243,   2.77575663,
                                  2.67295083,   2.57014503,   2.46733923,   2.36453343,
                                  2.26172762,   2.15892182,   2.05611602,   1.95331022,
                                  1.85050442,   1.74769862,   1.64489282,   1.54208702,
                                  1.43928122,   1.33647541,   1.23366961,   1.13086381,
                                  1.02805801,   0.92525221,   0.82244641,   0.71964061,
                                  0.61683481,   0.51402901,   0.4112232 ,   0.3084174 ,
                                  0.2056116 ,   0.1028058 ,  -0.        ,  -4.43047522,
                                 -8.86095044, -13.29142566, -17.72190088, -22.1523761 ,
                                -26.58285132, -31.01332654, -35.44380176, -39.87427698,
                                -44.3047522 , -48.73522742, -53.16570265, -57.59617787,
                                -62.02665309, -66.45712831, -70.88760353, -75.31807875,
                                -79.74855397, -84.17902919]) /(365*24*60*60) # in m/s
        
        # with centre of ocean circulation at 15S
        Var["w15s"] = np.array([-24.61650496, -23.38567971, -22.15485446, -20.92402922,
                                -19.69320397, -18.46237872, -17.23155347, -16.00072822,
                                -14.76990298, -13.53907773, -12.30825248, -11.07742723,
                                 -9.84660198,  -8.61577674,  -7.38495149,  -6.15412624,
                                 -4.92330099,  -3.69247574,  -2.4616505 ,  -1.23082525,
                                  0.        ,   0.22049188,   0.44098375,   0.66147563,
                                  0.88196751,   1.10245939,   1.32295126,   1.54344314,
                                  1.76393502,   1.98442689,   2.20491877,   2.42541065,
                                  2.64590252,   2.8663944 ,   3.08688628,   3.30737816,
                                  3.52787003,   3.74836191,   3.96885379,   4.18934566,
                                  4.40983754,   4.63032942,   4.85082129,   5.07131317,
                                  5.29180505,   5.51229693,   5.7327888 ,   5.95328068,
                                  6.17377256,   6.39426443,   6.61475631,   6.83524819,
                                  7.05574006,   7.27623194,   7.49672382,   7.2308191 ,
                                  7.13440818,   7.03799725,   6.94158633,   6.84517541,
                                  6.74876449,   6.65235357,   6.55594265,   6.45953173,
                                  6.36312081,   6.26670988,   6.17029896,   6.07388804,
                                  5.97747712,   5.8810662 ,   5.78465528,   5.68824436,
                                  5.59183343,   5.49542251,   5.39901159,   5.30260067,
                                  5.20618975,   5.10977883,   5.01336791,   4.91695699,
                                  4.82054606,   4.72413514,   4.62772422,   4.5313133 ,
                                  4.43490238,   4.33849146,   4.24208054,   4.14566962,
                                  4.04925869,   3.95284777,   3.85643685,   3.76002593,
                                  3.66361501,   3.56720409,   3.47079317,   3.37438225,
                                  3.27797132,   3.1815604 ,   3.08514948,   2.98873856,
                                  2.89232764,   2.79591672,   2.6995058 ,   2.60309487,
                                  2.50668395,   2.41027303,   2.31386211,   2.21745119,
                                  2.12104027,   2.02462935,   1.92821843,   1.8318075 ,
                                  1.73539658,   1.63898566,   1.54257474,   1.44616382,
                                  1.3497529 ,   1.25334198,   1.15693106,   1.06052013,
                                  0.96410921,   0.86769829,   0.77128737,   0.67487645,
                                  0.57846553,   0.48205461,   0.38564369,   0.28923276,
                                  0.19282184,   0.09641092,  -0.        ,  -4.79342319,
                                 -9.58684638, -14.38026958, -19.17369277, -23.96711596,
                                -28.76053915, -33.55396235, -38.34738554, -43.14080873,
                                -47.93423192, -52.72765511, -57.52107831, -62.3145015 ,
                                -67.10792469, -71.90134788, -76.69477107, -81.48819427,
                                -86.28161746, -91.07504065]) /(365*24*60*60) # in m/s
        
    if INPUT['res']==2.5:
        
        # with centre of ocean circulation at equator
        Var["w0"] = np.array([-34.41078516, -30.10943702, -25.80808887, -21.50674073,
                                -17.20539258, -12.90404444,  -8.60269629,  -4.30134815,
                                  0.        ,   0.39272328,   0.78544656,   1.17816984,
                                  1.57089312,   1.9636164 ,   2.35633969,   2.74906297,
                                  3.14178625,   3.53450953,   3.92723281,   4.31995609,
                                  4.71267937,   5.10540265,   5.49812593,   5.89084921,
                                  6.2835725 ,   6.67629578,   7.06901906,   7.46174234,
                                  6.99209319,   6.70075598,   6.40941876,   6.11808154,
                                  5.82674433,   5.53540711,   5.24406989,   4.95273268,
                                  4.66139546,   4.37005825,   4.07872103,   3.78738381,
                                  3.4960466 ,   3.20470938,   2.91337216,   2.62203495,
                                  2.33069773,   2.03936051,   1.7480233 ,   1.45668608,
                                  1.16534887,   0.87401165,   0.58267443,   0.29133722,
                                 -0.        , -10.16388037, -20.32776074, -30.49164112,
                                -40.65552149, -50.81940186, -60.98328223, -71.14716261]) /(365*24*60*60) # in m/s
                         
        # with centre of ocean circulation at 5S
        Var["w5s"] = np.array([-30.49574137, -26.6837737 , -22.87180603, -19.05983836,
                               -15.24787069, -11.43590301,  -7.62393534,  -3.81196767,
                                 0.        ,   0.43870305,   0.8774061 ,   1.31610915,
                                 1.7548122 ,   2.19351525,   2.6322183 ,   3.07092136,
                                 3.50962441,   3.94832746,   4.38703051,   4.82573356,
                                 5.26443661,   5.70313966,   6.14184271,   6.58054576,
                                 7.01924881,   7.45795186,   7.02418916,   6.75402804,
                                 6.48386692,   6.2137058 ,   5.94354467,   5.67338355,
                                 5.40322243,   5.13306131,   4.86290019,   4.59273907,
                                 4.32257794,   4.05241682,   3.7822557 ,   3.51209458,
                                 3.24193346,   2.97177234,   2.70161122,   2.43145009,
                                 2.16128897,   1.89112785,   1.62096673,   1.35080561,
                                 1.08064449,   0.81048336,   0.54032224,   0.27016112,
                                -0.        , -11.18676092, -22.37352185, -33.56028277,
                               -44.7470437 , -55.93380462, -67.12056555, -78.30732647]) /(365*24*60*60) # in m/s
        
        # with centre of ocean circulation at 10S
        Var["w10s"] = np.array([-26.61049343, -23.28418175, -19.95787007, -16.63155839,
                                -13.30524672,  -9.97893504,  -6.65262336,  -3.32631168,
                                  0.        ,   0.49752856,   0.99505712,   1.49258567,
                                  1.99011423,   2.48764279,   2.98517135,   3.48269991,
                                  3.98022847,   4.47775702,   4.97528558,   5.47281414,
                                  5.9703427 ,   6.46787126,   6.96539981,   7.46292837,
                                  7.06094488,   6.80876828,   6.55659167,   6.30441507,
                                  6.05223847,   5.80006187,   5.54788526,   5.29570866,
                                  5.04353206,   4.79135545,   4.53917885,   4.28700225,
                                  4.03482565,   3.78264904,   3.53047244,   3.27829584,
                                  3.02611923,   2.77394263,   2.52176603,   2.26958943,
                                  2.01741282,   1.76523622,   1.51305962,   1.26088301,
                                  1.00870641,   0.75652981,   0.50435321,   0.2521766 ,
                                 -0.        , -12.20185674, -24.40371348, -36.60557021,
                                -48.80742695, -61.00928369, -73.21114043, -85.41299716]) /(365*24*60*60) # in m/s
    
        # with centre of ocean circulation at 15S
        Var["w15s"] = np.array([-22.78461042, -19.93653412, -17.08845781, -14.24038151,
                                -11.39230521,  -8.54422891,  -5.6961526 ,  -2.8480763 ,
                                  0.        ,   0.57516242,   1.15032485,   1.72548727,
                                  2.30064969,   2.87581211,   3.45097454,   4.02613696,
                                  4.60129938,   5.1764618 ,   5.75162423,   6.32678665,
                                  6.90194907,   7.47711149,   7.10296005,   6.86619472,
                                  6.62942938,   6.39266405,   6.15589871,   5.91913338,
                                  5.68236804,   5.44560271,   5.20883737,   4.97207204,
                                  4.7353067 ,   4.49854137,   4.26177603,   4.0250107 ,
                                  3.78824536,   3.55148003,   3.31471469,   3.07794936,
                                  2.84118402,   2.60441869,   2.36765335,   2.13088802,
                                  1.89412268,   1.65735735,   1.42059201,   1.18382668,
                                  0.94706134,   0.71029601,   0.47353067,   0.23676534,
                                 -0.        , -13.20144232, -26.40288464, -39.60432696,
                                -52.80576928, -66.0072116 , -79.20865392, -92.41009624]) /(365*24*60*60) # in m/s
                         
    
    if INPUT['res']==5:
        
        # with centre of ocean circulation at equator
        Var["w0"] = np.array([-30.54702049, -22.91026537, -15.27351025,  -7.63675512,
                              0.        ,   0.8264291 ,   1.65285819,   2.47928729,
                              3.30571639,   4.13214549,   4.95857458,   5.78500368,
                              6.61143278,   7.43786187,   6.74939588,   6.18694623,
                              5.62449657,   5.06204691,   4.49959726,   3.9371476 ,
                              3.37469794,   2.81224829,   2.24979863,   1.68734897,
                              1.12489931,   0.56244966,  -0.        , -24.30999256,
                            -48.61998511, -72.92997767]) /(365*24*60*60) # in m/s
        
        # with centre of ocean circulation at 5S
        Var["w5s"] = np.array([-27.07157167, -20.30367875, -13.53578583,  -6.76789292,
                                  0.        ,   0.928925  ,   1.85785   ,   2.786775  ,
                                  3.71570001,   4.64462501,   5.57355001,   6.50247501,
                                  7.43140001,   6.79725234,   6.27438678,   5.75152121,
                                  5.22865565,   4.70579008,   4.18292452,   3.66005895,
                                  3.13719339,   2.61432782,   2.09146226,   1.56859669,
                                  1.04573113,   0.52286556,  -0.        , -26.75652062,
                                -53.51304125, -80.26956187]) /(365*24*60*60) # in m/s
        
        # with centre of ocean circulation at 10S
        Var["w10s"] = np.array([-23.62257311, -17.71692983, -11.81128655,  -5.90564328,
                                 0.        ,   1.06193178,   2.12386357,   3.18579535,
                                 4.24772714,   5.30965892,   6.37159071,   7.43352249,
                                 6.84729993,   6.35820708,   5.86911423,   5.38002137,
                                 4.89092852,   4.40183567,   3.91274282,   3.42364997,
                                 2.93455711,   2.44546426,   1.95637141,   1.46727856,
                                 0.9781857 ,   0.48909285,  -0.        , -29.18442913,
                               -58.36885827, -87.5532874 ]) /(365*24*60*60) # in m/s
        
        # with centre of ocean circulation at 15S
        Var["w15s"] = np.array([-20.22627377, -15.16970533, -10.11313689,  -5.05656844,
                                 0.        ,   1.24077045,   2.4815409 ,   3.72231136,
                                 4.96308181,   6.20385226,   7.44462271,   6.90057867,
                                 6.44054009,   5.98050151,   5.52046293,   5.06042435,
                                 4.60038578,   4.1403472 ,   3.68030862,   3.22027004,
                                 2.76023147,   2.30019289,   1.84015431,   1.38011573,
                                 0.92007716,   0.46003858,  -0.        , -31.57524024,
                               -63.15048048, -94.72572072]) /(365*24*60*60) # in m/s
        

    #--------------------------------------------
    # Ocean surface area and widths [ocean model]
    #--------------------------------------------
    
    # Seperate ocean widths and surface areas in the ocean circulation model,
    # where there is a constant ocean fractional width of 0.7 to avoid 
    # unrealistically large meridional velocities
    
    # ocean fraction (in ocean circulation model)
    f  = 0.7
    fb = 0.7
    
    # surface area of ocean at grid centres
    Var["oa"] = (
                # east-west width of latitude band
                (2*np.pi*Var["r_earth"]*f*np.cos(Var["olatr"]))
                * 
                # north-south length of latitude band 
                (Var["r_earth"]*Var["dlatr"])
                )
    
    # East-west width of ocean at grid boundaries
    Var["ow"] = (2*np.pi*Var["r_earth"]*fb*np.cos(Var["olatbr"]))
    
    #----------------------------
    # Ocean overturning constants
    #----------------------------
    
    # overturning rate scaling parameter (see Stap et al., 2014)
    Var["zeta"] = np.array([6.])
    
    # polar water freshening factor (see Stap et al., 2014)
    Var["gamma"] = np.array([-0.09])
    
    # standard PI surface ocean temperature in northern polar waters (60N - 80N)
    Var["ton0"] = np.array([272.16])
    
    # standard PI surface ocean temperature in southrn polar waters (70S - 50S)
    Var["tos0"] = np.array([275.53])
    
    # standard PI density difference between equator and polar regions in NH
    Var["drn0"] = np.array([5.60])
    
    # standard PI density difference between equator and polar regions in SH
    Var["drs0"] = np.array([5.41])
    
    #-----------------------------------------------------
    # Index for northern and southern hemisphere latitudes.
    #------------------------------------------------------
    
    # index for SH grid cells
    if INPUT['res']==1.:
        Var["idxsh"] = np.asarray(np.arange(0,89+1,1), dtype = 'f8') 
    if INPUT['res']==2.5:
        Var["idxsh"] = np.asarray(np.arange(0,35+1,1), dtype = 'f8') 
    if INPUT['res']==5.:
        Var["idxsh"] = np.asarray(np.arange(0,17+1,1), dtype = 'f8')

    # index for NH grid cells
    if INPUT['res']==1.:
        Var["idxnh"] = np.asarray(np.arange(90,179+1,1), dtype = 'f8')    
    if INPUT['res']==2.5:
        Var["idxnh"] = np.asarray(np.arange(36,71+1,1), dtype = 'f8') 
    if INPUT['res']==5.:
        Var["idxnh"] = np.asarray(np.arange(18,35+1,1), dtype = 'f8')

    #---------------------
    # Land and ocean masks
    #---------------------
    
    # Land mask 
    Var["land_mask"] = np.zeros((Var["lat"].size))
    Var["land_mask"][INPUT["land_fraction"] > 0]  = 1
    Var["land_mask"][INPUT["land_fraction"] == 0] = np.NaN
    
    # Ocean mask 
    Var["ocean_mask"] = np.zeros((Var["lat"].size))
    Var["ocean_mask"][1-INPUT["land_fraction"] > 0]  = 1
    Var["ocean_mask"][1-INPUT["land_fraction"] == 0] = np.NaN

    #-------------                              
    # Ocean height
    #-------------
    
    Var["ocean_height"] = np.zeros((Var["lat"].size))
    Var["ocean_height"] = Var["ocean_height"]*Var["ocean_mask"] # mas
        
    #------------
    # Mean height
    #------------
    
    Var["mean_height"] = weighted_average(INPUT, INPUT["land_height"], Var["ocean_height"])

    #------------------------------
    # Atmospheric related constants
    #------------------------------
    
    # Stefan-boltzman constant 
    Var["sigma"] = np.array([5.667E-8])  # (W m^-2 K^-1)
    
    # Lapse Rate
    Var["lr"] = np.array([0.0065]) # (K/m)
    
    # Atmospheric thickness 
    Var["atm_depth"] = np.array([8194.]) # (m)
    
    # Specific heat capacity of air 
    Var["atm_sphc"] = np.array([1004.]) # (J Kg^-1 K^-1)
    
    # Air Density 
    Var["atm_rho"] = np.array([1.25]) # (Kg m^-3)
    
    # Heat capacity of atmopsheric column (
    Var["atm_hc"] = Var["atm_depth"]*Var["atm_sphc"]*Var["atm_rho"] 
    
    # Cloud emissivity
    Var["cloud_emissivity"] = np.array([1.])
    
    #------------------------
    # Ocean related constants
    #------------------------
    
    # Specific heat capacity of sea water
    Var["ocean_sphc"] = np.array([3850.]) # (J Kg^-1 K^-1)
    
    # Seawater density
    Var["ocean_rho"] = np.array([1025.])  # Sea water density (Kg m^-3)
    
    # Heat capacity of each ocean layer 
    Var["ocean_hc1"] = Var["odepth"][0]*Var["ocean_sphc"]*Var["ocean_rho"] 
    Var["ocean_hc2"] = Var["odepth"][1]*Var["ocean_sphc"]*Var["ocean_rho"]
    Var["ocean_hc3"] = Var["odepth"][2]*Var["ocean_sphc"]*Var["ocean_rho"]
    Var["ocean_hc4"] = Var["odepth"][3]*Var["ocean_sphc"]*Var["ocean_rho"]
    Var["ocean_hc5"] = Var["odepth"][4]*Var["ocean_sphc"]*Var["ocean_rho"]
    Var["ocean_hc6"] = Var["odepth"][5]*Var["ocean_sphc"]*Var["ocean_rho"]

    # Concatenated heat capacities for each layer
    Var["ocean_hc"] = np.concatenate((np.zeros((Var["olat"].size)) + Var["ocean_hc1"]
                                     ,np.zeros((Var["olat"].size)) + Var["ocean_hc2"]
                                     ,np.zeros((Var["olat"].size)) + Var["ocean_hc3"]
                                     ,np.zeros((Var["olat"].size)) + Var["ocean_hc4"]
                                     ,np.zeros((Var["olat"].size)) + Var["ocean_hc5"]
                                     ,np.zeros((Var["olat"].size)) + Var["ocean_hc6"]
                                     ))

    # Indicies for each ocean layer in ZEMBA grid.
    Var["idxolyr1"] = np.array(np.arange(0, Var["lat"].size), dtype = 'f8')
    Var["idxolyr2"] = np.array(np.arange(Var["lat"].size,   Var["lat"].size*2), dtype = 'f8')
    Var["idxolyr3"] = np.array(np.arange(Var["lat"].size*2, Var["lat"].size*3), dtype = 'f8')
    Var["idxolyr4"] = np.array(np.arange(Var["lat"].size*3, Var["lat"].size*4), dtype = 'f8')
    Var["idxolyr5"] = np.array(np.arange(Var["lat"].size*4, Var["lat"].size*5), dtype = 'f8')
    Var["idxolyr6"] = np.array(np.arange(Var["lat"].size*5, Var["lat"].size*6), dtype = 'f8')
    
    # Indicies for each ocean layer in ZEMBA grid.
    Var["idxbolyr1"] = np.array(np.arange(0, Var["latb"].size), dtype = 'f8')
    Var["idxbolyr2"] = np.array(np.arange(Var["latb"].size,   Var["latb"].size*2), dtype = 'f8')
    Var["idxbolyr3"] = np.array(np.arange(Var["latb"].size*2, Var["latb"].size*3), dtype = 'f8')
    Var["idxbolyr4"] = np.array(np.arange(Var["latb"].size*3, Var["latb"].size*4), dtype = 'f8')
    Var["idxbolyr5"] = np.array(np.arange(Var["latb"].size*4, Var["latb"].size*5), dtype = 'f8')
    Var["idxbolyr6"] = np.array(np.arange(Var["latb"].size*5, Var["latb"].size*6), dtype = 'f8')

    #-----------------------
    # Land related constants
    #-----------------------
            
    # Land depth 
    Var["land_depth"] = np.array([2.2]) # (m)
    
    # Specific heat capacity of land 
    Var["land_sphc"]  = np.array([1480.]) # (J Kg^-1 K^-1)
    
    # Land density 
    Var["land_rho"]   = np.array([2000.])  # (Kg m^-3)
    
    # Antarctic heat capacity
    Var["ant_sphc"]   = np.array([2100.])  # (J Kg^-1 K^-1)
    
    # Heat capacity of land surface
    Var["land_hc"]  = np.zeros((Var["lat"].size)) + (Var["land_depth"] * Var["land_sphc"] * Var["land_rho"])

    #------------------------------------
    # Hydrolgical cycle related constants
    #------------------------------------
    
    # Water density  
    Var["water_rho"] = np.array([1000.]) # (Kg m^-3)
    
    # Latent heat of vaporization
    Var["Lv"] = np.array([2.5e6])   # (J kg^-1)
    
    # Specific humidity scale depth
    Var["H_q"] = np.array([1800.])  # (m)
    
    # Water availiability over land
    Var["W_land"] = np.array([0.7])  
    
    # Water availiability over ocean
    Var["W_ocean"] = np.array([1.])    

    #--------------------------
    # Sea ice related constants
    #--------------------------
    
    # Sea ice thickness (m)
    Var["si_ithick"] = np.zeros((Var["lat"].size)) 
    Var["si_ithick"][Var['idxsh'].astype('int')] = 2.
    Var["si_ithick"][Var['idxnh'].astype('int')] = 2.
    
    # Sea-ice specific heat capacity
    Var["sea_ice_sphc"] = np.array([2.2e3]) # (J Kg^-1 K^-1)
    
    # Sea-ice freezing temperature
    Var["Ksi"] = np.array([271.15]) # (K)

    #----------------------
    # Ice related constants
    #----------------------
    
    # Density of ice.
    Var["ice_rho"] = np.array([917.]) # (kg m^-3)
    
    # Latent heat of melting.
    Var["Lm"] = np.array([3.34e5])  # (J kg^-1) 
    
    # Melting point of snow.
    Var["K"] = np.array([273.15]) # (K)
    
    return Var


#-----------------------------------------------
# Function which returns initialized model state
#-----------------------------------------------

def initialize_state(Var, INPUT):
    
    '''
    Function that returns an initialized state of the model.
    
    '''
    
    State = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
    
    # load NorESM2
    
    #-------------------------
    # Surface temperatures (K)
    #-------------------------
    
    # atmospheric temperatures
    #-------------------------
    
    # average
    State['Ta'] = np.zeros((Var["lat"].size)) + (np.sin(np.linspace(0, np.pi, Var["lat"].size))*30)+273 
         
    # land.
    State["Tal"] = State["Ta"].copy()
    State["Tal"][np.asarray(Var["idxao"], dtype = "int")] = np.NaN
    
    # ocean.
    State["Tao"] = State["Ta"].copy()
    State["Tao"][np.asarray(Var["idxal"], dtype = "int")] = np.NaN
    
    # land temperatures
    #------------------
    
    State['Tl'] = np.zeros((Var["lat"].size)) + (np.sin(np.linspace(0, np.pi, Var["lat"].size))*30)+273 
    State["Tl"][np.asarray(Var["idxao"], dtype = "int")] = np.NaN
    
    # ocean temperature
    #------------------
    
    # [ocean model composed of six layers. Initialize temperatures for each.]
    
    # LAYER 1
    To1 = np.zeros((Var["lat"].size)) + (np.sin(np.linspace(0, np.pi, Var["lat"].size))*30)+273 
    To1[To1 < Var["Ksi"]] = Var["Ksi"]                    # no freezing ocean.
    To1[np.asarray(Var["idxal"], dtype = "int")] = np.NaN # no ocean in South

    # LAYER 2 
    To2 = np.copy(To1) - 0.1 
    To2[np.asarray(Var["idxnocs"], dtype = "int")] = np.NaN # no ocean in South
    To2[np.asarray(Var["idxnocn"], dtype = "int")] = np.NaN # no deep ocean in North
    
    # LAYER 3
    To3 = np.copy(To1) - 0.2
    To3[np.asarray(Var["idxnocs"], dtype = "int")] = np.NaN # no ocean in South
    To3[np.asarray(Var["idxnocn"], dtype = "int")] = np.NaN # no deep ocean in North
    
    # LAYER 4 
    To4 = np.copy(To1) - 0.3
    To4[np.asarray(Var["idxnocs"], dtype = "int")] = np.NaN # no ocean in South
    To4[np.asarray(Var["idxnocn"], dtype = "int")] = np.NaN # no deep ocean in North
    
    # LAYER 5 
    To5 = np.copy(To1) - 0.4
    To5[np.asarray(Var["idxnocs"], dtype = "int")] = np.NaN # no ocean in South
    To5[np.asarray(Var["idxnocn"], dtype = "int")] = np.NaN # no deep ocean in North
    
    # LAYER 6
    To6 = np.copy(To1) - 0.5
    To6[np.asarray(Var["idxnocs"], dtype = "int")] = np.NaN # no ocean in South
    To6[np.asarray(Var["idxnocn"], dtype = "int")] = np.NaN # no deep ocean in North
    
    # Concatenate ocean layers.
    State["To"] = np.concatenate((To1, To2, To3, To4, To5, To6))
    
    # Surface ocean temperature (includes sea-ice temperature.)
    #----------------------------------------------------------
    
    State['Tos'] =  np.zeros((Var["lat"].size)) + (np.sin(np.linspace(0, np.pi, Var["lat"].size))*30)+273 
    State["Tos"][np.asarray(Var["idxal"], dtype = "int")] = np.NaN # no ocean in South
    
    # Mean ocean temperature
    State["mot"] = np.zeros((1))
    
    # average surface temperature
    #----------------------------
    
    State['Ts'] = np.zeros((Var["lat"].size)) + (np.sin(np.linspace(0, np.pi, Var["lat"].size))*30)+273 
    
    # temperature of sea-ice
    #-----------------------
    
    # Temperature of sea-ice
    State["Tsi"] = np.zeros((Var["lat"].size)) + 273
    i = np.where(State["Tos"] < Var["Ksi"])
    State["Tsi"][i[0]] = State["Tos"][i[0]]
    
    # NaN values where no ocean in the south
    State["Tsi"][np.asarray(Var["idxal"], dtype = "int")] = np.NaN # no ocean in South
    
    
    # Lapse rate modified temperatures
    #---------------------------------
    
    State['Tax']  = State['Ta'].copy()
    State['Talx'] = State['Ta'].copy()
    State['Tlx']  = State['Ta'].copy()
    
    #------------------------------------
    # Shortwave radiative fluxes (W m^-2)
    #------------------------------------

    # Over land
    #----------
    State["rsut_land"] = np.zeros((Var["lat"].size)) # Upwards shortwave radiation at TOA 
    State["rsds_land"] = State["rsut_land"].copy() # Downwards shortwave radiation at surface 
    State["rsus_land"] = State["rsut_land"].copy()        # Upwards shortwave radiation at surface 
    State["rsdtnet_land"] = State["rsut_land"].copy()      # Net downwards shortwave radiation at TOA
    State["rsdsnet_land"] = State["rsut_land"].copy()      # Net downwards shortwave radiation at surface
    State["rsatmnet_land"] = State["rsut_land"].copy()     # Absorbed shortwave radiation by atmosphere
    

    # Over ocean
    #-----------
    State["rsut_ocean"] = State["rsut_land"].copy()    # Upwards shortwave radiation at TOA 
    State["rsds_ocean"] = State["rsut_land"].copy()    # Downwards shortwave radiation at surface 
    State["rsus_ocean"] = State["rsut_land"].copy()    # Upwards shortwave radiation at surface 
    State["rsdtnet_ocean"] = State["rsut_land"].copy() # Net downwards shortwave radiation at TOA
    State["rsdsnet_ocean"] = State["rsut_land"].copy() # Net downwards shortwave radiation at surface
    State["rsatmnet_ocean"] = State["rsut_land"].copy()# Absorbed shortwave radiation by atmosphere
    
    
    # Weighted Average
    #-----------------
    State["rsut"] = State["rsut_land"].copy()     # Upwards shortwave radiation at TOA 
    State["rsds"] = State["rsut_land"].copy()     # Downwards shortwave radiation at surface 
    State["rsus"] = State["rsut_land"].copy()     # Upwards shortwave radiation at surface 
    State["rsdtnet"] = State["rsut_land"].copy()  # Net downwards shortwave radiation at TOA
    State["rsdsnet"] = State["rsut_land"].copy()  # Net downwards shortwave radiation at surface
    State["rsatmnet"] = State["rsut_land"].copy() # Absorbed shortwave radiation by atmosphere

    #-----------------------------------
    # Longwave radiative fluxes (W m^-2)
    #-----------------------------------

    # Over land
    #----------
    State["rlus_land"] = State["rsut_land"].copy()     # Upwards longwave radiation at surface
    State["rlds_land"] = State["rsut_land"].copy()     # Downwards longwave radiation at surface 
    State["rlut_land"] = State["rsut_land"].copy()     # Upwards longwave radiation at TOA 
    State["rldsnet_land"] = State["rsut_land"].copy()  # Net downwards longwave radiation at surface
    State["rlatmnet_land"] = State["rsut_land"].copy() # Absorbed longwave radiation by atmosphere

    # Over ocean
    #-----------
    State["rlus_ocean"] = State["rsut_land"].copy()     # Upwards longwave radiation at surface 
    State["rlds_ocean"] = State["rsut_land"].copy()     # Downwards longwave radiation at surface 
    State["rlut_ocean"] = State["rsut_land"].copy()     # Upwards longwave radiation at TOA 
    State["rldsnet_ocean"] = State["rsut_land"].copy()  # Net downwards longwave radiation at surface
    State["rlatmnet_ocean"] = State["rsut_land"].copy() # Absorbed longwave radiation by atmosphere


    # Weighted average
    #-----------------
    State["rlus"] = State["rsut_land"].copy()     # Upwards longwave radiation at surface
    State["rlds"] = State["rsut_land"].copy()     # Downwards longwave radiation at surface 
    State["rlut"] = State["rsut_land"].copy()     # Upwards longwave radiation at TOA 
    State["rldsnet"] = State["rsut_land"].copy()  # Net downwards longwave radiation at surface
    State["rlatmnet"] = State["rsut_land"].copy() # Absorbed longwave radiation by atmosphere
    
    #-----------------------------------
    # Total radiative fluxes (in W m^-2)
    #-----------------------------------
    
    # Over land
    #----------
    State["rtnet_land"]   = State["rsut_land"].copy()# Net downwards radiation at TOA
    State["rsnet_land"]   = State["rsut_land"].copy()# Net downwards radiation at surface
    State["ratmnet_land"] = State["rsut_land"].copy()# Radiation absorbed by atmosphere
    
    # Over ocean
    #-----------
    State["rtnet_ocean"]   = State["rsut_land"].copy()# Net downwards radiation at TOA
    State["rsnet_ocean"]   = State["rsut_land"].copy()# Net downwards radiation at surface
    State["ratmnet_ocean"] = State["rsut_land"].copy()# Radiation absorbed by atmosphere
    
    # Weighted average
    #-----------------
    
    State["rtnet"]   = State["rsut_land"].copy()                     # Net downwards radiation at TOA
    State["rsnet"]   = State["rsut_land"].copy()                     # Net downwards radiation at surface
    State["ratmnet"] = State["rsut_land"].copy()                    # Radiation absorbed by atmosphere
    State["rtimb"]   = State["rsut_land"].copy()   
    
    #-------
    # Albedo
    #-------
    
    # over land
    State["alpha_land"]  = np.copy(State["rsut_land"]) + 0.2
    
    # over ocean
    State["alpha_ocean"] = np.copy(State["rsut_land"]) + 0.2 # Ocean
    
    #------------
    # Evaporation
    #------------
    
    # Over land
    State["evap_flux_land"]  = np.copy(State["rsut_land"])  # Evaporation flux (kg m^-2 s^-1)
    State["lhf_evap_land"]   = np.copy(State["rsut_land"])  # Evaporative heat flux (W m^-2)
    
    # Over ocean
    State["evap_flux_ocean"] = np.copy(State["rsut_land"]) # Evaporation flux (kg m^-2 s^-1)
    State["lhf_evap_ocean"]  = np.copy(State["rsut_land"]) # Evaporative heat flux (W m^-2)
    
    # Weighted average
    State["evap_flux"]  = np.copy(State["rsut_land"])  # Evaporation flux (kg m^-2 s^-1)
    State["lhf_evap"]   = np.copy(State["rsut_land"])  # Evaporative heat flux (W m^-2)
    
    #-------------------
    # Sensible heat flux
    #-------------------
    
    # Over land.
    State["shf_land"]  = np.copy(State["rsut_land"]) # Vertical SHF (W m^-2)
    
    # Over ocean.
    State["shf_ocean"] = np.copy(State["rsut_land"]) # Vertical SHF (W m^-2)
    
    # Weighted average.
    State["shf"]       = np.copy(State["rsut_land"]) # Vertical SHF (W m^-2)
    
    #---------------------------
    # Atmospheric heat transport
    #---------------------------
    
    # moist static energy
    #--------------------
    
    # Northward heat transport (J s^-1)
    State["mse_north"] = np.zeros((Var["latb"].size), dtype="f8")
    
    # Heat convergence/divergence (W m^-2)
    State["mse_conv"] = np.copy(State["rsut_land"]) 
    
    # hadley cell contribution
    #-------------------------
    
    # Northward dry transport (J s^-1)
    State["hadley_dry"] = State["mse_north"].copy()
    
    # Northward latent transport (J s^-1)
    State["hadley_latent"] = State["mse_north"].copy()
    
    # eddy contribution
    #------------------
    
    # Northward dry transport (J s^-1)
    State["eddy_dry"] = State["mse_north"].copy()
    
    # Northward latent transport (J s^-1)
    State["eddy_latent"] = State["mse_north"].copy()
    
    
    # total dry and latent transport
    #-------------------------------
    
    # Northward dry transport (J s^-1)
    State["dry_north"] = State["mse_north"].copy()
    
    # Dry convergence/divergence (W m^-2)
    State["dry_conv"] = np.copy(State["rsut_land"]) 
    
    # Northward latent transport (J s^-1)
    State["latent_north"] = State["mse_north"].copy()
    
    # Latent convergence/divergence (W m^-2)
    State["latent_conv"] = np.copy(State["rsut_land"]) 
      
    #----------------------------------------
    # Atmospheric (specific) humidity (kg/kg)
    #----------------------------------------
    
    # average
    State['Q'] = np.zeros((Var["lat"].size)) + (np.sin(np.linspace(0, np.pi, Var["lat"].size))*0.0175)
         
    # land.
    State["Q_land"] = np.copy(State["Q"]) 
    
    # ocean
    State["Q_ocean"] = np.copy(State["Q"])
    
    #---------------------------
    # Precipitation and snowfall
    #---------------------------
    
    # Over land
    State["precip_flux_land"]        = np.copy(State["rsut_land"])  # Precipitation flux (kg m^-2 s^-1))
    State["precip_rate_land"]        = np.copy(State["rsut_land"])  # Precipitation rate (m/year)
    State["snowfall_fraction_land"]  = np.copy(State["rsut_land"])  # Snowfall fraction of precipitation
    State["snowfall_flux_land"]      = np.copy(State["rsut_land"])  # Snowfall flux (kg m^-2 s^-1)
    State["snowfall_rate_land"]      = np.copy(State["rsut_land"])  # Snowfall rate (m/year)
    State["lhf_precip_land"]         = np.copy(State["rsut_land"])  # Precipitation heat flux (W m^-2)
    State["lhf_snowfall_land"]       = np.copy(State["rsut_land"])  # Snowfall heat flux (W m^-2)
    
    # Over ocean
    State["precip_flux_ocean"]       = np.copy(State["rsut_land"]) # Precipitation flux (kg m^-2 s^-1))
    State["precip_rate_ocean"]       = np.copy(State["rsut_land"]) # Precipitation rate (m/year)
    State["snowfall_fraction_ocean"] = np.copy(State["rsut_land"]) # Snowfall fraction of precipitation
    State["snowfall_flux_ocean"]     = np.copy(State["rsut_land"]) # Snowfall flux (kg m^-2 s^-1)
    State["snowfall_rate_ocean"]     = np.copy(State["rsut_land"]) # Snowfall rate (m/year)
    State["lhf_precip_ocean"]        = np.copy(State["rsut_land"]) # Precipitation heat flux (W m^-2)
    State["lhf_snowfall_ocean"]      = np.copy(State["rsut_land"]) # Snowfall heat flux (W m^-2)
    State["sc_edge_nh"]              = np.zeros((1))               # Snow cover edge in NH    
    State["sc_edge_sh"]              = np.zeros((1))               # Snow cover edge in SH   
    
    # Weighted average 
    State["precip_flux"]       = np.copy(State["rsut_land"]) # Precipitation flux (kg m^-2 s^-1))
    State["precip_rate"]       = np.copy(State["rsut_land"]) # Precipitation rate (m/year)
    State["snowfall_fraction"] = np.copy(State["rsut_land"]) # Snowfall flux (kg m^-2 s^-1)
    State["snowfall_flux"]     = np.copy(State["rsut_land"]) # Snowfall flux (kg m^-2 s^-1)
    State["snowfall_rate"]     = np.copy(State["rsut_land"]) # Snowfall rate (m/year)
    State["lhf_precip"]        = np.copy(State["rsut_land"]) # Precipitation heat flux (W m^-2)
    State["lhf_snowfall"]      = np.copy(State["rsut_land"]) # Snowfall heat flux (W m^-2)
    
    #--------
    # Sea-ice 
    #--------
    
    State["si_fraction"]    = np.copy(State["rsut_land"])     # Sea ice fraction of surface ocean
    State["si_thick"]       = np.copy(State["rsut_land"])     # Sea ice thickness (m)
    State["si_volume"]      = np.copy(State["rsut_land"])     # Sea ice volume (m^3)
    State["si_area_nh"]     = np.zeros((1))                   # Sea ice areal extent in NH (m^2)
    State["si_area_sh"]     = np.zeros((1))                   # Sea ice areal extent in SH (m^2)
    State["si_melt_flux"]   = np.copy(State["rsut_land"])     # Sea ice melt flux (W m^-2)
    
    # Set to sea-ice cover where surface ocean temperature is below zero.
    idx = np.where(State["Tos"] < Var["Ksi"])
    State["si_fraction"][idx[0]] = 1.
    State["si_thick"][idx[0]]    = 2.
    State["si_volume"][idx[0]]   = Var["oarea"][idx[0]] * State["si_thick"][idx[0]]
    
    # Sea-ice areal extent
    State["si_area_nh"][0] = np.nansum(State["si_fraction"][Var["idxnh"].astype('i8')]*Var["oarea"][Var["idxnh"].astype('i8')])
    State["si_area_sh"][0] = np.nansum(State["si_fraction"][Var["idxsh"].astype('i8')]*Var["oarea"][Var["idxsh"].astype('i8')])
    
    # Set to NaN where no ocean in South.
    State["si_fraction"][np.asarray(Var["idxal"], dtype = "int")] = np.NaN 
    State["si_thick"][np.asarray(Var["idxal"], dtype = "int")]    = np.NaN 
    State["si_volume"][np.asarray(Var["idxal"], dtype = "int")]   = np.NaN 
    State["si_edge_nh"] = np.zeros((1))     
    State["si_edge_sh"] = np.zeros((1))    

    #-------------
    # Ocean fluxes
    #-------------
    
    State["advf"]        = np.zeros((Var["olatb"].size*6))       # surface advective heat flux (W)
    State["advf_conv"]   = np.zeros((Var["olat"].size*6))        # surface advective heat flux convergence (W/m2)
    State["hdiffs"]      = np.zeros((Var["olatb"].size*6))       # surface diffusion heat flux (W)
    State["hdiffs_conv"] = np.zeros((Var["olat"].size*6))        # surface diffusion heat flux convergence(W/m2)
    State["hdiffi"]      = np.zeros((Var["olatb"].size*6))       # interior diffusion heat flux (W)
    State["hdiffi_conv"] = np.zeros((Var["olat"].size*6))        # interior diffusion heat flux convergence(W/m2)
    State["vdiff"]       = np.zeros((Var["olat"].size*7))        # vertical diffusion heat flux (W)
    State["vdiff_conv"]  = np.zeros((Var["olat"].size*6))        # vertical diffusion heat flux convergence (W/m2)
    State["vadvf"]       = np.zeros((Var["olat"].size*7))        # vertical advective heat flux (W)
    State["vadvf_conv"]  = np.zeros((Var["olat"].size*6))        # vertical advective heat flux convergence (W/m2)
    State["ocean_conv"]   = np.zeros((Var["olat"].size*6))       # total ocean heat transport convergence (W/m2)
    
    #------------------
    # Ocean overturning
    #------------------
    
    # index for centre of ocean circulation boundaries
    State["idxcos"] = np.array([INPUT['occ'][0]], dtype = 'f8')
    
    # vertical ocean velocities
    
    if State["idxcos"] == 0:
        
        State["ww"] = Var["w0"]
    
    if State["idxcos"] == -5:
        
        State["ww"] = Var["w5s"]
        
    if State["idxcos"] == -10:
        
        State["ww"] = Var["w10s"]
    
    if State["idxcos"] == -15:
        
        State["ww"] = Var["w15s"]
    
    # vertical ocean diffusion ceofficients
    State["dz"]  = np.zeros((Var["olat"].size)) + INPUT["dz0"]
    
    #----------------------
    # Land snow/ice melting
    #----------------------
    
    State["melt"]      = np.copy(State["rsut_land"]) # Melting (m/year)
    State["melt_flux"] = np.copy(State["rsut_land"]) # Melt flux (W m^-2)
    
    #---------------------
    # Surface Mass Balance
    #---------------------
    
    State["SMB"] = np.copy(State["rsut_land"]) # m/day
    
    #---------------------------------
    # Global mean surface temperatures
    #---------------------------------
    
    State["Tag"]   = np.array([global_mean(State["Ta"], Var)])
    State["Tagx"]  = np.array([global_mean(State["Ta"], Var)])
    State["Talgx"] = np.array([global_mean(State["Ta"], Var)])
    State["Taogx"] = np.array([global_mean(State["Ta"], Var)])
    State["Tsg"]   = np.array([global_mean(State["Ts"], Var)])
    State["Tsgx"]  = np.array([global_mean(State["Ts"], Var)])
    State["Tosg"]  = np.array([global_mean(State["Tos"], Var)])
    State["Tlgx"]  = np.array([global_mean(State["Tl"], Var)])
    
    #-----------------------
    # Snow and ice over land
    #-----------------------
    
    # Ice thickness
    #--------------
    
    State["ice_thick"] = np.zeros((Var["lat"].size))

    # Snowpack
    #---------
    
    State["snow_thick"]           = np.copy(State["rsut_land"]) # Snow thickness (m)
    State["snow_fraction_land"]   = np.copy(State["rsut_land"]) # Snow fraction over land
    
    return State
    
    