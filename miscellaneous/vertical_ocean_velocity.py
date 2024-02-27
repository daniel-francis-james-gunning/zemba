# -*- coding: utf-8 -*-
"""
@author: Daniel Gunning (University of Bergen)

Prescribed vertical ocean velocities in PyEBM. 

Following Bintanja (1997), velocity distribution is chosen for a mean upwelling
rate of 4 m/yr in the upwelling region (50S to 60N), and a global mean vertical
velocity of 0 m/yr
"""

import numpy as np


# declare resolution
res = 5

# declare centre of ocean circulation
cnt = -5

#----------------------
# Define some constants
#----------------------

# latitude
#---------

if res == 5:
    
    # 5 res
    lat = np.arange(-87.5, 87.5+5., 5.)
    
if res == 2.5:

    # 2.5 res
    lat = np.arange(-88.75, 88.75+2.5, 2.5) 
    
if res == 1:

    # 5 res
    lat = np.arange(-89.5, 89.5, 1.)


# latitude in radians
#--------------------

latr = np.deg2rad(lat)

# latitude spacing in radians
#----------------------------

dlatr = np.diff(latr)[0]

# radius of earth
#----------------

rearth = 6371e3

# surface area of earth (in m^2)
#-------------------------------

sarea = (# east-west width of latitude band
        (2*np.pi*0.7*rearth*np.cos(latr)) 
        * 
        # north-south length of latitude band 
        (rearth*dlatr)
        )  

# initialize upwelling array
#---------------------------

w = np.zeros((lat.size))


#----------------------------------------------------------------------
# Define indices for ocean upwelling and downwelling in each hemisphere
#----------------------------------------------------------------------

# index for whole ocean grid 
#---------------------------

if res == 5:
    
    oi = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -67.5)[0][0], np.where(lat == 77.5)[0][0] + 1)
   
if res == 2.5:
    
    oi = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -68.75)[0][0], np.where(lat == 78.75)[0][0] + 1)
    
if res == 1:
    
    oi = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -69.5)[0][0], np.where(lat == 79.5)[0][0] + 1)


# index of downwelling in the SH (from 70S to 50S)
#-------------------------------------------------

if res == 5:
    
    dwn_sh = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -67.5)[0][0], np.where(lat == -52.5)[0][0] + 1)
   
if res == 2.5:
    
    dwn_sh = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -68.75)[0][0], np.where(lat == -51.25)[0][0] + 1)
    
if res == 1:
    
    dwn_sh = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -69.5)[0][0], np.where(lat == -50.5)[0][0] + 1)
    
    
# index of upwelling in the SH (from 50S to eq.)
#-----------------------------------------------

if res == 5:
    
    if cnt == 0:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -47.5)[0][0], np.where(lat == -2.5)[0][0] + 1)

    if cnt == -5:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -47.5)[0][0], np.where(lat == -7.5)[0][0] + 1)
    
    if cnt == -10:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -47.5)[0][0], np.where(lat == -12.5)[0][0] + 1)
    
    if cnt == -15:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -47.5)[0][0], np.where(lat == -17.5)[0][0] + 1)
        
if res == 2.5:
    
    if cnt == 0:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -48.75)[0][0], np.where(lat == -1.25)[0][0] + 1)

    if cnt == -5:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -48.75)[0][0], np.where(lat == -6.25)[0][0] + 1)
    
    if cnt == -10:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -48.75)[0][0], np.where(lat == -11.25)[0][0] + 1)
    
    if cnt == -15:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -48.75)[0][0], np.where(lat == -16.25)[0][0] + 1)
        
if res == 1:
    
    if cnt == 0:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -49.5)[0][0], np.where(lat == -0.5)[0][0] + 1)

    if cnt == -5:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -49.5)[0][0], np.where(lat == -5.5)[0][0] + 1)
    
    if cnt == -10:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -49.5)[0][0], np.where(lat == -10.5)[0][0] + 1)
    
    if cnt == -15:
    
        upw_sh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -49.5)[0][0], np.where(lat == -15.5)[0][0] + 1)
        
        
# index of upwelling in the NH (from eq. to 60N)
#-----------------------------------------------

if res == 5:
    
    if cnt == 0:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == 2.5)[0][0], np.where(lat == 57.5)[0][0] + 1)

    if cnt == -5:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -2.5)[0][0], np.where(lat == 57.5)[0][0] + 1)
    
    if cnt == -10:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -7.5)[0][0], np.where(lat == 57.5)[0][0] + 1)
    
    if cnt == -15:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -12.5)[0][0], np.where(lat == 57.5)[0][0] + 1)
        
if res == 2.5:
    
    if cnt == 0:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == 1.25)[0][0], np.where(lat == 58.75)[0][0] + 1)

    if cnt == -5:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -3.75)[0][0], np.where(lat == 58.75)[0][0] + 1)
    
    if cnt == -10:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -8.75)[0][0], np.where(lat == 58.75)[0][0] + 1)
    
    if cnt == -15:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -13.75)[0][0], np.where(lat == 58.75)[0][0] + 1)
        
if res == 1:
    
    if cnt == 0:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == 0.5)[0][0], np.where(lat == 59.5)[0][0] + 1)

    if cnt == -5:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -4.5)[0][0], np.where(lat == 59.5)[0][0] + 1)
    
    if cnt == -10:
    
        upw_nh= np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -9.5)[0][0], np.where(lat == 59.5)[0][0] + 1)
    
    if cnt == -15:
    
        upw_nh = np.arange( # southern boundary          |     northern boundary
                        np.where(lat == -14.5)[0][0], np.where(lat == 59.5)[0][0] + 1)
        
        

# index of upwelling in either hemisphere (from 50S to 60N)
#----------------------------------------------------------

if res == 5:
    
    upw = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -47.5)[0][0], np.where(lat == 57.5)[0][0] + 1)
   
if res == 2.5:
    
    upw = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -48.75)[0][0], np.where(lat == 58.75)[0][0] + 1)
    
if res == 1:
    
    upw = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == -49.5)[0][0], np.where(lat == 59.5)[0][0] + 1)
    
# index of downwelling in the NH (from 60N to 80N)
#-------------------------------------------------

if res == 5:
    
    dwn_nh = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == 62.5)[0][0], np.where(lat == 77.5)[0][0] + 1)
   
if res == 2.5:
    
    dwn_nh = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == 61.25)[0][0], np.where(lat == 78.75)[0][0] + 1)
    
if res == 1:
    
    dwn_nh = np.arange( # southern boundary          |     northern boundary
                    np.where(lat == 60.5)[0][0], np.where(lat == 79.5)[0][0] + 1)
    




         

# #----------------------------------------------------------------------
# # Define indices for ocean upwelling and downwelling in each hemisphere
# #----------------------------------------------------------------------

# # index for ocean grid 
# oi = np.arange(
    
#                  # southern boundary
#                  #------------------
                 
#                   np.where(lat == -67.5)[0][0],  # 5. res (c. 5S)
#                  # np.where(lat == -68.75)[0][0], # 2.5 res (c. 5S)
#                  # np.where(lat == -68.75)[0][0], # 2.5 res (c. 0S)
#                  # np.where(lat == -69.5)[0][0],  # 1. res  (c. 0S)
#                  # np.where(lat == -69.5)[0][0],  # 1. res  (c. 5S)
                  
#                  # northern boundary 
#                   np.where(lat == 77.5)[0][0] + 1  # 5. res (c. 5S)
#                    # np.where(lat == 78.75)[0][0] + 1 # 2.5 res (c. 5S)
#                  # np.where(lat == 78.75)[0][0] + 1 # 2.5 res (c. 0S)
#                  # np.where(lat == 79.5)[0][0] + 1  # 1. res  (c. 0S)
#                  # np.where(lat == 79.5)[0][0] + 1  # 1. res  (c. 5S)
    
#                  )

# # index of downwelling in the SH (from 70S to 50S)
# dwn_sh = np.arange(
    
#                  # southern boundary 
#                   np.where(lat == -67.5)[0][0],  # 5. res (c. 5S)
#                    # np.where(lat == -68.75)[0][0], # 2.5 res (c. 5S)
#                  # np.where(lat == -68.75)[0][0], # 2.5 res (c. 0S)
#                  # np.where(lat == -69.5)[0][0],  # 1. res  (c. 0S)
#                  # np.where(lat == -69.5)[0][0],  # 1. res  (c. 5S)
                  
#                  # northern boundary 
#                   np.where(lat == -52.5)[0][0] + 1  # 5. res (c. 5S)
#                    # np.where(lat == -51.25)[0][0] + 1 # 2.5 res (c. 5S)
#                  # np.where(lat == -51.25)[0][0] + 1 # 2.5 res (c. 0S)
#                  # np.where(lat == -50.5)[0][0] + 1  # 1. res  (c. 0S)
#                  # np.where(lat == -50.5)[0][0] + 1  # 1. res  (c. 5S)
    
#                  )

# # index of upwelling in the SH (from 50S to eq.)
# upw_sh = np.arange(
    
#                  # southern boundary 
#                   np.where(lat == -47.5)[0][0],  # 5. res (c. 5S)
#                    # np.where(lat == -48.75)[0][0], # 2.5 res (c. 5S)
#                  # np.where(lat == -48.75)[0][0], # 2.5 res (c. 0S)
#                  # np.where(lat == -49.5)[0][0],  # 1. res  (c. 0S)
#                  # np.where(lat == -49.5)[0][0],  # 1. res  (c. 5S)
                  
#                  # northern boundary 
#                   np.where(lat == -17.5)[0][0] + 1  # 5. res (c. 5S)
#                    # np.where(lat == -6.25)[0][0] + 1 # 2.5 res (c. 5S)
#                  # np.where(lat == -1.25)[0][0] + 1 # 2.5 res (c. 0S)
#                  # np.where(lat == -0.5)[0][0] + 1  # 1. res  (c. 0S)
#                  # np.where(lat == -5.5)[0][0] + 1  # 1. res  (c. 5S)
    
#                  )


# # index of upwelling in the NH (from eq. to 60N)
# upw_nh = np.arange(
    
#                  # southern boundary 
#                   np.where(lat == -12.5)[0][0],  # 5. res (c. 5S)
#                    # np.where(lat == -3.75)[0][0], # 2.5 res (c. 5S)
#                  # np.where(lat == 1.25)[0][0],  # 2.5 res (c. 0S)
#                  # np.where(lat == 0.5)[0][0],   # 1. res  (c. 0S)
#                  # np.where(lat == -4.5)[0][0],  # 1. res  (c. 5S)
                  
#                  # northern boundary 
#                   np.where(lat == 57.5)[0][0] + 1  # 5. res (c. 5S)
#                    # np.where(lat == 58.75)[0][0] + 1 # 2.5 res (c. 5S)
#                  # np.where(lat == 58.75)[0][0] + 1 # 2.5 res (c. 0S)
#                  # np.where(lat == 59.5)[0][0] + 1  # 1. res  (c. 0S)
#                  # np.where(lat == 59.5)[0][0] + 1  # 1. res  (c. 5S)
    
#                  )

# # index of upwelling in either hemisphere (from 50S to 60N)
# upw = np.arange(
    
#                  # southern boundary  
#                   np.where(lat == -47.5)[0][0],   # 5. res (c. 5S)
#                    # np.where(lat == -48.75)[0][0],  # 2.5 res (c. 5S)
#                  # np.where(lat == -48.75)[0][0],  # 2.5 res (c. 0S)
#                  # np.where(lat == -49.5)[0][0],   # 1. res  (c. 0S)
#                  # np.where(lat == -49.5)[0][0],   # 1. res  (c. 5S)
                  
#                  # northern boundary 
#                   np.where(lat == 57.5)[0][0] + 1  # 5. res (c. 5S)
#                    # np.where(lat == 58.75)[0][0] + 1 # 2.5 res (c. 5S)
#                  # np.where(lat == 58.75)[0][0] + 1 # 2.5 res (c. 0S)
#                  # np.where(lat == 59.5)[0][0] + 1  # 1. res  (c. 0S)
#                  # np.where(lat == 59.5)[0][0] + 1  # 1. res  (c. 5S)
    
#                  )


# # index of downwelling in the NH (from 60N to 80N)
# dwn_nh = np.arange(
    
#                  # southern boundary 
#                   np.where(lat == 62.5)[0][0],  # 5. res (c. 5S)
#                    # np.where(lat == 61.25)[0][0], # 2.5 res (c. 5S)
#                  # np.where(lat == 61.25)[0][0], # 2.5 res (c. 0S)
#                  # np.where(lat == 60.5)[0][0],  # 1. res  (c. 0S)
#                  # np.where(lat == 60.5)[0][0],  # 1. res  (c. 5S)
                  
#                  # northern boundary 
#                   np.where(lat == 77.5)[0][0] + 1  # 5. res (c. 5S)
#                    # np.where(lat == 78.75)[0][0] + 1 # 2.5 res (c. 5S)
#                  # np.where(lat == 78.75)[0][0] + 1 # 2.5 res (c. 0S)
#                  # np.where(lat == 79.5)[0][0] + 1  # 1. res  (c. 0S)
#                  # np.where(lat == 79.5)[0][0] + 1  # 1. res  (c. 5S)
    
#                  )



#----------------------------------------------------
# Calculating upwelling rates in SH (from 50S to eq.)
#----------------------------------------------------

"""
The sum of the upwelling rates (w) in the SH, weighted by surface area, is assumed
equal to 4 m/yr (based on Bintanja 1997):
    
    Sum[ A(lat) * w(lat) ] / A_total = 4 m/yr
    
where A is the surface area and A_total is the total surface area for the 
upwelling region.

It is also assumed that w at 50S = 0 m/yr, which linearly increases
to the equator.

W at any latitude can therefore be approximated with the equation of a line:
    
    w = mx
    
where m is the linear rate of change of w and x is an arbitraty x-axis.
By setting w=0 where the arbitrary x-axis, x=0, the y-intercept dissapears.

Therefore, m can be solved with:
    
    Sum[ A(lat) * mx] / A_total = 4 m/yr
    
    m * Sum[ A(lat) * x] / A_total = 4 m/yr
    
    m = 4 m/yr * A_total / (Sum[A(lat) * x])
    
"""


# weighted surface area for upwelling region
warea_upw_sh = sarea[upw_sh] / np.sum(sarea[upw_sh])

# mean upwelling rate
refupw = 4 # m/yr

# arbitrary x-axis for equation of line from 50S to eq.
xaxis_upw_sh = np.arange(0, upw_sh.size)

# rate of change (gradient) of upwelling from 50S to eq.
grad_upw_sh = refupw /np.sum(xaxis_upw_sh * warea_upw_sh)

# upwelling with latitude based on linear relation
w[upw_sh] = grad_upw_sh * xaxis_upw_sh 


#------------------------------------------------------
# Calculating downwelling rates in SH (from 70S to 50S)
#------------------------------------------------------

# Calculate the average downwelling in the SH to match the average upwelling
# rate in the SH of 4 m/yr.

refdw_sh = -(np.sum(sarea[upw_sh])*refupw)/np.sum(sarea[dwn_sh])

# weighted surface area for upwelling region
warea_dwn_sh = sarea[dwn_sh] / np.sum(sarea[dwn_sh])

# arbitrary x-axis for equation of line from 50S to eq.
xaxis_dwn_sh = np.arange(-dwn_sh.size, 0)

# rate of change (gradient) of upwelling from 50S to eq.
grad_dwn_sh = refdw_sh /np.sum(xaxis_dwn_sh * warea_dwn_sh)

# upwelling with latitude based on linear relation
w[dwn_sh] = grad_dwn_sh * xaxis_dwn_sh 



#----------------------------------------------------
# Calculating upwelling rates in NH (from eq. to 60N)
#----------------------------------------------------

# weighted surface area for upwelling region
warea_upw_nh = sarea[upw_nh] / np.sum(sarea[upw_nh])

# mean upwelling rate
refupw = 4 # m/yr

# arbitrary x-axis for equation of line from 50S to eq.
xaxis_upw_nh = np.arange(-upw_nh.size, 0)

# rate of change (gradient) of upwelling from 50S to eq.
grad_upw_nh = refupw /np.sum(xaxis_upw_nh * warea_upw_nh)

# upwelling with latitude based on linear relation
w[upw_nh] = grad_upw_nh * xaxis_upw_nh 


#------------------------------------------------------
# Calculating downwelling rates in NH (from 70S to 50S)
#------------------------------------------------------

# Calculate the average downwelling in the SH to match the average upwelling
# rate in the SH of 4 m/yr.

refdw_nh = -(np.sum(sarea[upw_nh])*refupw)/np.sum(sarea[dwn_nh])

# weighted surface area for upwelling region
warea_dwn_nh = sarea[dwn_nh] / np.sum(sarea[dwn_nh])

# arbitrary x-axis for equation of line from 50S to eq.
xaxis_dwn_nh = np.arange(0, dwn_nh.size)

# rate of change (gradient) of upwelling from 50S to eq.
grad_dwn_nh = refdw_nh /np.sum(xaxis_dwn_nh * warea_dwn_nh)

# upwelling with latitude based on linear relation
w[dwn_nh] = grad_dwn_nh * xaxis_dwn_nh 


#---------------------------------------
# Average upwelling rate from 50S to 60N
#---------------------------------------

average_upwelling_rate = np.average(w[upw], weights = sarea[upw]/(sarea[upw].sum()))
print("Average Upwelling Rate: ", round(average_upwelling_rate, 6))

#---------------------------------------
# Average upwelling rate from 70S to 50S
#---------------------------------------

average_dwsh_rate = np.average(w[dwn_sh], weights = sarea[dwn_sh]/(sarea[dwn_sh].sum()))
print("Average Downwelling Rate in SH: ", round(average_dwsh_rate, 2))

#---------------------------------------
# Average upwelling rate from 60N to 80N
#---------------------------------------

average_dwnh_rate = np.average(w[dwn_nh], weights = sarea[dwn_nh]/(sarea[dwn_nh].sum()))
print("Average Downwelling Rate in NH: ", round(average_dwnh_rate, 2))

#------------------------------
# Global average upwelling rate
#------------------------------

global_mean_vertical_velocity = np.average(w, weights = sarea/(sarea.sum()))
print("Global Mean: ", round(global_mean_vertical_velocity, 2))

#-----------------------
# Tidy downwelling array
#-----------------------
 
w = w[oi]


import matplotlib.pyplot as plt

plt.plot(w)




