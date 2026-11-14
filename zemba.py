# -*- coding: utf-8 -*-
import numpy as np
from numba.core import types
from numba.typed import Dict
from numba import njit, prange
from radiation_core import shortwave_radiation, longwave_radiation
from hydrology_core import surface_evaporation, precipitation
from atm_transport_core import atm_transport
from albedo_core import ocean_albedo, land_albedo_temperature, land_albedo_snowcover
from snow_core import snowfall, snow_accumulation, snow_melt
from seaice_core import sea_ice
from ocean_transport_core import ocean_fluxes, horizontal_ocean_velocities

float_array_2d = types.float64[:,:]

@njit(nogil=True)
def combine(lf_i, of_i, x_l, x_o):
    xl = 0.0 if np.isnan(x_l) else x_l
    xo = 0.0 if np.isnan(x_o) else x_o
    return lf_i*xl + of_i*xo
    

@njit(nogil=True)
def global_nanmean(variable, area):
    num = 0.0    # accumulator for sum(variable*area)
    den = 0.0    # accumulator for sum(area) over non-NaN cells
    for i in range(variable.size):
        v = variable[i]
        if not np.isnan(v):
            num += v * area[i]
            den += area[i]
    return num / den

parallel = False
@njit(parallel=parallel, nogil=True)
def run_model(State, INPUT, Settings, Var, I, znth,
              fixed_si_fraction=None, 
              fixed_si_temp=None,
              fixed_sc_fraction=None,
              fixed_oalbedo=None,
              fixed_lalbedo=None): 
    
    StateYear = Dict.empty(key_type=types.unicode_type, value_type=float_array_2d)
    
   
    
    # Fill StateYear with State variables
    for keys, values in State.items(): # for every variable in State
        StateYear[keys] = np.zeros(( len(values), 365 ))
    StateYear['I'] =  np.zeros(( Var['lat'].shape[0], 365 ))
    
    
    lat = Var['lat']
    nlat = State["Tl"].size
    nlatb = nlat+1
    nyrs = int(INPUT['nyrs'][0])
    idxo = Var["idxolyr1"].astype(np.int64)
    lf   = INPUT['land_fraction']
    of   = 1.0 - lf
    latbr = Var['latbr']
    dlatr = Var['dlatr'][0]
    latb = Var['latb']
    r_earth= Var['r_earth'][0]
    
    # legacy.. needed now but should be removed at some point
    olat = Var["olat"]
    olatb = Var["olatb"]
    olatr = Var["olatr"]
    olatbr = Var["olatbr"]
    
    idxcos = State['idxcos'][0]
    
    sbi = np.where(Var["latb"] == -70.)[0][0]
    nbi = np.where(Var["latb"] == 80.)[0][0]
    co = np.where(Var["latb"] == idxcos)[0][0]    # full-grid latb, was olatb
    
    

    # some constants
    #---------------
    
    # settings
    hydro = Settings['hydro'][0]
    height = Settings['height'][0]
    hadley_cell = Settings['hadley_cell'][0]
    seaice = Settings['seaice'][0]
    
    # temperature
    tl   = State["Tl"]
    to   = State["To"]
    ta   = State["Ta"]
    ts   = State["Ts"]
    tal  = State["Tal"]
    tao  = State["Tao"]
    tos  = State["Tos"]
    tagx = State["Tagx"][0]
    tsi  = State['Tsi']
    
    # humidity
    q  = State["Q"]
    ql = State["Q_land"]
    qo = State["Q_ocean"]
    
    # sea ice
    si_thick = State['si_thick']
    si_frac = State['si_fraction']
    si_vol  = State['si_volume']
    si_meltflux  = State['si_melt_flux']
    snflf_o = State['snowfall_flux_ocean']
    # snow cover over land
    snowfrac_l  = State['snow_fraction_land']
    snowthick_l = State['snow_thick']
    
    # atm. heat transport
    mse_north = State["mse_north"] 
    FS        = State["dry_north"]
    FL        = State["latent_north"]
    dry_conv  = State["dry_conv"]
    lat_conv  = State["latent_conv"]
    
    # ocn. heat transport  
    advf   = State['advf']
    hdiffs = State['hdiffs']
    hdiffi = State['hdiffi']
    vdiff  = State['vdiff']
    vadvf  = State['vadvf']
    advf_conv   = State['advf_conv']
    hdiffs_conv = State['hdiffs_conv']
    hdiffi_conv = State['hdiffi_conv']
    vdiff_conv  = State['vdiff_conv']
    vadvf_conv  = State['vadvf_conv']
    

    
    # constants
    sigma = Var["sigma"][0]
    co2   = INPUT["co2"][0]
    lv = Var['Lv'][0]
    lm = Var['Lm'][0]
    K = Var['K'][0]
    icerho = Var['ice_rho'][0]
    wl = Var['W_land'][0]
    wo = Var['W_ocean'][0]
    secs_in_day = Var["secs_in_day"][0]
    land_hc = Var["land_hc"][0]
    ocean_hc1 = Var["ocean_hc1"][0]
    atm_hc = Var["atm_hc"][0]
    atmspc = Var["atm_sphc"][0]
    atmrho = Var["atm_rho"][0]
    atmdepth = Var["atm_depth"][0]
    waterrho = Var["water_rho"][0]
    lmask  = Var["land_mask"]
    omask  = Var["ocean_mask"]
    gms_factor = INPUT["hadley_constant"][0]
    EQ = int(Var['idxeq'][0])
    szb=Var['latb'].size
    area=Var['sarea']
    oarea=Var['oarea']
    ksi=Var['Ksi'][0]
    rholm=lm*icerho
    si_sphc = Var["sea_ice_sphc"][0]
    si_ithick = Var["si_ithick"]
    rhospc = Var["ocean_rho"][0]*Var["ocean_sphc"][0]
    od = Var["odepth"]
    oddiff = Var["odepth_diff"]

    ow = np.zeros(latb.size)              # latb.size = nlatb = 37
    ow[sbi:sbi+Var["ow"].size] = Var["ow"]
    rearthlat = r_earth*dlatr
    oceanhc = Var['ocean_hc']
    
    oa_full = np.zeros(Var["lat"].size)
    oa_full[sbi:sbi+Var["oa"].size] = Var["oa"]
    
    

    # atmospheric transport geometry prefactors
    gface = 2.0*np.pi*np.cos(latbr)*atmrho*atmdepth/dlatr
    if hadley_cell == 1:
        w_face = np.exp((-np.sin(latbr) ** 2) / (0.3 ** 2))
    else:
        w_face = np.zeros(latbr.size)
        
    res = INPUT['res'][0]      
    if res == 1.0:
        small_step = 15
    elif res == 2.5:
        small_step = 3
    elif res == 5.0:
        small_step = 1
    else:
        small_step = 1  
        
    dt_sub = secs_in_day/small_step
    
    
    mse_acc = np.zeros(nlatb)
    FS_acc = np.zeros(nlatb)
    FL_acc = np.zeros(nlatb)
    dry_conv_acc = np.zeros(nlat)
    lat_conv_acc = np.zeros(nlat)
    
    
    advf_acc = np.zeros(nlatb*6)
    hdiffs_acc = np.zeros(nlatb*6)
    hdiffi_acc = np.zeros(nlatb*6)
    vadvf_acc = np.zeros(nlat*7)
    vdiff_acc = np.zeros(nlat*7)
    
    advf_conv_acc = np.zeros(nlat*6)
    hdiffs_conv_acc = np.zeros(nlat*6)
    hdiffi_conv_acc = np.zeros(nlat*6)
    vadvf_conv_acc = np.zeros(nlat*6)
    vdiff_conv_acc = np.zeros(nlat*6)
    
    
   

    # input
    clt_l   = INPUT["ccl"]
    clt_o   = INPUT["cco"]
    z_l     = INPUT["land_height"]
    z       = Var["mean_height"]
    z_o     = Var["ocean_height"]
    clt_tau = INPUT["tau"][0]
    e_clt   = Var["cloud_emissivity"][0]
    ghg_factor = INPUT["GHG_amp"][0]
    tbhfcl = INPUT['tbhfcl'][0]
    tbhfco = INPUT['tbhfco'][0]
    r = INPUT['r'][0]
    D = INPUT['dt']
    alphasimn = INPUT["alphasimn"][0]
    alphasimx = INPUT["alphasimx"][0]
    alphag = INPUT["alphag"][0]
    alphacs = INPUT["alphas"][0]
    alphaws = INPUT["alphaws"][0]
    alphai = INPUT["alphai"][0]
    ice_frac = INPUT["ice_fraction"]
    dh = INPUT["dh"][0]
    dz = State["dz"][0]
    do = INPUT["do"][0]
    lapsel = 0.0065 * z_l
    lapse = 0.0065 * z
    

    
    # lapse rates
    lra = 0.0065*z
    

    # settings
    hydro = Settings['hydro'][0]
    height = Settings['height'][0]
    snow = Settings['snow'][0]
    
    
    # calc, ocean velocities
    
    # run on olat grid (unchanged from original)
    f  = np.zeros((olat.size))+0.7
    fb =  np.zeros((olatb.size))+0.7
    w  = State["ww"]
    
    
    
    
    u1_o, u2_o, w_o = horizontal_ocean_velocities(olat, olatb, olatr, olatbr, 
                                                  dlatr, idxcos, 80.0, -70.0, 
                                                  f, fb, r_earth, w, od)
    

    
    # pad all three to full grid
    sbi = np.where(latb == -70.)[0][0]
    nbi = np.where(latb == 80.)[0][0]  
    u1 = np.zeros(latb.size)   # u1/u2 are FACE quantities (olatb-shaped)
    u2 = np.zeros(latb.size)
    ww = np.zeros(lat.size)    # ww is a CELL quantity (olat-shaped)
    u1[sbi:sbi+u1_o.size] = u1_o
    u2[sbi:sbi+u2_o.size] = u2_o
    ww[sbi:sbi+w_o.size]  = w_o



    for year in range(nyrs):
        
        for day in range(365):
            
            I_day = I[:,day]
            znth_day = znth[:,day]
            
            for i in prange(nlat):
                
                # ====================
                # --------------------
                # --- Albedo block ---
                # --------------------
                # ====================
                
                # ----
                # LAND
                # ----
                
                if fixed_lalbedo is None: 
                
                    # no snow
                    if snow == -1:
                        alpha_l = alphag
                        
                    # snow cover and albedo as function of surface temperature
                    elif snow == 0:
                        alpha_l, snowfrac_l[i] = land_albedo_temperature(tal[i], tl[i], z_l[i], 
                                                             alphacs, alphaws, 
                                                             znth_day[i], alphai,
                                                             alphag, ice_frac[i], 
                                                             lmask[i])
                    
                    # albedo as function of snow cover
                    elif snow > 0: # as a function of snowfal/melt
                        
                            alpha_l = land_albedo_snowcover(tl[i], z_l[i], alphacs, 
                                                            alphaws, znth_day[i],
                                                            alphai, alphag, 
                                                            snowfrac_l[i], 
                                                            ice_frac[i], lmask[i])
                else:
    
                    alpha_l = fixed_lalbedo[i,day]
                    

                
                # -----
                # OCEAN
                # -----
                
                if fixed_oalbedo is None: 
                
                    alpha_o = ocean_albedo(si_frac[i], si_meltflux[i], alphasimn, 
                                              alphasimx, znth_day[i], omask[i])
                else:
                    alpha_o = fixed_oalbedo[i,day]
                
                
                # =======================
                # -----------------------
                # --- Radiation block ---
                # -----------------------
                # =======================
                
                # ----------------
                # SHORTWAVE — LAND
                # ----------------
                
                rsut_l, rsds_l, rsus_l = shortwave_radiation(clt_l[i], tal[i], alpha_l,
                                                          alpha_l, znth_day[i], z_l[i],
                                                          I_day[i], clt_tau)

                # -----------------
                # SHORTWAVE — OCEAN
                # -----------------
                
                rsut_o, rsds_o, rsus_o = shortwave_radiation(clt_o[i], tao[i], alpha_o, 
                                                          alpha_o, znth_day[i], z_o[i],
                                                          I_day[i], clt_tau)
                
                # ---------------
                # LONGWAVE — LAND
                # ---------------
                
                rlus_l, rlds_l, rlut_l = longwave_radiation(sigma, co2, tal[i], tl[i], tagx, 
                                                      z_l[i], clt_l[i], e_clt,  
                                                      ghg_factor)
                
                # ----------------
                # LONGWAVE — OCEAN
                # ----------------
                
                rlus_o, rlds_o, rlut_o = longwave_radiation(sigma, co2, tao[i], tos[i],
                                                      tagx, z_o[i], clt_o[i], e_clt, 
                                                      ghg_factor)
                
                
                # -------------------- 
                # NET SHORTWAVE FLUXES
                # --------------------
                
                # Net downwards shortwave radiation at the top of the atmosphere
                rsdtnet_l = I_day[i] - rsut_l
                rsdtnet_o = I_day[i] - rsut_o
            
                # Net downwards shortwave radiation at the surface
                rsdsnet_l = rsds_l - rsus_l
                rsdsnet_o = rsds_o - rsus_o
            
                # Net shortwave radiation absorbed by the atmosphere
                rsatmnet_l = rsdtnet_l - rsdsnet_l
                rsatmnet_o = rsdtnet_o - rsdsnet_o
            
                # -------------------
                # NET LONGWAVE FLUXES
                # -------------------
                    
                # Net downwards longwave radiation at the surface
                rldsnet_l = rlds_l - rlus_l
                rldsnet_o = rlds_o - rlus_o

                # Longwave radiation absorbed by atmosphere
                rlatmnet_l = rlus_l - (rlds_l  + rlut_l)
                rlatmnet_o = rlus_o - (rlds_o + rlut_o)
                
                # ----------------
                # NET TOTAL FLUXES
                # ----------------
            
                # Net downwards radiation at the top of the atmosphere
                rtnet_l = rsdtnet_l - rlut_l
                rtnet_o = rsdtnet_o - rlut_o
                
                # Net downwards radiation at the surface
                rsnet_l = rsdsnet_l + rldsnet_l
                rsnet_o = rsdsnet_o + rldsnet_o
                
                # Radiation absorbed by the atmosphere
                ratmnet_l = rsatmnet_l + rlatmnet_l
                ratmnet_o = rsatmnet_o + rlatmnet_o
                
                # =======================
                # -----------------------
                # --- Sensible block ---
                # -----------------------
                # =======================
                
                # ----
                # LAND
                # ----
                
                shf_l  = atmspc * atmrho * tbhfcl * (tl[i] - tal[i])
                shf_l *= lmask[i]
                
                # -----
                # OCEAN
                # -----
                
                shf_o  = atmspc * atmrho * tbhfco * (tos[i] - tao[i])
                shf_o *= omask[i]
                
                # =========================
                # -------------------------
                # --- Evaporation block ---
                # -------------------------
                # =========================
                
                # ----
                # LAND
                # ----
                
                evap_l, lhf_l = surface_evaporation(tl[i], tal[i], ql[i], z_l[i], lv, r, atmrho, tbhfcl, wl, hydro, height)
                
                # -----
                # OCEAN
                # -----
                
                evap_o, lhf_o = surface_evaporation(tos[i], tao[i], qo[i], z_o[i], lv, r, atmrho, tbhfco, wo, hydro, height)
                
                
                # =======================
                # -----------------------
                # --- Moisture source ---
                # -----------------------
                # =======================
                
                # moisture balance gain due to evaporation
                
                if (hydro == 1): # if hydrological cycle is on.
                
                    # ----
                    # LAND
                    # ----
                    
                    ql[i]  += secs_in_day / (atmrho * atmdepth) * (evap_l)
                    
                    # -----
                    # OCEAN
                    # -----
                    
                    qo[i]  += secs_in_day / (atmrho * atmdepth) * (evap_o) 
                    
                    
                # ===========================
                # ---------------------------
                # --- Precipitation block ---
                # ---------------------------
                # ===========================
                
                # Precipitation flux (kg m^-2 s^-1), precipitation rate (in m/yr)
                # and associated heat flux (in W m^-2)

                if (hydro == 1): # if hydrological cycle is on.
                
                    # ----
                    # LAND
                    # ----
                    
                    precipf_l, precipr_l, preciplhf_l = precipitation(tal[i], ql[i], z_l[i], lv, r, atmrho, atmdepth, secs_in_day, lmask[i], waterrho, height) 
                    
                    # -----
                    # OCEAN
                    # -----
                    
                    precipf_o, precipr_o, preciplhf_o = precipitation(tao[i], qo[i], z_o[i], lv, r, atmrho, atmdepth, secs_in_day, omask[i], waterrho, height) 
                    
   
                # =====================               
                # ---------------------
                # --- Moisture sink ---
                # ---------------------
                # =====================
                
                if (hydro == 1): # if hydrological cycle is on.
                
                    # moisture balance loss due to precipitation
                
                    # ----
                    # LAND
                    # ----
                    
                    ql[i]  -= secs_in_day / (atmrho * atmdepth) * (precipf_l)
                    
                    # -----
                    # OCEAN
                    # -----
                    
                    qo[i]  -= secs_in_day / (atmrho * atmdepth) * (precipf_o) 
                    
                    
                # ================               
                # ----------------
                # --- Snowfall ---
                # ----------------
                # ================
                
                if (hydro == 1) and (snow > 0):
                    
                    # ----
                    # LAND
                    # ----
                    
                    snflfrac_l, snflf_l, snflr_l, snfllhf_l = snowfall(tal[i], precipf_l, z_l[i], lm, icerho)
                    
                    
                    # -----
                    # OCEAN
                    # -----
                    
                    snflfrac_o, snflf_o[i], snflr_o, snfllhf_o = snowfall(tao[i], precipf_o, z_o[i], lm, icerho)
                    
                # ===========================             
                # ---------------------------
                # --- Snow acc. over land ---
                # ---------------------------
                # ===========================
                
                if (hydro == 1) and (snow > 0):
                    
                    
                    if fixed_sc_fraction is None: # snow is calculated internally...
                    
                        snowfrac_l[i], snowthick_l[i] = snow_accumulation(snowfrac_l[i], snowthick_l[i], area[i], snow, 
                                          snflr_l, snflfrac_l, precipf_l, icerho)
                        
                        
                    else: # snow is prescribed externally (for sensitivity experiments)
                    
                        snowfrac_l[i] = fixed_sc_fraction[i,day]
                        
                    
                   
                # ==========================
                # --------------------------
                # --- Update temperature ---
                # --------------------------
                # ==========================
                
                # surface land
                #tl[i]  += secs_in_day / land_hc * (rsnet_l)
                tl[i]  += secs_in_day / land_hc * (rsnet_l - shf_l - lhf_l)
             
                # surface ocean
                to[i]  += secs_in_day / ocean_hc1 * (rsnet_o - shf_o - lhf_o)
                #to[i]  += secs_in_day / ocean_hc1 * (rsnet_o)
                
                
                if (hydro == 1): # if hydrological cycle is on.
        
                    # atmopshere over land
                    #tal[i]  += secs_in_day / atm_hc * (ratmnet_l)
                    tal[i]  += secs_in_day / atm_hc * (ratmnet_l + shf_l + preciplhf_l  + snfllhf_l)
                    
                    # atmosphere over ocean
                    #tao[i]  += secs_in_day / atm_hc * (ratmnet_o)
                    tao[i]  += secs_in_day / atm_hc * (ratmnet_o + shf_o + preciplhf_o  + snfllhf_o)
        
                elif (hydro == 0): # if hydrological cycle is off.
        
                    # atmopshere over land
                    #tal[i]  += secs_in_day / atm_hc * (ratmnet_l)
                    tal[i]  += secs_in_day / atm_hc * (ratmnet_l + shf_l + lhf_l)
                    
                    # atmosphere over ocean
                    #tao[i]  += secs_in_day / atm_hc * (ratmnet_o)
                    tao[i]  += secs_in_day / atm_hc * (ratmnet_o + shf_o + lhf_o)
                    
                    
                # ===========================
                # ---------------------------
                # --- Snow melt over land ---
                # ---------------------------
                # ===========================
                
                

                if (hydro == 1) and (snow > 0):
                    
                    if fixed_sc_fraction is None: # snow is calculated internally...
                        
                        snowthick_l[i], snowfrac_l[i], tl[i], melt, pmelt, meltflux = snow_melt(tl[i], z_l[i], snow, snowthick_l[i], K, land_hc,
                                                                                         lm, icerho, snowfrac_l[i], secs_in_day, lmask[i])
                    
                    else: # snow is prescribed externally (for sensitivity experiments)
                    
                        continue
                    
                        
                
                # ========================================================
                # --------------------------------------------------------
                # --- Update zonal temperature prior to heat transport ---
                # --------------------------------------------------------
                # ========================================================
                
                    
                # zonal air temperature
                ta[i]  = combine(lf[i], of[i], tal[i], tao[i])
                
                # zonal specific humidity
                if (hydro == 1): 
                    q[i]  = combine(lf[i], of[i], ql[i], qo[i])
                  
                
                # ===================================
                # -----------------------------------
                # --- Save results from last year ---
                # -----------------------------------
                # ===================================
                
                if (StateYear is not None) and (year == nyrs - 1):
                    
                    # albedo
                    # ------
                    
                    StateYear["alpha_land"][i, day] = alpha_l
                    StateYear["alpha_ocean"][i, day] = alpha_o
                    
                    # shortwave radiation
                    # -------------------
                    
                    # Upward shortwave radiation at the top of the atmosphere
                    StateYear["rsut_land"][i, day] = rsut_l
                    StateYear["rsut_ocean"][i, day] = rsut_o
                    StateYear["rsut"][i, day] = combine(lf[i], of[i], rsut_l, rsut_o)
                    
                    
                    # Downward shortwave radiation at the surface
                    StateYear["rsds_land"][i, day] = rsds_l
                    StateYear["rsds_ocean"][i, day] = rsds_o
                    StateYear["rsds"][i, day] = combine(lf[i], of[i], rsds_l, rsds_o)
                    
                    # Upward shortwave radiation at the surface
                    StateYear["rsus_land"][i, day] = rsus_l
                    StateYear["rsus_ocean"][i, day] = rsus_o
                    StateYear["rsus"][i, day] = combine(lf[i], of[i], rsus_l, rsus_o)
                    
                    # Net downwards shortwave radiation at the top of the atmosphere
                    StateYear["rsdtnet_land"][i, day] = rsdtnet_l
                    StateYear["rsdtnet_ocean"][i, day] = rsdtnet_o
                    StateYear["rsdtnet"][i, day] = combine(lf[i], of[i], rsdtnet_l, rsdtnet_o)
    
                    # Net downwards shortwave radiation at the surface
                    StateYear["rsdsnet_land"][i, day] = rsdsnet_l
                    StateYear["rsdsnet_ocean"][i, day] = rsdsnet_o
                    StateYear["rsdsnet"][i, day] = combine(lf[i], of[i], rsdsnet_l, rsdsnet_o)
                    
                    # Shortwave radiation absorbed by atmosphere
                    StateYear["rsatmnet_land"][i, day]  = rsatmnet_l 
                    StateYear["rsatmnet_ocean"][i, day] = rsatmnet_o
                    StateYear["rsatmnet"][i, day] = combine(lf[i], of[i], rsatmnet_l, rsatmnet_o)
                    
                    # longwave radiation
                    # ------------------
                    
                    # Upward longwave radiation at the top of the atmosphere
                    StateYear["rlut_land"][i, day] = rlut_l
                    StateYear["rlut_ocean"][i, day] = rlut_o
                    StateYear["rlut"][i, day] = combine(lf[i], of[i], rlut_l, rlut_o)
                   
                    # Downward longwave radiation at the surface
                    StateYear["rlds_land"][i, day] = rlds_l
                    StateYear["rlds_ocean"][i, day] = rlds_o
                    StateYear["rlds"][i, day] = combine(lf[i], of[i], rlds_l, rlds_o)
         
                    # Upward longwave radiation at the surface
                    StateYear["rlus_land"][i, day] = rlus_l
                    StateYear["rlus_ocean"][i, day] = rlus_o
                    StateYear["rlus"][i, day] = combine(lf[i], of[i], rlus_l, rlus_o)
                   
                    # Net downwards longwave radiation at the surface
                    StateYear["rldsnet_land"][i, day] = rldsnet_l
                    StateYear["rldsnet_ocean"][i, day] = rldsnet_o
                    StateYear["rldsnet"][i, day] = combine(lf[i], of[i], rldsnet_l, rldsnet_o)
                    
                    # Longwave radiation absorbed by atmosphere
                    StateYear["rlatmnet_land"][i, day]  = rlatmnet_l 
                    StateYear["rlatmnet_ocean"][i, day] = rlatmnet_o
                    StateYear["rlatmnet"][i, day] = combine(lf[i], of[i], rlatmnet_l, rlatmnet_o)
                   
                    # total radiation
                    # ---------------
                    
                    # Net downwards radiation at the top of the atmosphere
                    StateYear["rtnet_land"][i, day]  = rtnet_l
                    StateYear["rtnet_ocean"][i, day] = rtnet_o
                    StateYear["rtnet"][i, day] = combine(lf[i], of[i], rtnet_l, rtnet_o)
                    
                    # Net downwards radiation at the surface
                    StateYear["rsnet_land"][i, day]  = rsnet_l
                    StateYear["rsnet_ocean"][i, day] = rsnet_o
                    StateYear["rsnet"][i, day] = combine(lf[i], of[i], rsnet_l, rsnet_o)
                    
                    # Radiation absorbed by atmosphere
                    StateYear["ratmnet_land"][i, day]  = ratmnet_l
                    StateYear["ratmnet_ocean"][i, day] = ratmnet_o
                    StateYear["ratmnet"][i, day] = combine(lf[i], of[i], ratmnet_l, ratmnet_o)
                    
                    
                    
                    # sensible heat fluxes
                    # --------------------
                    
                    StateYear["shf_land"][i, day]  = shf_l
                    StateYear["shf_ocean"][i, day] = shf_o
                    StateYear["shf"][i, day] = combine(lf[i], of[i], shf_l, shf_o)
                    
                    # surface evaporation
                    # -------------------
                    
                    StateYear["evap_flux_land"][i, day]  = evap_l
                    StateYear["evap_flux_ocean"][i, day]  = evap_o
                    StateYear["evap_flux"][i, day] = combine(lf[i], of[i], evap_l, evap_o)
                    
                    StateYear["lhf_evap_land"][i, day]  = lhf_l
                    StateYear["lhf_evap_ocean"][i, day]  = lhf_o
                    StateYear["lhf_evap"][i, day] = combine(lf[i], of[i], lhf_l, lhf_o)                 
                    
                    # precipitation
                    # -------------
                    
                    if hydro==1:
                    
                        StateYear["precip_flux_land"][i, day]  = precipf_l
                        StateYear["precip_flux_ocean"][i, day]  = precipf_o
                        StateYear["precip_flux"][i, day] = combine(lf[i], of[i], precipf_l, precipf_o)
                        
                        StateYear["precip_rate_land"][i, day]  = precipr_l
                        StateYear["precip_rate_ocean"][i, day]  = precipr_o
                        StateYear["precip_rate"][i, day] = combine(lf[i], of[i], precipr_l, precipr_o)
                        
                        StateYear["lhf_precip_land"][i, day]  = preciplhf_l
                        StateYear["lhf_precip_ocean"][i, day]  = preciplhf_o
                        StateYear["lhf_precip"][i, day] = combine(lf[i], of[i], preciplhf_l, preciplhf_o)
                    
                    # snowfall
                    # --------
                    
                    if (hydro == 1) and (snow > 0):
                    
                        # snowfall fraction
                        StateYear["snowfall_fraction_land"][i, day] = snflfrac_l
                        StateYear["snowfall_fraction_ocean"][i, day] = snflfrac_o
                        StateYear["snowfall_fraction"][i, day] = combine(lf[i], of[i], snflfrac_l, snflfrac_o)
                        
                        # snowfall flux
                        StateYear["snowfall_flux_land"][i, day] = snflf_l
                        StateYear["snowfall_flux_ocean"][i, day] = snflf_o[i]
                        StateYear["snowfall_flux"][i, day] = combine(lf[i], of[i], snflf_l, snflf_o[i])
                        
                        # snowfall rate
                        StateYear["snowfall_rate_land"][i, day] = snflr_l
                        StateYear["snowfall_rate_ocean"][i, day] = snflr_o
                        StateYear["snowfall_rate"][i, day] = combine(lf[i], of[i], snflr_l, snflr_o)
                        
                        # latent heat of snowfall
                        StateYear["lhf_snowfall_land"][i, day] = snfllhf_l
                        StateYear["lhf_snowfall_ocean"][i, day] = snfllhf_o
                        StateYear["lhf_snowfall"][i, day] = combine(lf[i], of[i], snfllhf_l, snfllhf_o)
                    
                    # snow cover
                    # ----------
                    
                    if hydro==1:
                        
                        if snow == 0:
                            
                            StateYear["snow_fraction_land"][i, day] = snowfrac_l[i]
                            
                        if snow > 0:
                            
                            StateYear["snow_fraction_land"][i, day] = snowfrac_l[i]
                            StateYear["snow_thick"][i, day] = snowthick_l[i]
                            
                            if fixed_sc_fraction is None:
                                
                                StateYear['melt'][i, day] = melt
                                StateYear['pmelt'][i, day] = pmelt
                            
                            
                    
            # =================================================================
            # =================================================================
            # --------------------- TRANSPORT SUBSTEP  ------------------------
            # =================================================================
            # =================================================================
            
            # --- ocean heat transport ---
            # ----------------------------
            
            advf_conv, hdiffs_conv, hdiffi_conv, vadvf_conv, vdiff_conv = ocean_fluxes(to, co, sbi, nbi, rhospc, oa_full, od, oddiff,
                                                                                       ow, rearthlat, ww, u1, u2, dh, dz, do, nlat, 
                                                                                       nlatb, advf, hdiffs, hdiffi, vdiff, vadvf, 
                                                                                       advf_conv, hdiffs_conv, hdiffi_conv, vadvf_conv, 
                                                                                       vdiff_conv)
            
            
            # --- convergence ---
            # -------------------
            
            for i in range(nlat):
                

                if (i >= sbi) and (i < nbi):
                  
                   
                    # ocean
                    for j in range(0, 6): # for each layer
                    
                        to[j*nlat+i] += (secs_in_day/oceanhc[j*nlat+i]) * (advf_conv[j*nlat+i] + vadvf_conv[j*nlat+i] + hdiffs_conv[j*nlat+i] + hdiffi_conv[j*nlat+i] + vdiff_conv[j*nlat+i])
                        
                
            if (StateYear is not None) and (year == nyrs - 1):
                
                StateYear["advf"][:,day] = advf
                StateYear["hdiffs"][:,day]  = hdiffs
                StateYear["hdiffi"][:,day] = hdiffi
                StateYear["vadvf"][:,day] = vadvf
                StateYear['vdiff'][:,day] = vdiff
                
                StateYear["advf_conv"][:,day] = advf_conv
                StateYear["hdiffs_conv"][:,day]  = hdiffs_conv
                StateYear["hdiffi_conv"][:,day] = hdiffi_conv
                StateYear["vadvf_conv"][:,day] = vadvf_conv
                StateYear['vdiff_conv'][:,day] = vdiff_conv
                            
                         
           
         
            

            for n in range(small_step):
                
                
                    # --- atmospheric transport ---
                    # -----------------------------
                
                    dry_conv, lat_conv = atm_transport(ta, q, atmspc, lv, z, 
                                                       r, EQ, szb, gface, D, w_face, 
                                                       gms_factor, nlat, area, hydro, 
                                                       FS, FL, mse_north, 
                                                       dry_conv, lat_conv)
                    
                    

                    
                    # --- convergence ---
                    # -------------------
                    
                    for i in range(nlat):
                        
                        # atmosphere
                        ta[i] += dt_sub/atm_hc * dry_conv[i]
                        if hydro == 1:
                            q[i] += dt_sub/(atmrho*atmdepth) * (lat_conv[i]/lv)
                            
                            
                    if (StateYear is not None) and (year == nyrs - 1):
                       
                        mse_acc[:] += mse_north
                        FS_acc[:]  += FS
                        FL_acc[:]  += FL
                        dry_conv_acc[:] += dry_conv
                        lat_conv_acc[:] += lat_conv
                        
                        
            # after sub-step loop, last year only:
            if (StateYear is not None) and (year == nyrs - 1):
                
                StateYear["mse_north"][:,day] = mse_acc / small_step
                StateYear["dry_conv"][:,day]  = dry_conv_acc / small_step
                StateYear["latent_conv"][:,day] = lat_conv_acc / small_step
                StateYear["dry_north"][:,day]  = FS_acc / small_step
                StateYear["latent_north"][:,day] = FL_acc / small_step
               
                # zero accumulators for next day
                mse_acc[:]=0.0; FS_acc[:]=0.0; FL_acc[:]=0.0
                dry_conv_acc[:]=0.0; lat_conv_acc[:]=0.0

                
            # ======================================================
            # ------------------------------------------------------
            # ---Reset land/ocean to zonal-mean (post-transport) ---
            # ------------------------------------------------------
            # ====================================================== 
            
            
            # zonal air temperature
            tal[:] = ta;  tao[:] = ta
            # zonal specific humidity
            if hydro == 1:
                ql[:] = q;  qo[:] = q
                

            for i in prange(nlat):
                
                # =====================================
                # -------------------------------------
                # --- Sea ice formation and melting ---
                # -------------------------------------
                # =====================================
                
                
                #------------------------------
                # Sea ice formation and melting
                #------------------------------
                            
                if seaice == 1 and omask[i] == 1.0: # if sea ice is on.
                
                    if fixed_si_fraction is None and fixed_si_temp is None: # snow is calculated internally...
                
                        to[i], tsi[i], si_thick[i], si_vol[i], si_frac[i], si_meltflux[i] = sea_ice(to[i], si_vol[i], tsi[i], si_frac[i], rholm, ksi, ocean_hc1, oarea[i], secs_in_day, snflf_o[i], icerho, si_sphc, si_ithick[i], lm, omask[i])  
                        
                    else: # snow is prescribed externally (for sensitivity experiments)
                     
                        si_frac[i] = fixed_si_fraction[i,day]
                        tsi[i]     = fixed_si_temp[i,day]
                     
            
                # =================================
                # ---------------------------------
                # --- Surface ocean temperature ---
                # ---------------------------------
                # =================================
                
                tos[i] = to[i]
                
                if seaice == 1:
                    
                    if si_frac[i] == 1.0:
                        
                        tos[i] = tsi[i]
                        
                        
                # ===========================
                # ---------------------------
                # --- Surface temperature ---
                # ---------------------------
                # ===========================
                        
                
                ts[i] = combine(lf[i], of[i], tl[i], tos[i])
                
                
                # ======================
                # ----------------------
                # --- Save last year ---
                # ----------------------
                # ======================
                
                if (StateYear is not None) and (year == nyrs - 1):
                    
                    if seaice == 1:
                        StateYear["Tsi"][i, day] = tsi[i]
                        StateYear["si_fraction"][i, day] = si_frac[i]
                        StateYear["si_volume"][i, day] = si_vol[i]
                        StateYear["si_melt_flux"][i, day] = si_meltflux[i]
            
                    
                    
                    
                
                             
            # ================================
            # --------------------------------
            # --- Global mean temperatures ---
            # --------------------------------
            # ================================
                
            tagx = global_nanmean(ta-lra, area)

            # ===================================
            # -----------------------------------
            # --- Save results from last year ---
            # -----------------------------------
            # ===================================
            
            if (StateYear is not None) and (year == nyrs - 1):
               
                StateYear["Ta"][:, day]  = ta
                StateYear["Tax"][:, day] = ta - lapse
                StateYear["Tal"][:, day] = tal
                StateYear["Talx"][:, day] = tal - lapsel
                StateYear["Tao"][:, day] = tao
                StateYear["Tl"][:, day]  = tl
                StateYear["Tlx"][:, day]  = tl - lapsel
                StateYear["To"][:, day]  = to
                StateYear["Tos"][:, day] = tos
                StateYear["I"][:, day] = I_day

                
                if hydro == 1:
                    StateYear["Q"][:, day]       = q
                    StateYear["Q_land"][:, day]  = ql
      
         
    
    return State, StateYear

