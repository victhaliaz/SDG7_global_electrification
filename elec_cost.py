# -*- coding: utf-8 -*-
"""
Code for calculing the LCOE for each electrification option
"""

import pandas as pd
import numpy as np
import os
import xarray as xr 
from scipy import interpolate
import matplotlib.pyplot as plt

os.getcwd()
DIR = "X:\\user\\zapatacasv\\sdg7\\scripts"   # Set the running directory here
os.chdir(DIR)
import elec_network_nogph as en
dat = en.dat #import own codes after defining home dir
gph = en.gph
import time
start_time = time.time()


# %%
# !*******************************************************************************
# !**************************** Electrification Costs ****************************
# !*******************************************************************************


# !*******************************************************
# !************* GRID Electrification COSTS**************
# !******************************************************


def fun():

        
    #Diesel retail price from worldbank.org for 2010 per country
    path ='..\\data\\diesel_retailprice_2010_dolperl_worldbank.nc'
    diesel_retail = xr.open_dataarray(path)

    #loading potentials and CF data obatined from by Gernaat et al.,
    #loadfactors
    path2 =     "..\\data\\PV_LF_5min-Hadgem_his_1970-2000.nc"
    loadfactor = xr.open_dataset(path2)
    path2 =     "..\\data\\Wind_LF_5min-Hadgem_his_1970-2000.nc"
    loadfactor['wind_Loadfac'] = xr.open_dataarray(path2)
    #LCOE for hydropower 
    path2 =  "..\\data\\COE_HYD_HadGEM-ES_hist_5min.nc"
    potential = xr.open_dataarray(path2)
    potential = potential.to_dataset(name = 'Hydro_COE')#$/kWh/cell/
    #potentials
    path2 =  "..\\data\\Wind_techpot_5min-Hadgem_his_1970-2000.nc"
    potential['wind_techpot'] = xr.open_dataarray(path2)     #kWh/cell/year
    path2 =  "..\\data\\PV_techpot_5min-Hadgem_his_1970-2000.nc"
    potential['PV_techpot'] = xr.open_dataarray(path2)   # (kWh/cell/year)   
    path2 =  "..\\data\\Hydro_techpot_5min-Hadgem_his_1970-2000.nc"
    potential['hydro_techpot'] = xr.open_dataarray(path2)*1e6   #kWh/cell/year
    print('ec: potential data loaded')
    
    ##! The annuallized cost of internal and extrnal network components per grid cell in $/kwh
    a = en.a.TDCapCostInt.to_dataset()
    a['TDCapCostExtPerKWh'] = en.a.TDCapCostExt.copy()
    a['TDCapCostIntPerKWh'] = en.a.TDCapCostInt.copy()
    nrc = en.nrc
    for r in nrc:
        a['TDCapCostIntPerKWh'] = xr.where((en.a.TDCapCostInt.imager == r)&(gph.Elecuseg>1e-6), en.annuity_factor.sel(region = r)*en.a.TDCapCostInt/gph.Elecuseg,  a['TDCapCostIntPerKWh'])
        a['TDCapCostExtPerKWh'] = xr.where((en.a.TDCapCostExt.imager == r)&(gph.Elecuseg>1e-6), en.annuity_factor.sel(region = r)*en.a.TDCapCostExt/gph.Elecuseg, a['TDCapCostExtPerKWh']) 

    #! Recurring annual cost of central grid $/KWh
    a['RecurringAnnCost'] = dat.var_cost_perkwh_avg.sum(dim = 'ntc2')*dat.elec_prod_share.sum(dim = 'ntc2')

    #! The internal and External grid extension costs including increase of power generation 
    a['TotCostperkwh_Internal'] =   xr.full_like(a['TDCapCostIntPerKWh'], np.nan)

    for r in nrc:
            a['TotCostperkwh_Internal'] = xr.where(a.TDCapCostIntPerKWh.imager == r, a.TDCapCostIntPerKWh + dat.system_total_cost_perkwh.sel(region = r, tres = 3), a['TotCostperkwh_Internal'] )

    a['TotalCostperkWh_Grid'] =  a['TotCostperkwh_Internal'] + a['TDCapCostExtPerKWh']     #generation cost + internal (distribution and metering) + external = total cost for the central grid

    print( 'ec: grid extension cost done')
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    # %%
    # # !*******************************************************
    # # !************* Modelling of MINIGRIDS ******************
    # # !*******************************************************

    OversizeFac = 1          #! An (under)oversize factor to balance the network cost for mini-grid systems

    ###!***************** SOLAR PV MINIGRIDS ******************
    #loadfactor = dat.loadfactor
    loadfactor = loadfactor.where((loadfactor.x == en.a.x) & (loadfactor.y == en.a.y), drop = True)   #useful when there is a filter for running only one region
    loadfactor.coords['imager'] = gph.Elecuseg.imager #IMAGER mask

    # It is assumed that electricity in househols are used 24 hours per day. For wind and solar the average load is used because the peakloads are supplied by batteries. 
    a['MiniG_ActualSyscap'] = loadfactor.PV_Loadfac.copy()
    a['MiniG_ActualSyscap'] = xr.where((loadfactor.PV_Loadfac > 1e-6) & (gph.Elecuseg.year >= dat.baseyear), gph.Elecuseg/(8760*loadfactor.PV_Loadfac*(1-dat.MiniG_distributionloss)), 0)   # (kW)  

    a['MiniG_ActualSyscap'] = a['MiniG_ActualSyscap'].expand_dims({'seis': ['PV', 'diesel', 'wind', 'hydro', 'PV-diesel', 'wind-diesel']}).copy()   #	! kw		-The system capacity [1= Solar, 2=Diesel, 3=Wind, 4=Hydro]
    a['MiniG_ActualSyscap'] = a['MiniG_ActualSyscap'].transpose('y', 'x', 'year', 'seis')

    #! $/year	-Annualized initial investment cost 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG

    annuity_factor_solar = xr.where(dat.interest > 1e-6, dat.interest /(1-(1+ dat.interest)**-dat.MiniG_TechLifetime[0]), 1)
    a['MiniG_AnnualizedCapEx'] = a['MiniG_ActualSyscap'].copy()
    for r in nrc:
        a['MiniG_AnnualizedCapEx'] = xr.where(a['MiniG_ActualSyscap'].imager == r, annuity_factor_solar.sel(region=r)*(dat.cap_cost_new.sel(ntc2=1, region=r)*a['MiniG_ActualSyscap'] + # ! The cost of panels and BOS ($)
                (dat.BatteryKwhCost[0]*dat.MiniG_StoragekwhperSyskw[0]*a['MiniG_ActualSyscap']) +   #! Storage for the evening hours and for peak-middle load demmand($)
                dat.elec_constrcost_new.sel(ntc2=1, region = r)*a['MiniG_ActualSyscap']), a['MiniG_AnnualizedCapEx'])                        #! Interest during construction ($)


    if dat.MiniG_TechLifetime[0] < dat.td_life and dat.MiniG_TechLifetime[0]>1e-6:
        PRC = (1-dat.PVBOSCost)/ dat.MiniG_TechLifetime[0] #! Panel Replacement Cost per item($)   
    else:
        PRC = 0
    if dat.PVBOSLifetime < dat.td_life and dat.PVBOSLifetime > 1e-6:
        BRC = dat.PVBOSCost/ dat.PVBOSLifetime          #! Balance replacement Cost per item ($)   
    else:
        BRC = 0
    if dat.BatteryLifetime < dat.td_life and dat.BatteryLifetime>1e-6:
        batteryRC = dat.BatteryKwhCost[0]*dat.MiniG_StoragekwhperSyskw[0]/dat.BatteryLifetime #! Battery Replacement Cost per item($)
    else:
        batteryRC = 0

    a['MiniG_RecurringAnnCost'] = a['MiniG_AnnualizedCapEx'].copy()   #! $/year	-Fixed annual cost 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG
    for r in nrc:
        a['MiniG_RecurringAnnCost'] = xr.where(a['MiniG_ActualSyscap'].imager == r, PRC*a['MiniG_ActualSyscap']*dat.cap_cost_new.sel(ntc2=1, region=r)+ #! Panel Replacement Cost ($)
                                                BRC*a['MiniG_ActualSyscap']*dat.cap_cost_new.sel(ntc2=1, region=r) + #! Balance replacement Cost ($)
                                                batteryRC*a['MiniG_ActualSyscap'] + #! Battery Replacement Cost ($)
                                                dat.om_cost_perkw_new.sel(ntc2 = 1, region = r)*a['MiniG_ActualSyscap'], a['MiniG_RecurringAnnCost'])#! ! Annual O&M cost ($)

    potential = potential.where((loadfactor.x == en.a.x) & (loadfactor.y == en.a.y), drop = True)   #used when there is a filter for running only one region
    ##! $/kwh		-The LCOE of electricity generation with minigrid 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG
    a['LCOE_MiniG_Total'] = xr.where((gph.Elecuseg > 1e-6) & (gph.Elecuseg <= potential.PV_techpot ),  (a['MiniG_AnnualizedCapEx']+ a['MiniG_RecurringAnnCost'])/gph.Elecuseg + OversizeFac*a.TDCapCostIntPerKWh, 1000)     #	! $/kwh		-Total cost of mini grid including LV network ( 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG

    a['OMShare_PV'] = xr.where((dat.om_cost_perkw_new.sel(ntc2 = 1) > 1e-6) & (dat.cap_cost_new.sel(ntc2 = 1) > 1e-6), dat.om_cost_perkw_new.sel(ntc2 = 1)/dat.cap_cost_new.sel(ntc2 = 1), 0)
    a['OMShare_wind'] = xr.where((dat.om_cost_perkw_new.sel(ntc2 = 4) > 1e-6) & (dat.cap_cost_new.sel(ntc2 = 4) > 1e-6), dat.om_cost_perkw_new.sel(ntc2 = 4)/dat.cap_cost_new.sel(ntc2 = 4), 0)   
    print( 'ec: MG-PV cost done')

    # %%
    ##!********************* DIESEL Mini-grid **********************
    DG_LoadFactor			= 	0.8								#! ESMAP (2007) TECHNICAL AND ECONOMIC ASSESSMENT OF OFF-GRID, MINI- GRID AND GRID ELECTRIFICATION TECHNOLOGIES p.52 #CF data is calculated based on baseload for 5000kw capacity
    a['timer_dieseltrend'] = dat.PriceSecFuel.sel(NS=3, NEC=2, year = gph.popg.year) /dat.PriceSecFuel.sel(NS=3, NEC=2, year = dat.t0)   
    a['DieselPrice'] = diesel_retail.expand_dims({'year': gph.popg.year})  #($/liter)

    for r in nrc:
        a['DieselPrice'] = xr.where(a.imager ==r, a['DieselPrice']*a['timer_dieseltrend'].sel(region = r), a['DieselPrice'])

    DG_HoursOperation = 24*365
    a['loaddg'] = gph.Elecuseg*1.7/DG_HoursOperation    #1.2Extra for peak load vs middle load of residential (average from India curve from Zapata et al., 2022) as there is no storage here
    a.MiniG_ActualSyscap[dict(seis= 1)] = a.loaddg/(DG_LoadFactor*(1-dat.MiniG_distributionloss))
    a['DG_FuelCOST'] = a['MiniG_ActualSyscap'][dict(seis= 1)].copy()  #!  Diesel Fuel Cost ($)
    for r in nrc:
        a['MiniG_AnnualizedCapEx'][dict(seis= 1)] = xr.where(a['MiniG_ActualSyscap'][dict(seis= 1)].imager == r, en.annuity_factor.sel(region=r)*(1.25*a['MiniG_ActualSyscap'][dict(seis= 1)]*dat.DG_CostPerSystemkw.sel(dos=1)), a['MiniG_AnnualizedCapEx'][dict(seis= 1)])  #! Diesel Genset Cost and 25% for Installation Cost ($)
    a['DG_FuelCOST'] =   a['DieselPrice']*a['MiniG_ActualSyscap'][dict(seis= 1)]*DG_HoursOperation*dat.DG_FuelConsumpperkwh[0]

    if dat.MiniG_TechLifetime[2] < dat.td_life  and dat.MiniG_TechLifetime[2]>1e-6:
        GRC = (dat.DG_CostPerSystemkw.sel(dos=1)*a['MiniG_ActualSyscap'][dict(seis= 1)])/ dat.MiniG_TechLifetime[2] #! !  Genset Replacement Cost per item($)
    else:
        GRC = 0

    #!  MiniG. RecurringAnnCost = O&M Cost ($)	+ Genset Replacement Cost per item($) + Diesel Fuel Cost ($)
    a.MiniG_RecurringAnnCost[dict(seis= 1)] = (0.05*dat.DG_CostPerSystemkw.sel(dos=1)*a['MiniG_ActualSyscap'][dict(seis= 1)]).transpose('y', 'x', 'year') + GRC +  a['DG_FuelCOST']   
    a['LCOE_MiniG_Total'][dict(seis= 1)] = xr.where((gph.Elecuseg>1e-6),  (a['MiniG_AnnualizedCapEx'][dict(seis= 1)]  + a.MiniG_RecurringAnnCost[dict(seis= 1)])/gph.Elecuseg +OversizeFac*a.TDCapCostIntPerKWh, 1000)    #!  LCOE DG ($/kwh) If electricity use is zero, the LCOE is very high (1000$/kwh) such as to make it unatractive
    print( 'ec: MG-diesel cost done')
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    # %%
    ##!************************ WIND Mini-grid ************************
    a['MiniG_ActualSyscap'][dict(seis= 2)] =xr.where((gph.Elecuseg.year >= dat.baseyear) & (loadfactor.wind_Loadfac > 1e-6), gph.Elecuseg / (8760*loadfactor.wind_Loadfac*(1-dat.MiniG_distributionloss)), 0).transpose('y', 'x', 'year')   # (kW)

    #! $/year	-Annualized initial investment cost 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG. When using dict starts on zero
    annuity_factor_wind = xr.where(dat.interest > 1e-6, dat.interest /(1-(1+ dat.interest)**-dat.MiniG_TechLifetime[1]), 1)
    for r in nrc: #($/year/grid)
        a['MiniG_AnnualizedCapEx'][dict(seis= 2)] = xr.where(a['MiniG_ActualSyscap'].imager == r, annuity_factor_wind.sel(region=r)*(0.9*dat.cap_cost_new.sel(ntc2=4, region=r)*a['MiniG_ActualSyscap'][dict(seis= 2)] + #! The cost of wind generator and components ($)  
                (dat.BatteryKwhCost[0]*dat.MiniG_StoragekwhperSyskw[2]*a['MiniG_ActualSyscap'][dict(seis= 2)]) +   #! 
                dat.elec_constrcost_new.sel(ntc2=4, region = r)*a['MiniG_ActualSyscap'][dict(seis= 2)]), a['MiniG_AnnualizedCapEx'][dict(seis= 2)])                        #! Interest during construction ($)

    if dat.MiniG_TechLifetime[1] < dat.td_life and dat.MiniG_TechLifetime[1]>1e-6:
        PRC = 0.9/dat.MiniG_TechLifetime[1] #! windmill Replacement Cost multiplier($)
    else:
        PRC = 0
    if dat.BatteryLifetime < dat.td_life and dat.BatteryLifetime>1e-6:
        batteryRC = dat.BatteryKwhCost[0]*dat.MiniG_StoragekwhperSyskw[2]/dat.BatteryLifetime #! Battery Replacement Cost per item($)
    else:
        batteryRC = 0

    for r in nrc:
        a['MiniG_RecurringAnnCost'][dict(seis= 2)]  = xr.where(a['MiniG_ActualSyscap'].imager == r, PRC*a['MiniG_ActualSyscap'][dict(seis= 2)]*dat.cap_cost_new.sel(ntc2=4, region=r)+
                                                BRC*a['MiniG_ActualSyscap'][dict(seis= 2)]*dat.cap_cost_new.sel(ntc2=4, region=r) + 
                                                batteryRC*a['MiniG_ActualSyscap'][dict(seis= 2)] + 
                                                dat.om_cost_perkw_new.sel(ntc2 = 4, region = r)*a['MiniG_ActualSyscap'][dict(seis= 2)] , a['MiniG_RecurringAnnCost'][dict(seis= 2)])#! ! Annual O&M cost ($)

    #cap with techpot
    #! $/kwh		-The LCOE of electricity generation with minigrid 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG
    a['LCOE_MiniG_Total'][dict(seis= 2)] = xr.where((gph.Elecuseg > 1e-6) & (gph.Elecuseg <= potential.wind_techpot ), (a['MiniG_AnnualizedCapEx'][dict(seis= 2)]+ a['MiniG_RecurringAnnCost'][dict(seis= 2)])/gph.Elecuseg +OversizeFac*a.TDCapCostIntPerKWh, 1000)     #	! $/kwh		-Total cost of mini grid including LV network ( 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG


    print( 'ec: MG-wind cost done')
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    # %%

    # ##!********************** MINI-HYDRO ****************************

    a['LCOE_MiniG_Total'][dict(seis= 3)] = xr.where((gph.Elecuseg > 1e-6) & (gph.Elecuseg <= potential.hydro_techpot ),  potential.Hydro_COE+OversizeFac*a.TDCapCostIntPerKWh, 1000)     #	! $/kwh		-Total cost of mini grid including LV network ( 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG
    
    ##!******************** PV-Diesel HYBRID TECHNOLOGY *********************
    RE_HybShare	= {1: 0.7} # 70% for PV
    DG_HybShare = {1: 0.3} # 30% for Diesel
   

    a['PVHybSystemCapacity'] =xr.where((loadfactor.PV_Loadfac > 1e-6) & (gph.Elecuseg.year >= dat.baseyear), RE_HybShare[1]*gph.Elecuseg/(8760*loadfactor.PV_Loadfac*(1-dat.MiniG_distributionloss)), 0)   # (kW)  
    a['DGHybSystemCapacity'] = xr.where((loadfactor.PV_Loadfac > 1e-6) & (gph.Elecuseg.year >= dat.baseyear), DG_HybShare[1]*a['loaddg']/(DG_LoadFactor*(1-dat.MiniG_distributionloss)), 0)   # (kW)   
    a['DGHyb_FuelCOST']  = a['DG_FuelCOST'].copy()
    for r in nrc:
        a['MiniG_AnnualizedCapEx'][dict(seis= 4)] = xr.where((a.imager == r), en.annuity_factor.sel(region=r)*(dat.cap_cost_new.sel(ntc2=1, region=r)*a.PVHybSystemCapacity +  
                (1.25*dat.DG_CostPerSystemkw.sel(dos = 1)*a.DGHybSystemCapacity) +   #! ! 1.25 is to include installation cost for Diesel System			
                dat.elec_constrcost_new.sel(ntc2=1, region = r)*a['PVHybSystemCapacity']), a['MiniG_AnnualizedCapEx'][dict(seis= 4)] )  #! Interest during construction cost
        
    a['DGHyb_FuelCOST'] =  a['DieselPrice']*a['DGHybSystemCapacity']*365*dat.load_hours_daily[0]*dat.DG_FuelConsumpperkwh[0]  

    if dat.MiniG_TechLifetime[0] < dat.td_life and dat.MiniG_TechLifetime[0]>1e-6:
        PRC = (1-dat.PVBOSCost)/ dat.MiniG_TechLifetime[0] #! Panel Replacement Cost per item($)   
    else:
         PRC = 0
    if dat.PVBOSLifetime < dat.td_life and dat.PVBOSLifetime > 1e-6:
        BRC = dat.PVBOSCost/ dat.PVBOSLifetime          #! Balance replacement Cost per item ($)  
    else:
        BRC = 0
    if dat.MiniG_TechLifetime[2] < dat.td_life and dat.MiniG_TechLifetime[2]>1e-6:
        gen = dat.DG_CostPerSystemkw.sel(dos = 1)/dat.MiniG_TechLifetime[2] #! Battery Replacement Cost per item($)
    else:
        gen = 0
    for r in nrc:
        a['MiniG_RecurringAnnCost'][dict(seis= 4)] = xr.where(a['MiniG_ActualSyscap'].imager == r, PRC*a['PVHybSystemCapacity']*dat.cap_cost_new.sel(ntc2=1, region=r)+ #! Panel Replacement Cost ($)
                                                BRC*a['PVHybSystemCapacity']*dat.cap_cost_new.sel(ntc2=1, region=r) + #! Balance replacement Cost ($)
                                                dat.om_cost_perkw_new.sel(ntc2 = 1, region = r)*a['PVHybSystemCapacity'] +    #! ! Annual O&M cost PV ($)
                                                0.05*dat.DG_CostPerSystemkw.sel(dos = 1)*a.DGHybSystemCapacity  +  #Annual O&M cost Diesel ($)
                                                gen*a.DGHybSystemCapacity + a.DGHyb_FuelCOST, a['MiniG_RecurringAnnCost'][dict(seis= 4)])    #! Genset Replacement Cost ($/year) + ! Fuel Cost for DG ($/year)
                        
    a['LCOE_MiniG_Total'][dict(seis= 4)] = xr.where((gph.Elecuseg > 1e-6) &  (RE_HybShare[1]*gph.Elecuseg <= potential.PV_techpot ),  ((a['MiniG_AnnualizedCapEx'][dict(seis= 4)]+a['MiniG_RecurringAnnCost'][dict(seis= 4)]) / gph.Elecuseg) + OversizeFac*a.TDCapCostIntPerKWh, 1000)     #	! $/kwh		-Total cost of mini grid including LV network ( 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG

    print( 'ec: MG PV-diesel done')
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    # %%
    ##!******************* Wind-Diesel HYBRID TECHNOLOGY ******************

    ###! The wind power system has a medium penetration of 50% maximum
    ###! When the wind capacity is below the maximum, the demand gap will be cvered by diesel generator

    RE_HybShare[2]			=	0.5
    DG_HybShare[2]			=	0.5
    
    a['WindHybSystemCapacity'] =xr.where((loadfactor.wind_Loadfac > 1e-6) & (gph.Elecuseg.year >= dat.baseyear), RE_HybShare[2]*gph.Elecuseg/(8760*loadfactor.wind_Loadfac*(1-dat.MiniG_distributionloss)), 0)   # (kW)  
    a['DGHybSystemCapacity2'] = xr.where((loadfactor.wind_Loadfac > 1e-6) & (gph.Elecuseg.year >= dat.baseyear), DG_HybShare[2]*a['loaddg']/(DG_LoadFactor*(1-dat.MiniG_distributionloss)), 0)   # (kW)  
    a['DGHyb_FuelCOST2']  = a['DG_FuelCOST']
    for r in nrc:
        a['MiniG_AnnualizedCapEx'][dict(seis= 5)] = xr.where((a.imager == r), annuity_factor_wind.sel(region=r)*(0.9*dat.cap_cost_new.sel(ntc2=4, region=r)*a.WindHybSystemCapacity +  
                (1.25*dat.DG_CostPerSystemkw.sel(dos = 1)*a.DGHybSystemCapacity2) +   #! ! 1.25 is to include installation cost for Diesel System			
                dat.elec_constrcost_new.sel(ntc2=4, region = r)*a['WindHybSystemCapacity']), a['MiniG_AnnualizedCapEx'][dict(seis= 5)] )  #! Interest during construction cost
        
    a['DGHyb_FuelCOST2'] =  a['DieselPrice']*a['DGHybSystemCapacity2']*365*dat.load_hours_daily[0]*dat.DG_FuelConsumpperkwh[0] 

    if dat.MiniG_TechLifetime[1] < dat.td_life and dat.MiniG_TechLifetime[1]>1e-6: #! Panel Replacement Cost per item($)   
        PRC = 0.9/dat.MiniG_TechLifetime[1] #! windmill Replacement Cost multiplier($)
    else:
        PRC = 0
    if dat.MiniG_TechLifetime[2] < dat.td_life and dat.MiniG_TechLifetime[2]>1e-6:
        gen = dat.DG_CostPerSystemkw.sel(dos = 1)/dat.MiniG_TechLifetime[2] #!! Genset Replacement Cost ($/year)
    else:
        gen = 0

    for r in nrc:
        a['MiniG_RecurringAnnCost'][dict(seis= 5)] = xr.where(a['MiniG_ActualSyscap'].imager == r, PRC*a['WindHybSystemCapacity']*dat.cap_cost_new.sel(ntc2=4, region=r)+ #! Panel Replacement Cost ($)
                                                dat.om_cost_perkw_new.sel(ntc2 = 4, region = r)*a['WindHybSystemCapacity'] +    #! ! Annual O&M cost PV ($)
                                                0.05*dat.DG_CostPerSystemkw.sel(dos = 1)*a.DGHybSystemCapacity2  +  #Annual O&M cost Diesel ($)
                                                gen*a.DGHybSystemCapacity2 + a.DGHyb_FuelCOST2, a['MiniG_RecurringAnnCost'][dict(seis= 5)])    #!  Genset Replacement Cost ($/year) +  Fuel Cost for DG ($/year)
                            
    a['LCOE_MiniG_Total'][dict(seis= 5)] = xr.where((gph.Elecuseg > 1e-6) & (RE_HybShare[2]*gph.Elecuseg <= potential.wind_techpot ), ((a['MiniG_AnnualizedCapEx'][dict(seis= 5)]+a['MiniG_RecurringAnnCost'][dict(seis= 5)])/ gph.Elecuseg) +OversizeFac*a.TDCapCostIntPerKWh, 1000)     #	! $/kwh		-Total cost of mini grid including LV network ( 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG

    print( 'ec: LCOE mini grid done')
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    a = a.drop(['MiniG_RecurringAnnCost', 'DGHyb_FuelCOST2', 'DGHyb_FuelCOST']) #', 'PVHybSystemCapacity', 'WindHybSystemCapacity',  'MiniG_ActualSyscap', 'DGHybSystemCapacity', 'DGHybSystemCapacity2'
   
    # %%
    # !******************** PV STANDALONE ***********************

    MG2SA	=	1.2			#! Multiplication factor from mini-gridSAl_SystemCapacity0 to standalone price


    elec_use_perhh = dat.elec_use_perhh.sel(year = a.year.data, turq = 1)
    a['SAl_SystemCapacity0'] = xr.full_like(gph.Elecuseg, np.nan)
    for r in nrc:
            a['SAl_SystemCapacity0'] = xr.where((loadfactor.imager == r) ,elec_use_perhh.sel(region = r), a['SAl_SystemCapacity0'])   # (kWh-use per hh) 


    a['SAl_SystemCapacity'] = a['SAl_SystemCapacity0'].expand_dims({'dos':['PV', 'diesel']}).copy().transpose('y', 'x', 'year', 'dos')  #		
    #a['SAl_SystemCapacity'][dict(dos= 0)] = a['SAl_SystemCapacity0'].where(a['SAl_SystemCapacity0'] < potential.PV_techpot)  # electricity use per household has to be lower than the PV technical potential 	

    a['SAl_SystemCapacity'][dict(dos= 0)] = xr.where((a['SAl_SystemCapacity0'] < potential.PV_techpot) & (loadfactor.PV_Loadfac > 1e-6) & (gph.hhnumberg.hhnumberg.sel(turq = 1) > 1e-6), a['SAl_SystemCapacity'][dict(dos= 0)]/(8760*loadfactor.PV_Loadfac), 0 ). transpose('y', 'x', 'year')   # (kW) capcity per TURQ total

    a['StAl_AnnuCapEx'] =  a['SAl_SystemCapacity'].copy()
    for r in nrc:
        a['StAl_AnnuCapEx'][dict(dos= 0)] = xr.where(a.imager == r, annuity_factor_solar.sel(region=r)*(MG2SA*dat.cap_cost_new.sel(ntc2=2, region=r)*a['SAl_SystemCapacity'][dict(dos= 0)]  + # ! The cost of panels and BOS ($/year)
                (dat.BatteryKwhCost[1]*dat.StAl_StoragekwhperSyskw[0]*a['SAl_SystemCapacity'][dict(dos= 0)])), a['StAl_AnnuCapEx'][dict(dos= 0)] )  

    if dat.StAl_TechLifetime[0] < dat.td_life and dat.StAl_TechLifetime[0]>1e-6:
        PRC = (1-dat.PVBOSCost)*MG2SA/ dat.StAl_TechLifetime[0] #! Panel Replacement Cost per item($)  
    else:
        PRC = 0
    if dat.PVBOSLifetime < dat.td_life and dat.PVBOSLifetime > 1e-6:
        BRC = dat.PVBOSCost*MG2SA/ dat.PVBOSLifetime          #! Balance replacement Cost per item ($)   
    else:
        BRC = 0
    if dat.BatteryLifetime < dat.td_life and dat.BatteryLifetime>1e-6:
        batteryRC = dat.BatteryKwhCost[1]*dat.StAl_StoragekwhperSyskw[0]/dat.BatteryLifetime #! Battery Replacement Cost per item($)
    else:
        batteryRC = 0

    a['StAl_RecuAnnuCOST'] = a['SAl_SystemCapacity'].copy()     #! $/year	-Fixed annual cost 1= Solar, 2=Diesel, 3=Wind, 4=Hydro, 5=Hybrid PV-DG, 6=Hybrid Wind-DG
    for r in nrc:
        a['StAl_RecuAnnuCOST'][dict(dos= 0)] = xr.where(a['SAl_SystemCapacity'][dict(dos= 0)].imager == r, PRC*a['SAl_SystemCapacity'][dict(dos= 0)]*dat.cap_cost_new.sel(ntc2=2, region=r)+ #! Panel Replacement Cost ($)
                                                BRC*a['SAl_SystemCapacity'][dict(dos= 0)]*dat.cap_cost_new.sel(ntc2=2, region=r) + #! Balance replacement Cost ($)
                                                batteryRC*a['SAl_SystemCapacity'][dict(dos= 0)] + #! Battery Replacement Cost ($)
                                                MG2SA*dat.om_cost_perkw_new.sel(ntc2 = 2, region = r)*a['SAl_SystemCapacity'][dict(dos= 0)], a['StAl_RecuAnnuCOST'][dict(dos= 0)])#! ! Annual O&M cost ($)

    ##!************************* STANDALONE DIESEL GENSET ***********************
    DGStAl_LoadFactor		=	0.5 
    a['SAl_SystemCapacity'][dict(dos= 1)] = xr.where((gph.hhnumberg.hhnumberg.sel(turq = 1) > 1e-6), a['SAl_SystemCapacity0']*1.7/(DG_HoursOperation*DGStAl_LoadFactor), 0 ). transpose('y', 'x', 'year')  # (kW) capcity per TURQ total ---- The 1.2 is the peak versus middle load ratio from my paper

    a['StAl_DG_FuelCOST']= xr.full_like(gph.Elecuseg, np.nan) #gph.hhnumberg.hhnumberg.sel(turq = slice(2,3))
    for r in nrc:
        a['StAl_AnnuCapEx'][dict(dos= 1)] = xr.where(a['SAl_SystemCapacity'][dict(dos= 1)].imager == r, en.annuity_factor.sel(region=r)* MG2SA *(1.25*a['SAl_SystemCapacity'][dict(dos= 1)]*dat.DG_CostPerSystemkw.sel(dos=1)), a['StAl_AnnuCapEx'][dict(dos= 1)])   
    a['StAl_DG_FuelCOST'] =  a['DieselPrice']*a['SAl_SystemCapacity'][dict(dos= 1)]*DG_HoursOperation*dat.DG_FuelConsumpperkwh[1]

    if dat.StAl_TechLifetime[1] < dat.td_life  and dat.StAl_TechLifetime[1]>1e-6:
        GRC = (dat.DG_CostPerSystemkw.sel(dos=1)*a['SAl_SystemCapacity'][dict(dos= 1)])/ dat.StAl_TechLifetime[1] #! !  Genset Replacement Cost ($)
    else:
        GRC = 0

    a.StAl_RecuAnnuCOST[dict(dos= 1)] = ((0.05 * MG2SA*dat.DG_CostPerSystemkw.sel(dos=2)*a['SAl_SystemCapacity'][dict(dos= 1)])+ MG2SA*GRC +  a['StAl_DG_FuelCOST'] ).transpose('y', 'x', 'year')  # = O&M Cost ($)	+ Genset Replacement Cost per item($) + Diesel Fuel Cost ($)

    a['LCOE_StAl'] = a.StAl_RecuAnnuCOST
    for r in nrc:
        a['LCOE_StAl'] = xr.where(a.imager == r, (a['StAl_AnnuCapEx']+ a['StAl_RecuAnnuCOST'])/ elec_use_perhh.sel(region = r), a['LCOE_StAl'] )    #! $/kwh/per grid cell amd quantile	

    a['LCOE_StAl'][dict(dos= 0)] = xr.where((gph.hhnumberg.hhnumberg.sel(turq = 1) > 1e-6) & (gph.Elecuseg <= potential.PV_techpot ), a['LCOE_StAl'][dict(dos= 0)], 1000 ).transpose('y', 'x', 'year')    #Filtering locations for PV depending on potential
    a['LCOE_StAl'][dict(dos= 1)] = xr.where((gph.hhnumberg.hhnumberg.sel(turq = 1) > 1e-6), a['LCOE_StAl'][dict(dos= 1)], 1000 ).transpose('y', 'x', 'year')
    
   
    a = a[['LCOE_StAl', 'LCOE_MiniG_Total', 'TotalCostperkWh_Grid', 'TotCostperkwh_Internal', 'SAl_SystemCapacity', 'PVHybSystemCapacity', 'WindHybSystemCapacity',  'MiniG_ActualSyscap', 'DGHybSystemCapacity', 'DGHybSystemCapacity2']].drop([ 'tres', 'ntc2', 'band', 'NS', 'NEC']) #'RecurringAnnCost',  'TDCapCostIntPerKWh', 'MiniG_AnnualizedCapEx', 'MiniG_RecurringAnnCost', "StAl_AnnuCapEx", 'StAl_RecuAnnuCOST',
    return a
a = fun()
print('elec_cost done')
print("--- %s minutes ---" % ((time.time() - start_time)/60))

a.to_netcdf('..\\output\\ec_a_world.nc', mode='w')

print('elec_cost saved')
