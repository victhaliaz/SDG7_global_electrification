# %%
# %%
# -*- coding: utf-8 -*-
"""
This code executes the selection of systems for electrification based mainly on cost or distance to grid. 
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
import elec_cost as ec
dat = ec.dat #import own codes after defining home dir
gph =ec.gph
en = ec.en
# #import elec_rate as er
import time
start_time = time.time()



# #When running in parts
# import data_import as dat
# import test as en
# en.kmlines = xr.open_dataset('..\\output\\SSP2\\en_kmlines_world.nc').sel(year = slice(dat.t0, dat.tf))
# en.capcost = xr.open_dataset('..\\output\\SSP2\en_capcost_world.nc').sel(year = slice(dat.t0, dat.tf))
# import test as gph
# gph.Elecuseg = xr.open_dataarray('..\\output\\gph_SSP2\\gph_Elecuseg_world_SSP2.nc').sel(year = slice(dat.t0, dat.tf))
# gph.popg = xr.open_dataset('..\\output\\gph_SSP2\\gph_popg_world_SSP2.nc').sel(year = slice(dat.t0, dat.tf))
# gph.regional_pop = xr.open_dataset('..\\output\\gph_SSP2\\gph_regionalpop_world_SSP2.nc').sel(year = slice(dat.t0, dat.tf))
# gph.hhnumberg = xr.open_dataset('..\\output\\gph_SSP2\\gph_hhnumberg_world_SSP2.nc').sel (year = np.arange(dat.t0, dat.tf+1))
# import test as ec
# ec.a = xr.open_dataset('..\\output\\SSP2\\ec_a_world.nc').sel(year = slice(dat.t0, dat.tf))
import test as er
er.elec_rate = xr.open_dataset('..\\output\\er_SSP2_world_annual.nc').sel(year = slice(dat.t0, dat.tf))



def fun_es():

    #For reference see Dagnachew 2017, https://doi.org/10.1016/j.energy.2017.07.144.

    # %%
    # !*******************************************************
    # !************* Grid cell with access to the network*****
    # !******************************************************

    # #! The aim is to sort gridcells ascending based on the distance to the power line. Then, calculate the cumulativesum of the total population per grid cell following order of the sort. 
    #distace to grid
    path2 =     "..\\data\\OSM_distance2grid_calculated_on_ESRI54009_popweighted_5minutes.nc"   # 
    s = xr.open_dataset(path2) 

    s = s.where((s.x == gph.popg.x) & (s.y == gph.popg.y), drop = True)   #used when there is a filter for running only one region  
    s['popg_cumsum'] = gph.popg['tpop'].sel(year = dat.baseyear).copy()
    s['imager'] = gph.popg['imager']

    nrc = dat.nrc  #number of regions
    for i in nrc:
        x = s.where(s.imager == i)
        x = x.stack(z = ('y', 'x'))
        x =  x.sortby(x.d2g_wupscale )
        x['popg_cumsum'] = x.popg_cumsum.cumsum(dim = 'z', skipna=True)
        x = x.unstack()
        s['popg_cumsum'] = xr.where((s.imager == i),  x['popg_cumsum'],  s['popg_cumsum'])  
    s['popg_cumsum'] =  s.popg_cumsum.where(gph.popg.tpop.sel(year = dat.baseyear)>0) 

    # %%
    s['pop_connected_BL'] = er.elec_rate['total'].expand_dims({'turq': np.arange(1,4)}).copy()  #turq1=total, 2 = urban, 3 = rural
    s['pop_connected_BL'] = s['pop_connected_BL'].transpose('region', 'year', 'turq')
    s['pop_connected_BL'][dict(turq= 1)] = er.elec_rate['urban'] * gph.regional_pop.upop
    s['pop_connected_BL'][dict(turq= 2)] = er.elec_rate['rural'] * gph.regional_pop.rpop
    s['pop_connected_BL'][dict(turq= 0)] = s['pop_connected_BL'][dict(turq= 1)] + s['pop_connected_BL'][dict(turq= 2)]

    s['pop_connected_univ'] = xr.full_like(s['pop_connected_BL'], np.nan)                        
    s['pop_connected_univ'][dict(turq= 1)] = er.elec_rate['urban_target'] * gph.regional_pop.upop
    s['pop_connected_univ'][dict(turq= 2)] = er.elec_rate['rural_target'] * gph.regional_pop.rpop
    s['pop_connected_univ'][dict(turq= 0)] = s['pop_connected_univ'][dict(turq= 1)] + s['pop_connected_univ'][dict(turq= 2)]
    s['pop_connected_BL'] = xr.where(s['pop_connected_BL'] > s['pop_connected_univ'], s['pop_connected_univ'], s['pop_connected_BL']) 

    # %%
    #! Grid-cells that are the closest to the power line are assumed to be already electrified at the model-projected electrification level 
    s['GridElecCellBaseYear'] = xr.zeros_like(s.imager)
    for i in nrc:
        s['GridElecCellBaseYear'] = xr.where((s.imager == i) , s.pop_connected_BL.sel(region = i, turq= 1, year =  dat.baseyear), s['GridElecCellBaseYear']  ) 
    s['GridElecCellBaseYear'] = xr.where((s.popg_cumsum <= s['GridElecCellBaseYear']),  1, np.nan ) 
    s = s.drop('popg_cumsum')
    print("s['GridElecCellBaseYear'] done")
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    # %%
    #! The EDL is the length of the HV and MV lines that can be covered by the difference between the cost of the cheapest off-grid technology cost & the internal cost of grid expansion
    #EDL: is the economic distance limit 
    cost_expansion = (en.kmlines.HV*dat.ElecCapCost.sel(type =1) + en.kmlines.MV*dat.ElecCapCost.sel(type =2)).sel(year = slice(dat.baseyear , dat.baseyear+dat.td_life)).mean(dim = 'year') #[$*km]   
    s['Elecuseg'] = gph.Elecuseg  
    s['EDL'] = (ec.a.LCOE_MiniG_Total.min(dim = 'seis')-ec.a.TotCostperkwh_Internal).sel(year = slice(dat.baseyear , dat.baseyear+dat.td_life)).mean(dim = 'year')* s.Elecuseg.sel(year = slice(dat.baseyear , dat.baseyear+dat.td_life)).sum(dim = 'year') / (cost_expansion)  #
    s['EDL'] = xr.where( (cost_expansion > 1e-6), s['EDL']*1000, np.nan)   #[m]
    s['EDL'] = s.EDL.where(s.EDL > 1e-6)

    s['GridPopDensity']= gph.popg.tpop/gph.popg['inhab_area']
    #! The minimum threshold density for minigrid based on electric use per km2 of 37500kWh/km2. IE. 250person/km2 consuming 150kwh/person/year
    s['ThresholdDensity'] = xr.where(s.Elecuseg > 1e-6, (250*150* gph.popg.tpop)/s.Elecuseg, 0)    #[pop/km2]

    # %%
    #!  Find the lowest cost of electricity of either grid extension, minigrid or standalone and save it in s['LeastCostElecCOST']  and LeastCostTechID
    s['central_grid_cost_inplace'] = xr.full_like(s['GridPopDensity'], np.nan) 

    EL = s.d2g_wupscale < s.EDL  

    #Cost for central grid where there is already connection and where lower than EDL
    for r in nrc: 
        s['central_grid_cost_inplace'] = xr.where((s.imager == r) & (s['GridElecCellBaseYear']== 1), dat.system_total_cost_perkwh.sel(tres = 3, region = r, year = s.year), s['central_grid_cost_inplace'] )  # Where there is already grid

    s['central_grid_cost'] =xr.where(  (s['d2g_wupscale'] < 50e3) & (s['GridElecCellBaseYear']!= 1), ec.a.TotCostperkwh_Internal , s['central_grid_cost_inplace'] )  #for  distance lower than 50 km to ensure that private investment on mini grid or stand alone get their revenue before the central grid reaches them
    s['central_grid_cost'] =xr.where(  (s['d2g_wupscale'] > 50e3) & (s['GridElecCellBaseYear']!= 1) & (EL), ec.a.TotalCostperkWh_Grid , s['central_grid_cost'] ) # When distance 2 grid is lower than the EDL

    ##Conditions for stand alone
    sa2 = ec.a['LCOE_MiniG_Total'].min(dim = 'seis') > ec.a['LCOE_StAl'].min(dim = 'dos') #min(LCOE mini grid) > min(LCOE stand alone)
    sa3 = s['GridPopDensity'] < s['ThresholdDensity']  # Stand alone locations should have a density lower than the threshold density (max 250person/km2 consuming 150kwh/person/year)
    s['minig_sa']= xr.full_like(s['GridPopDensity'], np.nan) 
    s['minig_sa'] =xr.where(sa2 &  sa3, ec.a['LCOE_StAl'].min(dim = 'dos'), ec.a['LCOE_MiniG_Total'].min(dim = 'seis')) #Deciding between minigrid or stand alone .sel(turq = 1)
    #there is also the condition of available potencial for solar, wind and hydro, but it's already considered in the elec_cost module
    
    # all locations for CG, MG and SA
    s['LeastCostElecCOST'] = xr.where(s['central_grid_cost'] > 1e-6, s['central_grid_cost'], s['minig_sa'] )   #The least cost elec cost is the LCOE for the different options
    s['LeastCostElecCOST'] = s['LeastCostElecCOST'].where(gph.popg.tpop > 1)

    s = s[[ 'LeastCostElecCOST',  'EDL', 'pop_connected_univ', 'pop_connected_BL', 'central_grid_cost_inplace', 'GridElecCellBaseYear']]

    print("s['LeastCostElecCOST'] done")

    #Leas cost technology selected per grid-cell and per year
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  s['central_grid_cost_inplace'], 1, np.nan)   #! Grid cell already has access to central grid
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a.TotCostperkwh_Internal , 2,s['LeastCostTechID']) #! Distribution network plus generation cost  
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a.TotalCostperkWh_Grid , 3,s['LeastCostTechID'])    #! Transmission and Distribution network plus generation cost  
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a['LCOE_MiniG_Total'].sel(seis ='PV' ), 4,s['LeastCostTechID']) #Mini grid
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a['LCOE_MiniG_Total'].sel(seis = 'diesel'), 5,s['LeastCostTechID'])
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a['LCOE_MiniG_Total'].sel(seis = 'wind'), 6,s['LeastCostTechID'])
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a['LCOE_MiniG_Total'].sel(seis = 'hydro'), 7,s['LeastCostTechID'])
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a['LCOE_MiniG_Total'].sel(seis = 'PV-diesel'), 8,s['LeastCostTechID']) # #Mini grid hybrid
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a['LCOE_MiniG_Total'].sel(seis = 'wind-diesel'), 9,s['LeastCostTechID'])
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a['LCOE_StAl'].sel( dos = 'PV'), 10,s['LeastCostTechID'])     # Standalone turq = 1,
    s['LeastCostTechID'] = xr.where(s['LeastCostElecCOST'] ==  ec.a['LCOE_StAl'].sel( dos = 'diesel'), 11,s['LeastCostTechID']) #turq = 1,
    print("s['LeastCostTechID'] done")
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    # %%
    #!*****************************************************************************************					
    #!***************************** SCENARIOS calculations ************************************  
    #!*****************************************************************************************
    #The cells are sorted based on cost for electrification
    s['CumGridPopLeastCost'] =  xr.full_like(s['LeastCostElecCOST'], np.nan).transpose('y', 'x', 'year')  #! Cummulative grid population after sorting it out by least cost value, Regional and 1=Total 2=Urban 3=Rural
    ds =  gph.popg.tpop.to_dataset(name = 'tpop')
    ds['LeastCostElecCOST'] = s.LeastCostElecCOST
    for i  in nrc:
        x = ds.where(ds.imager == i)
        x = x.stack(z = ('y', 'x'))
        for t in range(len(x.year)):
                y = x.sel(year = x.year[t]) #, turq= q+1)
                y = y.sortby(y.LeastCostElecCOST)
                y['tpop'] = y.tpop.cumsum(dim = 'z', skipna=True)
                y = y.unstack()
                s['CumGridPopLeastCost'][dict(year = t)] = xr.where(s.imager == i, y.tpop,  s['CumGridPopLeastCost'][dict(year = t)])
    s['CumGridPopLeastCost'] =  s['CumGridPopLeastCost'].where(s.LeastCostElecCOST>=0) 
    print( "s['CumGridPopLeastCost'] done")
    print("--- %s minutes ---" % ((time.time() - start_time)/60))

    #Cell with electricity access per year
    s['ElecAccess_univ'] = xr.zeros_like(s.CumGridPopLeastCost)
    s['ElecAccess_BL'] = xr.zeros_like(s.CumGridPopLeastCost)
    s['ElecAccess_univ0'] = xr.zeros_like(s.CumGridPopLeastCost)
    s['ElecAccess_BL0'] = xr.zeros_like(s.CumGridPopLeastCost)
    for i in nrc:
        s['ElecAccess_univ0'] = xr.where((s.imager == i) , s.pop_connected_univ.sel(region = i, turq = 1), s['ElecAccess_univ0']  ) 
        s['ElecAccess_BL0'] = xr.where((s.imager == i) , s.pop_connected_BL.sel(region = i, turq = 1), s['ElecAccess_BL0']  ) 
    s['ElecAccess_univ'] = xr.where(s.CumGridPopLeastCost <= s['ElecAccess_univ0'],  1, np.nan ) 
    s['ElecAccess_BL'] = xr.where(s.CumGridPopLeastCost <= s['ElecAccess_BL0'],  1, np.nan ) 
    s['ElecAccess_univ'][dict(year= 0)] = s['GridElecCellBaseYear']
    s['ElecAccess_BL'][dict(year= 0)] = s['GridElecCellBaseYear']
    print("s['ElecAccess_BL'] and s['ElecAccess_univ'] done")
    
    #Cells gaining access every year  
    # #2 had access in the previous years, 1 gained access this year
    s['new_access_univ'] = s['ElecAccess_univ'].copy()
    s['new_access_BL'] = s['ElecAccess_BL'].copy()
    year = s.LeastCostElecCOST.year.data
    for y in range(len(year)):
        if y !=0:
            s['new_access_univ'][dict(year= y)] = xr.where(s.ElecAccess_univ.sel(year = year[y-1]) == 1, 2,  s['new_access_univ'][dict(year= y)])  #2 had access in the previous year, 1 gained access this year
            s['new_access_BL'][dict(year= y)] = xr.where(s.ElecAccess_BL.sel(year = year[y-1]) == 1, 2,  s['new_access_BL'][dict(year= y)])  #2 had access in the previous year, 1 gained access this year

    print("s.new_access done")
    print("--- %s minutes ---" % ((time.time() - start_time)/60))
    s = s[['LeastCostTechID', 'LeastCostElecCOST', "new_access_BL", "new_access_univ" ]] #, 'EDL' 
    
    return s

s  = fun_es()
s.to_netcdf('..\\output\\es_s_world.nc', mode='w')
print("--- %s minutes for all and saved ---" % ((time.time() - start_time)/60))


