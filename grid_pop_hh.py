# -*- coding: utf-8 -*-
"""
This code analyses population grid data and calculates electricity use per grid-cell.
The electricity use per grid-cell is calculated using rural and urban data on:
population maps, regional average household size and electricity use per household imported from IMAGE-TIMER
"""

import numpy as np
import os
import xarray as xr 
import glob
import time
start_time = time.time()
print(start_time)


#********************************************************************
#***************************** Grid data *****************
#********************************************************************

def fun_pop():
     os.getcwd()
     DIR = "X:\\user\\zapatacasv\\sdg7\\scripts"   # Set the running directory here
     os.chdir(DIR)
     import data_import as dat #import own code after defining home dir

     #####loading gridded pop data
     ssp = dat.ssp# 
     df = glob.glob('..\\data\\pop_highres\\' + ssp + "\\" + "*5m.nc")
     popg = xr.combine_by_coords([xr.open_dataset(i) for i in df], combine_attrs = "override")
     popg = popg.fillna(value=0)  

     #loading image region mask
     df= xr.open_dataset('..\\data\\GREG.nc')
     popg = popg.assign_coords(x = ("x", df.x.values))
     popg = popg.assign_coords(y = ("y", df.y.values)) 
     popg.coords['imager'] = (('y', 'x'), df.imager.values)
      #remove regions with 100% electricity acces based on World bank data for rural rates
     popg['imager'] = popg.imager.where((popg.imager != 1) | (popg.imager != 2) | (popg.imager != 5) |(popg.imager != 7) |(popg.imager != 11) | (popg.imager != 12) | (popg.imager != 13) | (popg.imager != 14) | (popg.imager != 15) |(popg.imager != 16) |(popg.imager != 23) , drop = True )    
     popg['year'] = popg['year'].astype(int)
     df= xr.open_dataset('..\\data\\garea_cr_km2_5min.nc')   
     popg['area'] = (('y', 'x'), df.area.values)
     popg = popg.where(popg.imager>0)   # setting oceans to nan

     #interpolate over time
     if dat.dt!=10:
          popg = popg.interp(year= range(dat.t0, dat.tf+1, dat.dt), method="linear")
     else:
          popg = popg.sel(year = slice(dat.t0, dat.tf))
     regional_pop = popg.groupby("imager").sum()
     regional_pop = regional_pop.transpose('year', 'imager')
     regional_pop[dict(imager = 26)] = regional_pop.sum(dim= 'imager')

     #####Correcting gridded pop data with TIMER pop data (only for total)
     regional_pop['tpop_diff']= (['year', "imager"], dat.pop.values*1e6/regional_pop["tpop"].values)  #correct with total to keep ratios urban versus rural from gridded data
     diff = regional_pop.sel(year = popg.year)
     diff = diff['tpop_diff']
     del regional_pop['tpop_diff']
     nrc = dat.nrc
     for i  in nrc:
          popg['rpop'] = xr.where(popg.imager == i, popg['rpop']*diff.sel(imager = i), popg['rpop'] )
          popg['upop'] = xr.where(popg.imager == i, popg['upop']*diff.sel(imager = i), popg['upop'] )
     popg['tpop'] = popg['rpop'] + popg['upop']
     
     #This file was obtained by calculating the share of grid cells that had more than 2 people per cell when upscaling population maps from 30'arcsec to 5min resolution
     inhab_area = xr.open_dataarray('..\\data\\areapop_threshold2_5minutes.nc') 
     popg['inhab_area'] = inhab_area    
     
     #regional pop calibrated with timer pop
     regional_pop = popg.groupby("imager").sum()
     regional_pop['rpop_density'] = regional_pop['rpop']/(regional_pop['area']*dat.disagg_factor) #area in km2
     regional_pop['upop_rate'] = regional_pop['upop']/regional_pop['tpop'] 
     regional_pop = regional_pop.rename({'imager':'region'})

     ###for test in SSA
     #popg = popg.where((popg.imager == 8) | (popg.imager == 9) | (popg.imager == 10) | (popg.imager == 26), drop = True)
     
     ###Calculating household numbers per grid cell using hh size from TIMER 
     #turq 1=total, 2 = urban, 3 = rural, 4-8: quantiles urban, 9-13: quantiles rural
     hhsize= dat.hhsize.sel(year = popg.year, turq = slice(1,3))
     hhsize['year'] = hhsize['year'].astype(int)
     popg['popg_tur'] = popg.tpop.expand_dims({'turq': np.arange(1,4)}).copy().transpose('y', 'x', 'year', 'turq')
     popg['popg_tur'][dict(turq= 0)] = popg.tpop
     popg['popg_tur'][dict(turq= 1)] = popg.upop
     popg['popg_tur'][dict(turq= 2)] = popg.rpop

     # #When using quantiles. However all quantiles were assigned the same weight
     # popgturq = popg['tpop'].expand_dims({'turq':np.arange(1,4)}).copy()
     # popgturq[dict(turq= 0)] = popg['tpop']
     # popgturq[dict(turq= 1)] = popg['upop']
     # popgturq[dict(turq= 2)] = popg['rpop']
     # for j in np.arange(3,8):
     #      popgturq[dict(turq= j)] = popg['upop']/4
     # for j in np.arange(8,13):
     #      popgturq[dict(turq= j)] = popg['rpop']/4

     hhnumberg =xr.full_like(popg['popg_tur'], np.nan)  
     for i  in nrc:
          hhnumberg= xr.where(hhnumberg.imager == i,popg['popg_tur']/hhsize.sel(region = i, turq = slice(1,3)), hhnumberg ) 
     hhnumberg = hhnumberg.to_dataset(name = 'hhnumberg')
     hhnumberg[dict(turq= 0)] = hhnumberg.sel(turq = slice(2,3)).sum(dim = 'turq')  
     hhnumberg = hhnumberg.drop('region')
     
     ##regional hhnumber
     hhnumber = hhnumberg.groupby("imager").sum()
     hhnumber = hhnumber.rename({'imager':'region'})
     popg =  popg.drop_vars(['rpop', 'upop'])  #For reducing memory

     # !******************** Electricity use per grid cell ****************************
     ElecConsum_PerHH = dat.elec_use_perhh.sel(year = hhnumberg.year.data,  turq = slice(1,3)) #[kWh]
     Elecuseg = xr.full_like(hhnumberg.hhnumberg, np.nan)
     for r in nrc:
          Elecuseg = xr.where((Elecuseg.imager == r) , hhnumberg.hhnumberg*ElecConsum_PerHH.sel(region = r), Elecuseg)  
     Elecuseg = Elecuseg.sel(turq = slice(2,3)).sum(dim = 'turq')

     return popg, regional_pop, hhnumberg, hhnumber, Elecuseg

popg, regional_pop, hhnumberg, hhnumber, Elecuseg = fun_pop()


print('gph done')
print("--- %s minutes for gph ---" % ((time.time() - start_time)/60))

popg.to_netcdf('..\\output\\gph_popg_world_SSP2.nc', mode='w')
regional_pop.to_netcdf('..\\output\\gph_regional_pop_world_SSP2.nc', mode='w')
hhnumberg.to_netcdf('..\\output\\gph_hhnumberg_world_SSP2.nc', mode='w')
hhnumber.to_netcdf('..\\output\\gph_hhnumber_world_SSP2.nc', mode='w')
Elecuseg.to_netcdf('..\\output\\gph_Elecuseg_world_SSP2.nc', mode='w')
print('gph saved')
print("--- %s minutes for gph ---" % ((time.time() - start_time)/60))
