
# -*- coding: utf-8 -*-
"""
Code for obtaining the electricity network required for each grid-cell and its cost
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
import data_import as dat #import own codes after defining home dir
import grid_pop_hh as gph
import time
start_time = time.time()

# #when running in parts
# import test as gph 
# gph.popg = xr.open_dataset('..\\output\\gph_popg_world_SSP2.nc').sel (year = np.arange(dat.t0, dat.tf+1))
# gph.hhnumberg = xr.open_dataset('..\\output\\gph_hhnumberg_world_SSP2.nc').sel (year = np.arange(dat.t0, dat.tf+1))
# gph.Elecuseg = xr.open_dataarray('..\\output\\gph_Elecuseg_world_SSP2.nc').sel (year = np.arange(dat.t0, dat.tf+1))
# gph.regional_pop = xr.open_dataset('..\\output\\gph_regionalpop_world_SSP2.nc').sel(year = slice(dat.t0, dat.tf))



def fun_en():
  
        ######### Calculating transmission and distribution cost following Van Ruijven et al., 2012

        ########
        #! Capital Costs are function of HV, MV and LV line-length, and number of tranformation equipment, and assumed lifetime of T&D capital.

        a= gph.hhnumberg.sel(turq =1) 
        a['inhab_area'] = gph.popg['inhab_area']
        #! Calculates peak load (in kW)
        elec_use_perhh = dat.elec_use_perhh.sel(year = a.year.data, turq=1)
        hh_peakload = xr.full_like(elec_use_perhh, np.nan)
        for r in dat.nrc:
                for y in range(elec_use_perhh.year.size):
                        if elec_use_perhh.sel(year = elec_use_perhh.year[y], region = r) < 700:
                                hh_peakload[dict(region = r-1, year = y)] = elec_use_perhh.sel(year = elec_use_perhh.year[y], region = r)/(365*dat.load_hours_daily[1]) 
                        else:
                                hh_peakload[dict(region = r-1,  year = y)] = elec_use_perhh.sel(year = elec_use_perhh.year[y], region = r)*2/(365*24)# 
        a['div'] =  xr.full_like(a.hhnumberg, np.nan) 
        nrc = dat.nrc     
        for i  in nrc:
                a['div']  = xr.where(a.imager == i, a.hhnumberg * hh_peakload.sel(region = i), a['div'] )  
        a['SWERlinesg'] = (a['div']/dat.SWERcapacity)   
        a['SWERlinesg'] = np.ceil(a['SWERlinesg']).where(a.hhnumberg > 1e-6) # number of swerlines per grid cell (no decimals with roundup)

        # # #! SWER reach in km  (maximum swer reach is 50km)
        a['SWERreach']  = xr.where((a.SWERlinesg > 1e-6) & (a.inhab_area > 1e-6), np.fmin(50,  a.inhab_area/a.SWERlinesg)/(2*np.sqrt( a.inhab_area)), 50)  

        # # #! Additional HV lines needed to cover a grid cell in number of lines		
        a['extra_lines'] = xr.where((a.inhab_area > 1e-5) & (a.SWERreach > 1e-5), np.fmax((np.sqrt(a.inhab_area)/(2*a.SWERreach))-1, 0), 0)#


        # %%
        # Extra lines required of HV,MV and LV
        kmlines  = a['extra_lines'] 
        kmlines = kmlines.to_dataset(name= 'extra_lines')
        kmlines['HV'] = xr.where(a['extra_lines'] > 1e-6, a['extra_lines']*np.sqrt(a.inhab_area)*np.sqrt(gph.popg.area/2), np.sqrt(gph.popg.area/2))

        #! Medium Voltage line in km
        kmlines['MV'] = a.SWERlinesg * a.SWERreach  
        a = a.drop(['SWERreach'])

        ####Now calculate km lines for low voltage
        maxarea_LVline = (dat.max_LVdist/ np.sqrt(2))**2  #This comes to 450 km2

        a['LVperSWER_linelength'] = xr.where((a.SWERlinesg>1e-6) & (maxarea_LVline>1e-6), (a.inhab_area / a.SWERlinesg) / maxarea_LVline, 0) # The minimal number of LV network per SWER line
        a['LVperSWERCapacity'] = xr.where(a.SWERlinesg > 1e-5, a.div/dat.LV_capac/ a.SWERlinesg, 0)
        a['LVgroups_perSWER']	= 	np.fmin(a.hhnumberg, np.fmax(a.LVperSWER_linelength, a.LVperSWERCapacity))

        #  Calculates the number of households per LV line
        a['HHs_perLVline'] = xr.where((a.LVgroups_perSWER >1e-6) & (a.SWERlinesg > 1e-6), a.hhnumberg / (a.LVgroups_perSWER * a.SWERlinesg), 0)

        #a['unitLVlinelength'] = xr.where(a.hhnumberg>1e-5,  np.sqrt(a.inhab_area/a.hhnumberg) *np.sqrt(2)/2, 0)
        a['LVline_perLVgroup'] = xr.where(a.hhnumberg>1e-5,  a.HHs_perLVline*dat.LVunitfactor*(np.sqrt(a.inhab_area/a.hhnumberg) *np.sqrt(2)/2), 0)    #a.HHs_perLVline  * a.unitLVlinelength  * dat.LVunitfactor 
        a = a.drop('inhab_area')

        kmlines['LV']  = 	a.SWERlinesg * a.LVgroups_perSWER * a.LVline_perLVgroup  #! Low Voltage line in km
        a = a.drop(['LVline_perLVgroup','HHs_perLVline'] )

        # %%
        ####First the electricity use per grid cell is calculated, then calculations of transmission and distribution cost
        annuity_factor = xr.where(dat.interest > 1e-6, dat.interest /(1-(1+ dat.interest)**-dat.td_life), 1)
        
        ##Calculating electricity demand from cooking using TIMER shares and assuming fixed per capita consumption 
        a['Elecuseg'] = gph.Elecuseg

        #! For each voltage-level or line-split-off a transformer is placed. Furthermore, a transformer is also placed per hhnumber for internal cost
        a['Equipment'] = a.SWERlinesg + a.LVgroups_perSWER*a.SWERlinesg + a['extra_lines']

        #! Internal Cost of grid = LV lines and household wiring & metering
        a['TDCapCostInt']  = xr.where((a['Elecuseg']> 1e-6) & (dat.td_life > 1e-6), (kmlines['LV']* dat.ElecCapCost.sel(type = 3)) + (a.hhnumberg * dat.ElecCapCost.sel(type = 4)), 0)  #a.hhnumberg * dat.ElecCapCost is used to calculate the price of transformer placed per hh.

        #! External cost of grid = HV lines, MV lines, Transformers
        a['TDCapCostExt'] = xr.where((a['Elecuseg']> 1e-6) & (dat.td_life > 1e-6), (kmlines['HV']*dat.ElecCapCost.sel(type = 1)) + (kmlines['MV']*dat.ElecCapCost.sel(type = 2))  + (a['Equipment']*dat.ElecCapCost.sel(type = 5)), 0) #HV + MV +transformers

        #! Total electrification cost ($) (capped at TDCapCost_Urban ) #############################################  
        a['TDCapCostg'] = xr.where((a.hhnumberg > 1e-6), np.fmax(dat.TDCapCost_Urban * a.hhnumberg,(a.TDCapCostInt + a.TDCapCostExt)),0 )


        #! T&D Capital costs per HH for connections and transformers  ($/household)
        TDCapCost_perHHg = xr.where((a.hhnumberg > 1), a['TDCapCostg']  / a.hhnumberg, 0) 
                        
        #! Annualized T&D cost
        ann_TDCapCostg = xr.full_like(a['TDCapCostg'], np.nan)
        for i  in nrc:
                ann_TDCapCostg = xr.where((a.TDCapCostg.imager == i), a['TDCapCostg']  * annuity_factor.sel(region = i), ann_TDCapCostg ) 

        #! Calculates the total annualised transmission and distribution costs in USD per kWh for every gridcell
        capcost = a['TDCapCostg'] 
        capcost = capcost.to_dataset(name = 'TDCapCostg')
        capcost = capcost.where(a['Elecuseg']> 1e-6)
        capcost['TDCapCostg_perkwh'] = xr.where((a['Elecuseg']> 1e-6), ann_TDCapCostg / a.Elecuseg, np.nan)   #very large outliers for this one for region 9
        capcost['TDCapCost_perHHg'] = TDCapCost_perHHg
        capcost = capcost.drop(['region', 'type', 'turq'])

        a = a[['TDCapCostInt', 'TDCapCostExt']].drop(['region', 'type', 'turq'])

        return a, nrc, annuity_factor, kmlines, capcost #, elec_use_perhh

a, nrc, annuity_factor, kmlines, capcost = fun_en()


annuity_factor.to_netcdf('..\\output\\en_annuity_factor_world.nc', mode='w')   
a.to_netcdf('..\\output\\en_a_world.nc', mode='w')
kmlines.to_netcdf('..\\output\\en_kmlines_world.nc', mode='w')
capcost.to_netcdf('..\\output\\en_capcost_world.nc', mode='w')

print('elec_network done')
print("--- %s minutes ---" % ((time.time() - start_time)/60))
