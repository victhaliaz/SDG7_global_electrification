# -*- coding: utf-8 -*-
"""
24/02/2022

@author: zapatacasv
"""

import os
#from tkinter import Y
   
os.getcwd()
DIR = "X:\\user\\zapatacasv\\sdg7\\scripts"   # Set the running directory here
os.chdir(DIR)

import numpy as np
from pym import read_mym
import xarray as xr       


####time constants
t0 = 2020
tf= 2030
dt = 1 #10 # This can be changed to 1 year to follow the TIMER frequency
baseyear = 2020 #year of the distance to grid map 
cal_year = 2020   #Last year of empirical data on electrification for calibration
target_year = 2030

#Geographic scope
nrc = np.arange(1,28)   #SSA: [8,9,10,26]

#Scenarios:
ssp = 'SSP2' #for this model data folder (population grid)
####Loading data from TIMER
scenario = 'SSP2' #for scenlib library
scen_outputlib = 'PEP-2C-AdvPE' #'SSP2' #'PEP-2C-AdvPE_DLS'    #'SSP2_DLS'  ##  #SSP2_DLS #           
TIMER_dir = "X:\\user\\zapatacasv\\TIMER\\TIMER\\2_TIMER\\"

####Other constants in alfabetical order
BatteryLifetime = 10			#! years	-The technical lifetime of the storage battery - DFID (2013), Low Carbon Mini Grids- REN21 (2014), Mini-grid Policy Toolkit= 8
co2_ldiesel = 2.68 #kg CO2/l from fue combustion   https://ghgprotocol.org/calculation-tools#cross_sector_tools_id  
CookEnDemCap = 3			#! MJ/capita per day, Zubi et.al (2017) estimates 0.6kWh/day for a family of 6 with an efficient multi-cooker
ClimIndEff = 1    #Real main.ClimInduced_Efficiency   #	! increase in efficiency from climate policy
CookEff =	0.85	#! Efficiency of an induction stove
DG_DieselEnergyCont = 38.29		#	! MJ/lt	-The energy content of diesel fuel 
DG_FuelConsumpperkwh = [0.266, 0.303]		      #! lt/kwh	-The fuel consumed per kwh produced at full load (lt), DGHeatRate/DG_DieselEnergyCont, 1=for 95KW, 2=for 30KW- Pelland et.al., 2011
disagg_factor = 0.5            	# Disaggregation factor, represents clustering of settlements. # Value of zero for clustered settlements, 1 for fully scattered distribution. 0.5 is a first estimate, half the rural population lives in villages
GJ2KWh = 277.7778      #	! KWh/GJ - GJ to KWh conversion factor
load_hours_daily= [8,	4]   #!Total Load hours per day  1= total, 2= evening peak
LV_capac = 10 				      #! kw	-Maximum capacity of an LV line
LVunitfactor = 1.3333 			#! Volgt uit Sebitosi et al (2006) en eigen analyse voor Alternative model 2 in T&DinvestmentspreadsheetSensitivity2.xls
max_LVdist = 30				     #! km	-Maximum length of an LV line
MiniG_distributionloss = 0.10   #	! frac	-The distribution loss of the minigrid network fixed at 10% - Mainali (2014) (This is 2% in Parshal et.al., 2009, unrealistic for SSA)
MiniG_TechLifetime = [25, 25, 25]	      #! years	-The technical life time of the mini grid systems (years)  F.F. Nerini et al. (2016), ESMAP (2007) p.52, Taliotis (2012) TEMBA model
MiniG_StoragekwhperSyskw = [4, 0, 2, 0, 0, 0]	  #! Mini grid hours of storage for the calculated system capacity, 1= Solar, 2=Diesel, 3=Wind. The system capacity on solar and wind is based on middle load, while for diesel it covers already the peakload
PVBOSCost =	0.45			#! frac	-Balance of system costs are taken to be 45% of the whole system cost
PVBOSLifetime =	15			#! years	-Mini grid component technical lifetime- SunShot Vision Study – February 2012
StAl_StoragekwhperSyskw = [4, 0]			#! Storage capacity required- storage for one full day damand 1=Soalr, 2=Diesel
StAl_TechLifetime = [25, 10]			#! years	-Technical lifetime of standalone PV panels[1] and Genset[2] Taliotis (2012) TEMBA model
SWERcapacity = 50 				#! kw	-Maximum capacity of a SWER line. Nu 50 kW nl de helft van 100 kW, dit is gebaseerd op de aanname dat het systeem in vraag/Load moet kunnen verdubbelen... maar zou ook verdrievoudigen / verviervoudigen kunnen zijn...
TDCapCost_Urban = 500			#! USD	-Capital cost of Urban electrification, per household. Based on Rosnes & vennemo (2012), The cost of providing electricity to Africa
td_life = 20                     #years of life time for transmission and distribution lines
td_capcost_urban = 500			# USD	-Capital cost of Urban electrification, per household. Based on Rosnes & vennemo (2012), The cost of providing electricity to Africa

#   Scenario data from TIMER
path = TIMER_dir + "scenlib\\TIMER_3_3\\baselines\\" + scenario + "\\"
#Population    
dat = read_mym(path + 'pop\\pop.scn') 
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
pop = xr.DataArray(data = dat[0][r0:r1, :], dims = ['year', 'region'], coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1)})
pop = pop.interp(year= range(t0, tf+1, dt), method="linear")

##########  Connects to data\\endem\\residential\\
#betas  
path = TIMER_dir + "TIMER_3_3\\data\\data\\endem\\"
betas = { 'Beta_0' :  read_mym(path + 'residential\\Beta_0.dat'),  'Beta_1' :  read_mym(path + 'residential\\Beta_1.dat'), 'Beta_2' :  read_mym(path + 'residential\\Beta_2.dat') , 'Beta_3' :  read_mym(path + 'residential\\Beta_3.dat')}

########## Timer ouputlib 
path = TIMER_dir + "outputlib\\TIMER_3_3\\TIMER\\" + scen_outputlib + "\\"
# GDP_PPP    
dat = read_mym(path + 'tuss\\global\\gdp_ppp.scn') 
r0 = int(np.where(dat[1]==t0)[0])
r1 = int(np.where(dat[1]==tf)[0])+1
gdp_ppp = dat[0][:,:-1] #resizing from 28 to 27 regions
gdp_ppp = xr.DataArray(data = gdp_ppp[r0:r1, :], dims = ['year', 'region'], coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1)})

#ElecProdShare
dat = read_mym(path + 'tuss\\enepg\\ElecProdShare.out') 
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
elec_prod_share = xr.DataArray(data = dat[0][r0:r1, 0:27, :], dims = ['year', 'region', 'ntc2'], 
        coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 
                  'ntc2':np.arange(1,dat[0].shape[2]+1)})

#SystemTotalCostperkWh[NRC2,3](t);! $/kWhe (1 = fixed, 2 = variable, 3 = total costs)
dat = read_mym(path + 'tuss\\enepg\\SystemTotalCostperkWh.out') 
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
pad_width = ((0, 0), (0, 1), (0, 0)) # create padding
system_total_cost_perkwh =  np.pad(dat[0], pad_width, mode='constant', constant_values=np.nan)
system_total_cost_perkwh = xr.DataArray(data = system_total_cost_perkwh[r0:r1, :, :], dims = ['year', 'region', 'tres'], 
        coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 
                  'tres':np.arange(1, dat[0].shape[2]+1)})

#PriceSecFuel [NR27,NS,NEC](t)	! TIMER fuel prices (used for calculating diesel LCOE). PriceSecFuel[NRC,NS,NEC](t),   ! $2005/GJ, NS=3   Diesel use [R,3,2] 2 for liquid fuels, 3 for sector choice. This price goes up with a carbon tax.
dat = read_mym(path + 'tuss\\endem\\PriceSecFuel.out') 
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
PriceSecFuel = xr.DataArray(data = dat[0][r0:r1, :], dims = ['year', 'region', 'NS', 'NEC'], coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 'NS' : np.arange(1,6,1), 'NEC' : np.arange(1,9,1)})

#CapCostNew[NRC,NTC2](t),	! $/kWe		Capital cost of new capacity
dat = read_mym(path + 'tuss\\enepg\\CapCostNew.out') 
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
cap_cost_new = xr.DataArray(data = dat[0][r0:r1, :, :], dims = ['year', 'region', 'ntc2'],  coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 'ntc2':np.arange(1,dat[0].shape[2]+1)})

#ElecConstrCostNew[NRC2,NTC2](t)	
dat = read_mym(path + 'tuss\\enepg\\ElecConstrCostNew.out') 	
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
pad_width = ((0, 0), (0, 1), (0, 0)) # create padding
elec_constrcost_new =  np.pad(dat[0], pad_width, mode='constant', constant_values=np.nan)
elec_constrcost_new = xr.DataArray(data = elec_constrcost_new[r0:r1, :, :], dims = ['year', 'region', 'ntc2'],  coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 'ntc2':np.arange(1,dat[0].shape[2]+1)})

#OMCostperkWNew[NRC2,NTC2](t)	
dat = read_mym(path + 'tuss\\enepg\\OMCostperkWNew.out')	
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
pad_width = ((0, 0), (0, 1), (0, 0)) # create padding
om_cost_perkw_new =  np.pad(dat[0], pad_width, mode='constant', constant_values=np.nan)
om_cost_perkw_new = xr.DataArray(data = om_cost_perkw_new[r0:r1, :, :], dims = ['year', 'region', 'ntc2'],  coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 'ntc2':np.arange(1,dat[0].shape[2]+1)})

#Household size #time from 1971 to 2050, 27 regions, 13 turq #! Nr. of population groups (1=Total, 2=Urban, 3=Rural, 4-8=Urban quintiles, 9-13=Rural quintiles)
dat = read_mym(path + 'T2RT\\endem\\residential\\res_HouseholdSize.out')
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
hhsize = xr.DataArray(data = dat[0][r0:r1, :], dims = ['year', 'region', 'turq'], coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 'turq': np.arange(1,14,1)})


# var_cost_perkwh_avg
dat = read_mym(path + 'T2RT\\enepg\\VarCostperkWhAvg.out') 
r0 = int(np.where(dat[1]==t0)[0])
r1= int(np.where(dat[1]==tf)[0])+1
pad_width = ((0, 0), (0, 1), (0, 0)) # create padding
var_cost_perkwh_avg =  np.pad(dat[0], pad_width, mode='constant', constant_values=np.nan)
var_cost_perkwh_avg = xr.DataArray(data = var_cost_perkwh_avg[r0:r1, :, :], dims = ['year', 'region', 'ntc2'], 
        coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 
                  'ntc2':np.arange(1,dat[0].shape[2]+1)})

#! kgC/GJ-Elec Emissions of the power sector in TIMER main.em.CCElec[28](t) #CFCO2C = 12/44, #! g C/g CO2        Conversion factor CO2-C
dat = read_mym( TIMER_dir +  "outputlib\\TIMER_3_3\\TIMER\\" + scen_outputlib + "\\indicators\\EmissionFactors\\CCelec.out")
r0 = int(np.where(dat[1]==t0)[0])
r1 = int(np.where(dat[1]==tf)[0])+1
CCelec = xr.DataArray(data = dat[0][r0:r1, :-1]*3.66, dims = ['year', 'region'], coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1)})  #3.6 for converting to CO2 emissions

#energy mix for electricity ElecProdTotShare[NRCT=28,14](t),			! share GJ/GJ 	Electricity production share per aggregated technology groups
#!1: Solar, !2: Wind, !3: Hydro !4: Other renewables !5: Hydrogen !6: Nuclear !7: Coal !8: Coal CCS !9: Oil !10: Oil CCS !11: Natural gas !12: Natural gas CCS !13: Biomass !14: Biomass CCS
dat = read_mym( TIMER_dir +  "outputlib\\TIMER_3_3\\TIMER\\" + scen_outputlib + "\\T2RT\\enepg\\ElecProdTotShare.out")
r0 = int(np.where(dat[1]==t0)[0])
r1 = int(np.where(dat[1]==tf)[0])+1
ElecProdTotShare = xr.DataArray(data = dat[0][r0:r1, :-1, :], dims = ['year', 'region', 'esource'], coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 'esource':np.arange(1,dat[0].shape[2]+1)}) 

########## Not yet in outputlib, but it is an output from TIMER
#elec_use_perhh 
dat = read_mym(path + 'T2RT\\endem\\residential\\ElecUsePerHH.out') 
r0 = int(np.where(dat[1]==t0)[0])
r1 = int(np.where(dat[1]==tf)[0])+1
elec_use_perhh = xr.DataArray(data = dat[0][r0:r1, :], dims = ['year', 'region', 'turq'], coords = {'year': dat[1][r0:r1], 'region': np.arange(1,28,1), 'turq': np.arange(1,14,1)})


################# Own module data
path2 =     "..\\data\\Target.scn"
target = read_mym(path2)

#interest #all countries set to 10% interest
path2 =     "..\\data\\Interest.dat"
interest = read_mym(path2)
r0 = int(np.where(interest[1]==t0)[0])
r1= int(np.where(interest[1]==tf)[0])+1
interest = xr.DataArray(data = interest[0][r0:r1, :], dims = ['year', 'region'], coords = {'year': interest[1][r0:r1], 'region': np.arange(1,28,1)})
interest = interest.interp(year= range(t0, tf+1, dt))

#!Lead-Acid battery cost per kilowatt-hour- DFID 2013, Low Carbon Mini Grids p24, 1=Minigrid, 2=standalone
path2 =     "..\\data\\BatteryKwhCost.dat"
BatteryKwhCost = read_mym(path2)

#ElecCapCost: $/km	-Specifies capital costs for electrification infrastructure  1=69kV HV line ($28000/km), 2= 11kV MV line ($6000/km), 3 = cost of LV line $/km, 4= transformer price to LV, 5= ttransformer price MV/HV
path2 =     "..\\data\\ElecCapCost_SSA.dat"
ElecCapCost = read_mym(path2)
ElecCapCost = xr.DataArray(data = ElecCapCost[0], dims = ['year', 'type'], coords = {'year': ElecCapCost[1], 'type': np.arange(1,6,1)})
ElecCapCost = ElecCapCost.interp(year= range(t0, tf+1, dt))

#DG_CostPerSystemkw  ! The cost of the generator per system kilowatt ($/kw) - Deichmann et.al., 2011, 1=minigrid, 2=standalone
path2 =     "..\\data\\DG_CostPerSystemkw.dat"
DG_CostPerSystemkw = read_mym(path2)
DG_CostPerSystemkw = xr.DataArray(data = DG_CostPerSystemkw[0], dims = ['year', 'dos'], coords = {'year': DG_CostPerSystemkw[1], 'dos': np.arange(1,3,1)})
DG_CostPerSystemkw = DG_CostPerSystemkw.interp(year= range(t0, tf+1, dt), method = 'nearest')  #nearest for using the literature data only
DG_CostPerSystemkw = xr.where(DG_CostPerSystemkw.year > 2030, DG_CostPerSystemkw.sel(year = 2030),  DG_CostPerSystemkw) #for 2040 and 2050 I use th elast available value of 2030

####World bank historical data on electricity rates  
path2 = "..\\data\\hisElec_rate_WorldBank_iregion_popweighted.nc" #pre-processed data to scale from country to region using popweights. Includes rural, urban and total
his_elec_rates = xr.open_dataset(path2) 

del path2
del dat
