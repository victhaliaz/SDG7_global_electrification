# -*- coding: utf-8 -*-
"""
This code calculates electrification rate progress over time for each IMAGE region.
It is based on regression coefficients obtained by Van Ruijven et al., DOI:10.1016/j.energy.2011.11.037

"""


import numpy as np
import os
import xarray as xr 
import time
start_time = time.time()

import grid_pop_hh as gph
import data_import as dat

#When running in parts for gph output
# import test as gph
# gph.regional_pop = xr.open_dataset('..\\output\\gph_regionalpop_world_SSP2.nc')
# gph.popg = xr.open_dataset('..\\output\\gph_popg_world_SSP2.nc')


os.getcwd()
DIR = "X:\\user\\zapatacasv\\sdg7\\scripts"   # Set the running directory here
os.chdir(DIR)

######regional electrification level with regression coefficients for total population
#data
regional_pop = gph.regional_pop
gdp_ppp = dat.gdp_ppp.sel(year = gph.popg.year)
gdp_ppp = gdp_ppp.transpose()
betas = dat.betas

#regression using Van Ruijven et al., coefficients
# Electricity rate regression for developing regions
elec_rate_regression = []
y = ()
for i in  dat.nrc-1:
    y = np.exp(betas['Beta_0'][i] + betas['Beta_1'][i]*np.log(gdp_ppp.values[i, ]) + betas['Beta_2'][i]*np.log(regional_pop['rpop_density'].values[i, ]) +betas['Beta_3'][i]*regional_pop['upop_rate'].values[i, ]*100) /   (np.exp(betas['Beta_0'][i] + betas['Beta_1'][i]*np.log(gdp_ppp.values[i, ]) + betas['Beta_2'][i]*np.log(regional_pop['rpop_density'].values[i, ]) + betas['Beta_3'][i]*regional_pop['upop_rate'].values[i, ]*100)+1)
    elec_rate_regression.append(y)

elec_rate_regression = np.asarray(elec_rate_regression)  


# Electricity rate regression for developed regions
elec_rate_regression1 = []
y = ()
for i in dat.nrc-1:
    y = np.exp(betas['Beta_0'][0] + betas['Beta_1'][0]*np.log(gdp_ppp.values[i, ]) + betas['Beta_2'][0]*np.log(regional_pop['rpop_density'].values[i, ]) +betas['Beta_3'][0]*regional_pop['upop_rate'].values[i, ]*100) /   (np.exp(betas['Beta_0'][0] + betas['Beta_1'][0]*np.log(gdp_ppp.values[i, ]) + betas['Beta_2'][0]*np.log(regional_pop['rpop_density'].values[i, ]) + betas['Beta_3'][0]*regional_pop['upop_rate'].values[i, ]*100)+1)
    elec_rate_regression1.append(y)

elec_rate_regression1 = np.asarray(elec_rate_regression1) 

mnm = (gdp_ppp >= 30000).astype(int)
elec_rate_regression = mnm* elec_rate_regression1 + (1-mnm) * elec_rate_regression
print('er regression done')
print("--- %s minutes---" % ((time.time() - start_time)/60))

#######calibrating total elec rate with historical data
his_elec_rates = dat.his_elec_rates.sel(year = gph.popg.year.where(gph.popg.year <= dat.cal_year, drop=True))
his_elec_rates = his_elec_rates.round(4)

diff = his_elec_rates['total'] - elec_rate_regression.where(elec_rate_regression.year <= dat.cal_year, drop=True)   #The difference in 2020 is spread until 2100 to ensure a smooth calibration
diff2 = []
b = diff.sel(year= dat.cal_year).values
year = elec_rate_regression.year.values
year =year[year>dat.cal_year]
a=()
for y in year:
        a = b*(1-((y - dat.cal_year)/(2100-dat.cal_year)))
        diff2.append(a)

diff2 = xr.DataArray(data = diff2, dims = ['year', 'region'], coords = {'year': year, 'region': np.arange(1,28,1)})
diff = xr.combine_by_coords([diff, diff2])

elec_rate = np.fmin(1, elec_rate_regression + diff)  


#function for avoiding peaks in the future typical from GDP but not for electrification rates.
def fun(x):
    o = np.where(x.year.values == dat.cal_year)[0]
    o=o[0]
    year = x.year.values
    elec_rate = x.values
    for y in range(o,elec_rate.shape[1]-1):   
        for r in range(0, elec_rate.shape[0]-1):
            elec_rate[r,y+1] = np.fmax(elec_rate[r,y], elec_rate[r,y+1])
    
    y = xr.DataArray(data = elec_rate, dims = ['region', 'year'], coords = {'year': year, 'region': dat.nrc})
    return(y)

elec_rate = fun(x = elec_rate)
elec_rate = elec_rate.to_dataset(name = 'total')

print('er calibration done total')
print("--- %s minutes---" % ((time.time() - start_time)/60))

#######Estimates the rural electrification level based on an empirical relation between rural and total electrification levels (Van Ruijven et al., coefficients) 
elec_rate['rural'] = np.fmin(elec_rate['total'], (0.9675*elec_rate['total'])**2 +(0.0183*elec_rate['total'])+0.004)

# Calibration of rural electrification
diff = his_elec_rates['rural'] - elec_rate['rural'].where(elec_rate['rural'].year <= dat.cal_year)   #The difference in 2020 is spread until 2100 to ensure a smooth calibration
diff2 = []
b = diff.sel(year= 2020).values
year = elec_rate_regression.year.values
year =year[year>2020]
a=()
for y in year:
        a = b*(1-((y - 2020)/(2100-2020)))
        diff2.append(a) 

diff2 = xr.DataArray(data = diff2, dims = ['year', 'region'], coords = {'year': year, 'region': dat.nrc}, name = 'rural')
diff = xr.combine_by_coords([diff, diff2])

temp = np.fmin(1, elec_rate['rural'] + diff)
elec_rate['rural'] = fun(temp['rural'])
print('er calibration done rural')
print("--- %s minutes---" % ((time.time() - start_time)/60))

# Calculates the urban electrification fraction from the total and rural rates. 
regional_pop['rpop_rate'] = 1- regional_pop['upop_rate']
elec_rate['urban'] =  np.fmin(1,((elec_rate['total']) - (elec_rate['rural']*regional_pop['rpop_rate']))/regional_pop['upop_rate'])
#calibration of urban electrification
diff = his_elec_rates['urban'] - elec_rate['urban'].where(elec_rate['urban'].year <= 2020)   #The difference in 2020 is spread until 2100 to ensure a smooth calibration
diff2 = []
b = diff.sel(year= 2020).values
year = elec_rate_regression.year.values
year =year[year>2020]
for y in year:
        a = b*(1-((y - 2020)/(2100-2020)))
        diff2.append(a) 

diff2 = xr.DataArray(data = diff2, dims = ['year', 'region'], coords = {'year': year, 'region':np.arange(1,28) }, name = 'urban')
diff = xr.combine_by_coords([diff, diff2])

temp = np.fmin(1, elec_rate['urban'] + diff)
elec_rate['urban'] = fun(temp['urban'])
print('er urban done')
print("--- %s minutes---" % ((time.time() - start_time)/60))

################## Universal access rate scenario based on target for rural and urban
###Calculating urban electrification level such as it achieves electrification targets set    
target = dat.target/100   
year = elec_rate_regression.year.values  
u =   (target-elec_rate['urban'].sel(year = dat.cal_year))/(dat.target_year-dat.cal_year) #Increase per unit of time 
elec_rate['urban_target'] = elec_rate['urban'].copy()
for y in year[year>dat.cal_year]:
    yy = int(np.where(year==y)[0])
    #print((y-dat.cal_year))
    elec_rate['urban_target'][dict(year= yy)] =  elec_rate['urban'].sel(year = dat.cal_year)+(u*(y-dat.cal_year))
print('er UA urban')

####Calculating rural electrification level such as it achieves electrification targets set
r =   (target-elec_rate['rural'].sel(year = dat.cal_year))/(dat.target_year-dat.cal_year) #Increase per unit of time 
elec_rate['rural_target'] = elec_rate['rural'].copy()
for y in year[year>dat.cal_year]:
    yy = int(np.where(year==y)[0])
    #print((y-dat.cal_year))
    elec_rate['rural_target'][dict(year= yy)] =  elec_rate['rural'].sel(year = dat.cal_year)+(r*(y-dat.cal_year))
print('er UA rural')
print("--- %s minutes---" % ((time.time() - start_time)/60))


## Calculating total from rural and urban with target 
a = np.fmax(elec_rate.total, elec_rate[ 'rural_target']*regional_pop['rpop_rate'] + elec_rate['urban_target']*regional_pop['upop_rate'])
elec_rate['total_target'] = xr.where(elec_rate.year <= dat.cal_year, elec_rate['total'] ,a)  
print('er UA rural')
elec_rate = elec_rate.transpose('region', 'year')

#elec_rate.to_netcdf('..\\output\\er.nc', mode='w')
print("--- %s minutes for elec rate ---" % ((time.time() - start_time)/60))