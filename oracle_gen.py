# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:04:43 2023

@author: Monami
"""
#import libraries
import scipy.io as sio
import numpy as np
import os
import sys
from pathlib import Path
from functions_EH_multi_horizon import *
import pandas as pd
import csv

import seaborn as sns
import matplotlib.pyplot as plt

data_dir = Path(r'data/EH_data')

date_time_data = sio.loadmat(data_dir / "date_time_matrix_all.mat")['date_time_matrix_all']
date_matrix = sio.loadmat(data_dir / "date_matrix_all.mat")['date_matrix']

#Indices in the date matrix
year_idx = 0;
month_idx = 1;
day_idx = 2;

#Get dates for train data
eh_dates_2015 = get_date_matrix(2015, year_idx, date_matrix)
#Get dates for validation data
eh_dates_2016 = get_date_matrix(2016, year_idx, date_matrix)
#Get dates for test data
eh_dates_2017 = get_date_matrix(2017, year_idx, date_matrix)
eh_dates_2018 = get_date_matrix(2018, year_idx, date_matrix)
eh_dates_2019 = get_date_matrix(2019, year_idx, date_matrix)
eh_dates_2020 = get_date_matrix(2020, year_idx, date_matrix)

# Load energy by minute data for 6 years

data_2015 = sio.loadmat(data_dir / 'EH_byminute_2015.mat')['energy_by_minute'] # 2015
data_2016 = sio.loadmat(data_dir / 'EH_byminute_2016.mat')['energy_by_minute'] # 2016
data_2017 = sio.loadmat(data_dir / 'EH_byminute_2017.mat')['energy_by_minute'] # 2017
data_2018 = sio.loadmat(data_dir / 'EH_byminute_2018.mat')['energy_by_minute'] # 2018
data_2019 = sio.loadmat(data_dir / 'EH_byminute_2019.mat')['energy_by_minute'] # 2019
data_2020 = sio.loadmat(data_dir / 'EH_byminute_2020.mat')['energy_by_minute'] # 2020

EH_data_hourly_2015 = np.zeros((np.size(data_2015,0),24))
EH_data_hourly_2016 = np.zeros((np.size(data_2016,0),24))
EH_data_hourly_2017 = np.zeros((np.size(data_2017,0),24))
EH_data_hourly_2018 = np.zeros((np.size(data_2018,0),24))
EH_data_hourly_2019 = np.zeros((np.size(data_2019,0),24))
EH_data_hourly_2020 = np.zeros((np.size(data_2020,0),24))

#Convert minute data into hourly data
for hour in range(24):
    EH_data_hourly_2015[:,hour] = np.sum(data_2015[:,60*(hour):60*(hour)+60],axis = 1)
    EH_data_hourly_2016[:,hour] = np.sum(data_2016[:,60*(hour):60*(hour)+60],axis = 1)
    EH_data_hourly_2017[:,hour] = np.sum(data_2017[:,60*(hour):60*(hour)+60],axis = 1)
    EH_data_hourly_2018[:,hour] = np.sum(data_2018[:,60*(hour):60*(hour)+60],axis = 1)
    EH_data_hourly_2019[:,hour] = np.sum(data_2019[:,60*(hour):60*(hour)+60],axis = 1)
    EH_data_hourly_2020[:,hour] = np.sum(data_2020[:,60*(hour):60*(hour)+60],axis = 1)
    
#load data

data_by_minute = np.concatenate((data_2015, data_2016, data_2017, data_2018, 
                                      data_2019, data_2020), axis =0)    

data = np.concatenate((EH_data_hourly_2015, EH_data_hourly_2016, EH_data_hourly_2017, EH_data_hourly_2018,
                             EH_data_hourly_2019, EH_data_hourly_2020), axis =0)
date_matrix = np.concatenate((eh_dates_2015, eh_dates_2016, eh_dates_2017, eh_dates_2018, 
                                   eh_dates_2019, eh_dates_2020), axis =0)

#Load data
actual_energy = data

Ec_optimal = np.zeros((np.shape(actual_energy)[0],T))
Ebat_optimal = np.zeros((np.shape(actual_energy)[0],T+1))

exp_params = {'minimum_energy' : 10, 
              'start battery' : 100,
              'battery_target' : 100,
              'beta_value' : 0.99,
              'alpha_value' : 1,
              'utility_ME' : 8.0153,
              'alpha_EWMA': 0.5}

#run energy management optimal algorithm to get the optimal Ec, Ebat
for day in range(0, np.shape(predicted_energy_matrix)[1]): 
    

    optimal_results = energy_management_optimal(actual_energy,
                                                        exp_params, T, day)        
     
    utility_optimal.append(optimal_results['Utility'])
    Ec_optimal[day,:] = optimal_results['E_consumption']
    Ebat_optimal[day,:] = optimal_results['E_bat']
    
    
    
