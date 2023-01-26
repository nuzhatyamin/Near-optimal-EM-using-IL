
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:35:52 2022

@author: nuzhat.yamin
"""

#import libraries
import scipy.io as sio
import numpy as np
import os
import sys
from pathlib import Path
from functions import *
import pickle

import pandas as pd
import matplotlib.pyplot as plt

import time
import copy

'''load data to generate features'''
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


date_df = pd.DataFrame(date_matrix[:,0:3], columns = ['year','month','day'])

years = list(set(date_matrix[:,year_idx]))

'''load oracle policy data'''
EM_data_dir = Path(r'data/EM_data')

Ebat_optimal = np.load(EM_data_dir / 'optimal_Ebat.npy')
target_Ec = np.load(EM_data_dir / 'optimal_Ec.npy')

# get initial energy allocation values
initial_hour = 0
initial_Ec = np.load(EM_data_dir / 'initial_Ec.npy')

#Filtering out the slot indexes that yield no solar energy at any time of year
active_slots = np.nonzero(np.any(data != 0, axis=0))[0]

warmup_period = 3;  #Testing will be done from 4th day. 


'''feature matrix description'''
feature_vector = {'month': 0, 'date' : 1, 'slot' : 2,
                     'energy_s1': 3, 'energy_s2': 4, 'energy_s3': 5, 'energy_s4': 6,
                     'energy_3ptderivative': 7, 
                     'ratios12' : 8, 'ratios23' : 9, 'ratios34' : 10,
                     'avg_15min_derivative': 11, 'avg_30min_derivative': 12, 'avg_60min_derivative': 13,
                     'energy_d1_s1' : 14, 'energy_d2_s1' : 15, 'energy_d3_s1' : 16,
                      'Eh_init_s4' : 17, 'Eh_init_s5' : 18, 'Eh_init_s6' : 19, 'Eh_init_s7' : 20, 
                      'Eh_init_s8' : 21,'Eh_init_s9' : 22, 'Eh_init_s10' : 23, 'Eh_init_s11' : 24, 
                      'Eh_init_s12' : 25, 'Eh_init_s13' : 26, 'Eh_init_s14' : 27, 'Eh_init_s15' : 28, 
                      'Eh_init_s16' : 29,'Eh_init_s17' : 30, 'Eh_init_s18' : 31, 'Eh_init_s19' : 32, 
                      'Ebat_optimal' : 33, 'Ebat_target' : 34}


feature_vector_size = len(feature_vector)

T = 24
n_minutes = 1440
n_previous_slots = 4 
n_previous_days = 3
# Get initial features

previous_day_energy = np.zeros((3, 24)) # three previous days for 24 hours

previous_day_energy[0,:] = data[0,:]
previous_day_energy[1,:] = data[1,:]
previous_day_energy[2,:] = data[2,:]

previous_slot_energies = np.zeros((n_previous_slots))

previous_slot_energies[:] = data[2, T-n_previous_slots:T]

previous_hour_energy = np.zeros((60 + 5, 1))  # plus 5 due to the five pt derivative
previous_hour_start_index = (active_slots[0] - 1)*60 - 5
previous_hour_end_index = (active_slots[0])*60

previous_hour_energy = data_by_minute[2,n_minutes-(60+5):n_minutes]
feature_matrix = np.zeros(((np.shape(data)[0]-warmup_period-1)*T,len(feature_vector)+1))


'''Generate the training data features'''
feature_index = 0
for day in range(warmup_period+1, np.shape(data)[0]):
    train_feature_matrix = np.zeros((T+1, feature_vector_size))
    train_actual_Ec = np.zeros((T+1,1))
    train_actual_EH = np.zeros((T+1,1))
    curr_feature_idx = 1
    curr_year = date_df.iloc[day]['year']
    curr_month = date_df.iloc[day]['month']
    curr_day = date_df.iloc[day]['day']
    #Eh_initial = Eh_est_initial[day,:]
    Eh_initial = data[day-1, :]
    Ebat = Ebat_optimal[day,:]
    predicted_energy = np.zeros(horizon_len)
    
    # go through the slots and get the features
        
    for slot in range(T):
        #print(previous_hour_energy)
        actual_energy_curr_slot = data[day, slot]
        #EH_initial_est = EH_init_mat[day,:]
        curr_features = generate_feature_vector_simile(slot, curr_month, curr_day, previous_slot_energies, previous_hour_energy,
                                                previous_day_energy,  Eh_initial, Ebat, active_slots)

        feature_matrix[feature_index, 0] = curr_year
        feature_matrix[feature_index, 1:] = curr_features
        train_feature_matrix[curr_feature_idx, :] = np.transpose(curr_features)
        train_actual_Ec[curr_feature_idx] = target_Ec[day,slot]
        train_actual_EH[curr_feature_idx] = actual_energy_curr_slot
        
        curr_feature_idx = curr_feature_idx + 1
        
        # Update the previous slot energies and previous hour energies for the next feature generation
        
        previous_hour_start_index = slot*60 
        previous_hour_end_index = (slot+1)*60
        previous_hour_energy[0:5] = previous_hour_energy[60:]
        previous_hour_energy[5:] = data_by_minute[day,previous_hour_start_index:previous_hour_end_index]
        
        previous_slot_energies[0:n_previous_slots-1] = previous_slot_energies[1:n_previous_slots]
        previous_slot_energies[n_previous_slots-1] = actual_energy_curr_slot
        #replace initial energy harvest estimate with actual energy of current hour 
        Eh_initial[slot] = actual_energy_curr_slot
        feature_index = feature_index + 1
    
    # End of the day. Update the previous day energies
    previous_day_energy[0:2, :] = previous_day_energy[1:3, :]
    previous_day_energy[2,:] = data[day,:]
    
    train_matrix = np.concatenate((train_feature_matrix, train_actual_Ec, train_actual_EH), axis =1)
    #fill up initial hour features and target
    train_matrix[initial_hour, :] = train_matrix[1, :]
    train_matrix[initial_hour, feature_vector['slot']] = initial_hour
    train_matrix[initial_hour, len(feature_vector)] = initial_Ec[day]
    if curr_year == 2015:
        file_loc = 'Data/Train_files/'
    elif curr_year == 2016:
        file_loc = 'Data/Valid_files/'
    else:
        file_loc = 'Data/Test_files/'
    with open(file_loc + 'day_' + str(curr_month) + '_' + str(curr_day) + '_' + str(curr_year) + '.p','wb') as f:
        pickle.dump(train_matrix, f)
