# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:30:50 2022

@author: nuzhat.yamin
"""

# In this code the functions are defined to run the main energy prediction loop

#Import libraries
import math                                                                     # package and call
                                                                                # it pd
import numpy as np  
import random
import copy
from scipy.stats import norm
import cvxpy as cvx
from sklearn.preprocessing import MinMaxScaler                                                            # import numpy
                                                                                # package and
                                                                                # call it np
import matplotlib.pyplot as plt      




# function to generate feature matrix            

def generate_feature_vector(slot, month, day, previous_slot_energies, previous_hour_energy, 
                                   previous_day_energy, EH_initial_estimation, Ebat_optimal, active_slots):
    
    # get the length of slot vector
    n_slots_prev = np.size(previous_slot_energies) - 1
    r1 = get_ratio_feature(previous_slot_energies[n_slots_prev], previous_slot_energies[n_slots_prev - 1])
    r2 = get_ratio_feature(previous_slot_energies[n_slots_prev - 1], previous_slot_energies[n_slots_prev - 2])
    r3 = get_ratio_feature(previous_slot_energies[n_slots_prev - 2], previous_slot_energies[n_slots_prev - 3])
    
    three_pt_derivative = three_point_derivative(previous_slot_energies[n_slots_prev - 2], 
                                                 previous_slot_energies[n_slots_prev - 1], previous_slot_energies[n_slots_prev])
    
    
    # Get the five point derivative
    
    n_minutes = 60
    five_pt_derivative_hour = np.zeros((n_minutes, 1))
    
    for m in range(5, np.size(previous_hour_energy)):
        five_pt_derivative_hour[m - 5] = five_point_derivative(previous_hour_energy[m-5],
                                                           previous_hour_energy[m-4],
                                                           previous_hour_energy[m-3],
                                                           previous_hour_energy[m-2],
                                                           previous_hour_energy[m-1])
        
    mean_5pt = np.mean(five_pt_derivative_hour)
    mean_5pt_30_min = np.mean(five_pt_derivative_hour[30:])
    mean_5pt_15_min = np.mean(five_pt_derivative_hour[45:])
 
    feature_vector = [month, day, slot+1, previous_slot_energies[n_slots_prev], 
                      previous_slot_energies[n_slots_prev - 1], previous_slot_energies[n_slots_prev - 2], 
                      previous_slot_energies[n_slots_prev - 3], 
                      three_pt_derivative, 
                      r1, r2, r3,
                      mean_5pt_15_min, mean_5pt_30_min, mean_5pt, 
                      previous_day_energy[2, slot], previous_day_energy[1, slot],
                      previous_day_energy[0, slot]]

    for hour in active_slots:
        feature_vector.append(EH_initial_estimation[hour])
    feature_vector.append(Ebat_optimal[slot])
    feature_vector.append(100-Ebat_optimal[slot])
                    
    
    return feature_vector 

def three_point_derivative(energy_1, energy_2, energy_3):
    ''' Function to get the three point derivative '''
    # energy_1: previous 3rd slot's energy
    # energy_2: previous 2nd slot's energy
    # energy_3: previous slot's energy
    
    derivative_3_pt = (energy_1 - 4*energy_2 + 3*energy_3)/2 ;
    
    return derivative_3_pt


def five_point_derivative(energy_1, energy_2, energy_3, energy_4, energy_5):
    ''' Function to get the five point derivative'''
    #energy_1:previous 4th slot's energy
    #energy_2:previous 3rd slot's energy
    #energy_3:previous 2nd slot's energy
    #energy_4:previous slot's energy
    #energy_5:current slot's energy

    derivative_5_pt = (3*energy_1 - 16*energy_2 + 36*energy_3 - 48*energy_4 + 25*energy_5)/12;
    
    return derivative_5_pt

def get_ratio_feature(energy_1, energy_2):
    ''' Function to get the ratio of two energies'''

    if energy_2 == 0: # When 0/0, -> 1
        energy_ratio = 1;    
    else: 
        energy_ratio = energy_1 / energy_2; 
        if energy_ratio > 100:
            energy_ratio = 100
    
    return energy_ratio

def get_date_matrix(year, year_index, date_data):
    date_req_month_indices = (date_data[:,year_index] == year)
    eh_dates = date_data[date_req_month_indices, :]
    return eh_dates

def energy_management(E_bat, EH_mat, Ec_alloc, exp_params, T, hour):
    '''function to implement energy management algorithm '''
    
    exp_params_relaxed = exp_params.copy()
    E_min = exp_params['minimum_energy']
    ME = exp_params['utility_ME']
    beta = exp_params['beta_value']
    alpha = exp_params['alpha_value']
    

    E_bat_new, Ec_est_new = energy_consumption_optimization(E_bat, Ec_alloc, EH_mat, 
                                                                     exp_params, T, hour)
    
    #if optimization is infeasible, relax target energy constraint
    if Ec_est_new is None:
        exp_params_relaxed['battery_target'] = E_bat[hour]
        E_bat_new, Ec_est_new = energy_consumption_optimization(E_bat, Ec_alloc, EH_mat, 
                                                                             exp_params_relaxed, T, hour)
    #allocate optimized energy consumption for the current hour
    if Ec_est_new is None:
        Ec_alloc[hour] = 0.0
    else:
        Ec_alloc[hour] = Ec_est_new[0]
    #ensure non-negative values
    if Ec_alloc[hour] < 0:
        Ec_alloc[hour] = 0.0
        
    return Ec_alloc
    

        
def energy_consumption_optimization(E_bat_initial, Ec_initial, EH_est, exp_params, T, hour):
    '''near optimal energy consumption optimization'''
    E_min = exp_params['minimum_energy']
    E_target = exp_params['battery_target']
    beta = exp_params['beta_value']
    alpha = exp_params['alpha_value']
    ME = exp_params['utility_ME']
    
    # Define and solve the CVXPY problem.
    n_hours_opt = T - hour
    E_bat = cvx.Variable(n_hours_opt+1)
    E_consumption = cvx.Variable(n_hours_opt)
    
    #create the constrains
    constraints = []
    constraints += [E_bat[0] == E_bat_initial[hour]]
    for t in range(0, n_hours_opt):
        constraints += [E_bat[t+1] == E_bat[t] + EH_est[t+hour] - E_consumption[t],
                        E_bat[t+1] >= E_min,
                        E_consumption[t] >= 0]
    
    constraints += [E_bat[n_hours_opt] >= E_target]

        
#    if hour > 0:
#        constraints += [E_bat[0:hour+1] == E_bat_initial[0:hour+1]]
#        constraints += [E_consumption[0:hour] == Ec_initial[0:hour]]
    utility = 0
    #define the objective
    for t in range(0, n_hours_opt):
        utility += (beta**(t+hour))*cvx.log((E_consumption[t]/ME)**alpha + 1e-5)
    objective = cvx.Maximize(utility)
    #form the problem and solve    
    prob = cvx.Problem(objective,constraints)
    prob.solve(solver=cvx.SCS )
    #print(hour, 'Problem status', prob.status)
    optimal_utility = prob.value
    
    return E_bat.value, E_consumption.value

def calculate_utility(energy_consumption, T, beta_val, ME, alpha):
    
    utility = 0
    utility_hourly = np.zeros(T)
    for t in range(0,T):
        if (energy_consumption[t] <= 0):
            energy_consumption[t] = 0
            curr_utility = -100
        else:
            curr_utility = (beta_val**t)*np.log((energy_consumption[t]/ME)**alpha)
        utility += curr_utility
        utility_hourly[t] = curr_utility
        
    return utility, utility_hourly


def energy_management_optimal(actual_energy_mat,
                                exp_params,
                                T, day):

    '''function to implement energy management algorithm'''
	
    exp_params_relaxed = exp_params.copy()
    E_begin = exp_params['start battery']
    E_min = exp_params['minimum_energy']
    ME = exp_params['utility_ME']
    beta = exp_params['beta_value']
    alpha = exp_params['alpha_value']
    alpha_ewma = exp_params['alpha_EWMA']
    

    #do the optimization to get initial energy allocation    
    Ec_alloc_opt = [0]*T
    E_bat_opt = [0]*(T+1)
    E_bat_opt[0] = E_begin
 
        
    print('day',day)
    
    EH_act = actual_energy_mat[day,:]

    #get optimal solution using actual EH
    E_bat_optimal, Ec_est_optimal = energy_consumption_optimization(E_bat_opt, Ec_alloc_opt, EH_act, 
                                                    exp_params, T, 0)
    
    #calculate optimal utility
    utility_optimal, Ec_est_optimal = calculate_utility(Ec_est_optimal, T, beta, ME, alpha)
    


    optimal_results = {'E_consumption' : Ec_est_optimal,
                       'E_bat' : E_bat_optimal, 
                       'Utility' : utility_optimal}
    
    return optimal_results


