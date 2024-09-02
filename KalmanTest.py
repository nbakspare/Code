import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback, State
import json
import re
import os
import time
from datetime import datetime
from jupyterthemes import jtplot
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# currently installed theme will be used to
# set plot style if no arguments provided
from pykalman import KalmanFilter 
import numpy as np
import pandas as pd
#from scipy import poly1d  
from statsmodels.tsa.stattools import adfuller
from sklearn import linear_model
import statsmodels.api as sm
from scipy import signal
import plotly.graph_objs as go
from datetime import timedelta
from dash import dash_table

def get_freq(x,n,dt):
    '''EXTRACTS TOP THREE FREQUENCIES USING DFT'''
    # Expects x as dataframe, n as integer and dt as positive number
    x = x - sum(x)/len(x)
    assert sum(x)/len(x) < 0.0001 and sum(x)/len(x) > -0.0001
    f, Pxx_den = signal.periodogram(x,1/dt)
    res,i = [],0
    while i < n:
        freq_index = np.argmax(Pxx_den)
        res.append(f[freq_index])
        Pxx_den[freq_index] = min(Pxx_den)
        i += 1
    return res

def find_crossings_to_zero(data):
    crossings = (data.shift(1) >= 0) & (data < 0) | (data.shift(1) < 0) & (data >= 0)
    crossing_dates = data[crossings].index
    return crossing_dates

def find_crossings_to_x(data, x):
    crossings = (data.shift(1) >= x) & (data < x) | (data.shift(1) < x) & (data >= x)
    crossing_indices = data[crossings].index
    return crossing_indices

# Get the current working directory
current_directory = os.getcwd()

print("here")
# Print the current working directory
print("Current Directory:", current_directory)

#data_kalman = pd.read_csv('trades1.csv')
data_k = pd.read_csv('C:/Users/n_bakrania/Desktop/Projects/trades1.csv', index_col ='Unnamed: 0', parse_dates=True)
column_names = data_k.columns
column_names_list = column_names.tolist()
rate_1 = data_k[column_names_list[0]]
rate_2 = data_k[column_names_list[1]]
ratio = rate_2 - rate_1
indices = rate_1.index.tolist()

kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=0,
    initial_state_covariance=1,
    observation_covariance=1,
    transition_covariance=0.0001
)

mean, cov = kf.filter(ratio)
mean, std = mean.squeeze(), np.std(cov.squeeze())
freq = get_freq((ratio.values)[-90:],3,1)
HL = 1/(freq[1]*2)  

# Create the figure for the Kalman Filter Error
fig = go.Figure()
fig.add_trace(go.Scatter(x=indices, y=(ratio - mean)[-252*20:], mode='lines', line=dict(color='magenta', width=1)))
fig.add_trace(go.Scatter(x=indices, y=5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1)))
fig.add_trace(go.Scatter(x=indices, y=5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='red', width=1)))
fig.add_trace(go.Scatter(x=indices, y=4.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1)))
fig.add_trace(go.Scatter(x=indices, y=4.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='green', width=1)))
fig.add_trace(go.Scatter(x=indices, y=4 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1)))
fig.add_trace(go.Scatter(x=indices, y=4 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='yellow', width=1)))
fig.add_trace(go.Scatter(x=indices, y=3.5 * np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1)))
fig.add_trace(go.Scatter(x=indices, y=3.5 * -np.sqrt(cov.squeeze()[-252*20:]), mode='lines', line=dict(color='blue', width=1)))


fig.update_layout(
    title='Kalman Filter Error',
    xaxis_title='Day',
    yaxis_title='Value'
)

fig.update_xaxes(type='date')
fig.update_layout(width=1400)
fig.update_layout(height=1200)
y = (ratio - mean)[-252*20:]
date_of_largest_value = y.abs().idxmax()
crossing_dates = find_crossings_to_zero(y)
x = (ratio.values - mean)[-1]  # Replace with your desired value
crossing_indices = find_crossings_to_x(y, x)
d1 = crossing_indices
d2 = crossing_dates
# Initialize a global variable to store the time differences
global_diff = 0
totals = 0
nearest = datetime(2010, 10, 20)
columns = ['Historical Entry Dates', 'Historical Exit Dates', 'PnL']

main = pd.DataFrame(columns=columns)
print("here")
# Iterate through d1
for date1 in d1:
    if date1 < nearest:
        continue
    else:
        if d2 is None:
            break
        d2 = d2[d2 > date1]
        nearest_date2 = None
        min_time_diff= 0
        # Find the nearest date in d2 that is after or at the same time as date1
        time_diff = 0
        for date2 in d2:
            if date2 >= date1:
                time_diff = date2 - date1
                if nearest_date2 is None or date2 < nearest_date2:
                    min_time_diff = time_diff.days
                    nearest_date2 = date2
        # print(date1, nearest_date2, time_diff)
        print("here1")
        if str(date1) =='NaT':
            PnL = "-"
        else:
            PnL = ratio[nearest_date2]-ratio[date1]
        print("here2")
        data = {
            'Historical Entry Dates': date1,
            'Historical Exit Dates': nearest_date2,
            'PnL': PnL
        }
        main.loc[len(main.index)] = data
        if not nearest_date2:
            break
        if nearest_date2:
            if min_time_diff ==0:
                min_time_diff += 1
            # Add the time difference to the global variable
            global_diff += min_time_diff
            #print(global_diff, min_time_diff)
            totals += 1
        
        nearest = nearest_date2
    print("global diff: ",global_diff, " totals: ",totals)
    print(date1, nearest_date2)
average = global_diff/totals

print(main)