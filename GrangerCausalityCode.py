import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import grangercausalitytests

# This mini project reads in stock price data from Apple and Tesla, then performs a Granger Causality test to determine if one Granger causes the other.

filename_X = 'data/AAPL.csv'
filename_Y = 'data/TSLA.csv'

# Read the dates and scalar values of the time series into their dataframes.
df_X_raw = pd.read_csv(filename_X, usecols=['Date','Adj Close'])
df_X_raw.columns = ['Date','X']
df_Y_raw = pd.read_csv(filename_Y, usecols=['Date','Adj Close'])
df_Y_raw.columns = ['Date','Y']

# Apply the difference method to enforce stationarity.
df_X_diff = df_X_raw
df_Y_diff = df_Y_raw
df_X_diff[:]['X'] = df_X_diff[:]['X'].diff().dropna()
df_Y_diff[:]['Y'] = df_Y_diff[:]['Y'].diff().dropna()

# Combine the two dataframes and remove persisting NaN values.
df_diff = pd.merge(df_X_diff, df_Y_diff, on='Date', how='right')
df_diff['Date'] =  pd.to_datetime(df_diff['Date'])
df_diff = df_diff.set_index('Date').rename_axis('Indicator', axis=1)
df_diff = df_diff.dropna()
df_diff.head()

# ADF test null hypothesis assumes time series is not stationary.
def adf_test(df):
    result = adfuller(df.values)
    return result[1] # Returns p-value, which we want to be below 0.05.

# KPSS test null hypothesis assumes time series is stationary.
# NOTE: This function has p-value boundaries at 0.01 and 0.1 and often gives warnings. This still works for our purposes.
def kpss_test(df):    
    statistic, p_value, n_lags, critical_values = kpss(df.values)
    return p_value # Returns p-value, which we want to be above 0.05.

if adf_test(df_diff['X']) > 0.05:
    print("X is not stationary.")
if adf_test(df_diff['Y']) > 0.05:
    print("Y is not stationary.")
if kpss_test(df_diff['X']) < 0.05:
    print("X is not stationary.")
if kpss_test(df_diff['Y']) < 0.05:
    print("Y is not stationary.")
    
# If a time series does not pass the stationarity tests after differencing, it should be excluded from all Granger tests.

# Should set the lag so that it corresponds to the number of samples that are taken within the desired causal period.
lag = 15

result_XY = grangercausalitytests(df_diff[['Y','X']],maxlag=lag,verbose=False)
p_value_XY = round(result_XY[lag][0]['ssr_chi2test'][1],4)
print("p-value for X Granger causing Y: %f" % p_value_XY)

result_YX = grangercausalitytests(df_diff[['X','Y']],maxlag=lag,verbose=False)
p_value_YX = round(result_YX[lag][0]['ssr_chi2test'][1],4)
print("p-value for Y Granger causing X: %f" % p_value_YX)

# Typically, if a p-value is less than 0.05 we say that the first time series Granger causes the second.

# Next steps could be to iterate through a range of lags so that the strongest causal period can be found (which would have the smallest p-value).
