import sys,os
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import PCA

#################################################################################################################################
'''READ WRITE DATA SECTION'''

def readdata(path,**kwargs):
    '''READS DATA FROM CSV FILE'''
    # Expects path as string and outputs x,y,z as dataframes
    date = kwargs.get('date', None)
    inp_1 = kwargs.get('inp_1', None)
    inp_2 = kwargs.get('inp_2', None)
    inp_3 = kwargs.get('inp_3', None)
    if date == None:
        raise ValueError('must enter date location')
    all_data = pd.read_csv(path, sep=',')
    t = all_data.iloc[:,date]
    if inp_1 == None: 
        print('No inp_1 detected')
        x = None
    else: 
        x = all_data.iloc[:,inp_1]
    if inp_2 == None: 
        print('No inp_2 detected')
        y = None
    else: 
        y = all_data.iloc[:,inp_2]
    if inp_3 == None: 
        print('No inp_3 detected')
        z = None
    else: 
        z = all_data.iloc[:,inp_3]
    if z.any() == None:
        return t,x,y
    else:
        return t,x,y,z

def writedata(inp):
    '''WRITE DATA TO CSV FILE'''
    # Expects input to be stacked array/list
    np.savetxt('trade_list.csv', inp, delimiter=',', fmt='%s',header="Entry Date,Exit Date,Entry Abs,Exit Abs,Entry Z,Exit Z,Time Held,PnL %,PnL Abs,Outcome,Exit Reason")
    
#################################################################################################################################
'''ANALYSIS SECTION'''

def pca_2(x,y):
    '''PCA BETWEEN TWO TIME SERIES'''
    # Expects x,y as dataframes and outputs spread and weights
    out = np.column_stack((x,y))
    pca = PCA()
    pca.fit(out)
    x_w = pca.components_[0,1]
    y_w = pca.components_[0,0]
    return (x_w * x - y_w * y),x_w,y_w

def pca_3(x,y,z):
    '''PCA BETWEEN THREE TIME SERIES'''
    # Expects x,y,z as dataframes and outputs spread and weights
    out = np.column_stack((x,y,z))
    pca = PCA()
    pca.fit(out)
    x_w = pca.components_[0,2]
    y_w = pca.components_[0,1]
    z_w = pca.components_[0,0]
    return (x_w * x - y_w * y - z_w * z),x_w,y_w,z_w

def EMA_STD(df,lookback):
    '''GET EXPONENTIAL MOVING AVERAGE AND STD'''
    # Expects df as dataframe
    return df.ewm(com=lookback).mean(),df.std()

def half_life(df):
    '''CALCULATES HALFLIFE OF MEAN REVERSION'''
    # Expects df as dataframe
    df_lag = np.roll(df,1)
    df_lag[0] = 0
    df_ret = np.asarray(df - df_lag)
    df_ret[0] = 0
    df_lag2 = sm.add_constant(df_lag)
    model = sm.OLS(df_ret,df_lag2)
    res = model.fit()
    return np.log(0.5)/res.params[1],res.rsquared

#################################################################################################################################
'''MISC SECTION'''

def progressbar(it, prefix="", size=60, file=sys.stdout):
    '''PRINTS PROGRESS BAR'''
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
