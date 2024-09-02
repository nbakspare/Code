import sys,os
import warnings
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import signal
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
        z = None
        return t,x,y
    else: 
        z = all_data.iloc[:,inp_3]
        return t,x,y,z

def writedata(inp,three_leg=True):
    '''WRITE DATA TO CSV FILE'''
    # Expects input to be stacked array/list
    if three_leg:
        np.savetxt('trade_list.csv', inp, delimiter=',', fmt='%s',header="Entry Date,Exit Date,Entry Abs,Exit Abs,Entry Z,Exit Z,Time Held,Weight 1,Weight 2,ADF,PnL %,PnL Abs,Outcome,Exit Reason")
    else: 
        np.savetxt('trade_list.csv', inp, delimiter=',', fmt='%s',header="Entry Date,Exit Date,Entry Abs,Exit Abs,Entry Z,Exit Z,Time Held,Weight,ADF,PnL %,PnL Abs,Outcome,Exit Reason")

def writedata_trend(inp):
    '''WRITE DATA TO CSV FILE'''
    # Expects input to be stacked array/list
    np.savetxt('trade_list.csv', inp, delimiter=',', fmt='%s',header="Entry Date,Exit Date,Entry Abs,Exit Abs,Time Held,PnL %,PnL Abs,Outcome,Exit Reason")
        
#################################################################################################################################
'''ANALYSIS SECTION'''

def ols_2(x,y):
    '''OLS REGRESSION BETWEEN TWO TIME SERIES'''
    # Expects x,y as dataframes and outputs spread and weights
    y_mod = sm.add_constant(y)
    model = sm.OLS(x,y_mod)
    res = model.fit()
    return x - res.params[1]*y,1,res.params[1]

def ols_trend(x,y):
    '''OLS REGRESSION BETWEEN TWO TIME SERIES'''
    # Expects x,y as dataframes and outputs weights
    x_mod = sm.add_constant(x)
    model = sm.OLS(y,x_mod)
    res = model.fit()
    residue = x * res.params[1] + res.params[0] - y
    return res.params[1],res.params[0],residue.std()

def ols_3(x,y,z):
    '''OLS REGRESSION BETWEEN THREE TIME SERIES'''
    # Expects x,y,z as dataframes and outputs spread and weights
    yz_comb = np.column_stack((y,z))
    yz_mod = sm.add_constant(yz_comb)
    model = sm.OLS(x,yz_mod)
    res = model.fit()
    return x - res.params[1]*y - res.params[2]*z,1,res.params[1],res.params[2]

def pca_2(x,y):
    '''PCA BETWEEN TWO TIME SERIES'''
    # Expects x,y as dataframes and outputs spread and weights
    out = np.column_stack((x,y))
    pca = PCA()
    pca.fit(out)
    x_w = pca.components_[0,1]
    y_w = pca.components_[0,0]
    return (x_w * x - y_w * y),x_w,y_w

def weight_threshold(weightthres,three_leg,w_1,*args):
    '''SETS WEIGHT THRESHOLD FOR ENTRY'''
    # Expect weightthres in the form of [lower1,upper1],[lower2,upper2] if 3 leg
    if three_leg == True:
        for a in args:
            w_2 = a
        if w_1 > weightthres[0,0] and w_1 < weightthres[0,1] and w_2 > weightthres[1,0] and w_2 < weightthres[1,1]:
            return True 
        else: 
            return False
    else: 
        if w_1 > weightthres[0,0] and w_1 < weightthres[0,1]:
            return True 
        else: 
            return False

def EMA_STD(df,com):
    '''GET EXPONENTIAL MOVING AVERAGE AND STD'''
    # Expects df as dataframe
    return df.ewm(com=com).mean(),df.std()

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

def plot_multiple(df,entry_exit,index,df_entry,ema_entry,status,tpsl_abs,tpsl_std,outcome,extype,df_exit,ema_exit,std,wind):
    '''PLOTS MULTIPLE GRAPHS'''
    num_trades = len(index)
    i,t = 0,0
    tp_std_buy,sl_std_buy,tp_std_sell,sl_std_sell = [],[],[],[]
    while t < num_trades:
        tp_std_buy.append(np.asarray(ema_exit[t] + std[t]*tpsl_std[t][1]))
        sl_std_buy.append(np.asarray(ema_exit[t] - std[t]*abs(tpsl_std[t][2])))
        tp_std_sell.append(np.asarray(ema_exit[t] + std[t]*tpsl_std[t][3]))
        sl_std_sell.append(np.asarray(ema_exit[t] + std[t]*abs(tpsl_std[t][0])))
        t += 1
    while i < num_trades:
        plt.figure(figsize=(12.8,3.8))
        plt.subplot(1,3,1)
        plt.plot(df_entry[i])
        plt.plot(ema_entry[i])
        plt.plot(wind,entry_exit[i][0],'og',label='Entry')
        if status[i][0] == True:
            plt.plot(wind,tpsl_abs[i][1],'^g',label='Take Profit')
            plt.plot(wind,tpsl_abs[i][2],'^r',label='Stop Loss')
            plt.title('Trade no. (entry frame): ' + str(i+1) + ' (buy signal)')
        else: 
            plt.plot(wind,tpsl_abs[i][3],'^g',label='Take Profit')
            plt.plot(wind,tpsl_abs[i][0],'^r',label='Stop Loss')
            plt.title('Trade no. (entry frame): ' + str(i+1) + ' (sell signal)')
        plt.legend(loc='best',prop={'size': 8})
        plt.xlabel('Days')
        plt.ylabel('Spread Value')
        plt.subplot(1,3,2)
        plt.plot(df_exit[i])
        plt.plot(ema_exit[i])
        plt.plot(wind,entry_exit[i][1],'or')
        if wind - index[i] - 1 >= 0:
            plt.plot(wind - index[i] - 1,entry_exit[i][0],'og')
        if status[i][0] == True:
            plt.plot(tp_std_buy[i],'--g',label='Std Take Profit')
            plt.plot(sl_std_buy[i],'--r',label='Std Stop Loss')
            plt.plot(wind,tpsl_abs[i][1],'^g')
            plt.plot(wind,tpsl_abs[i][2],'^r')
            plt.title('Trade no. (exit frame): ' + str(i+1) + ' (buy signal)')
        else: 
            plt.plot(tp_std_sell[i],'--g',label='Std Take Profit')
            plt.plot(sl_std_sell[i],'--r',label='Std Stop Loss')
            plt.plot(wind,tpsl_abs[i][3],'^g')
            plt.plot(wind,tpsl_abs[i][0],'^r')
            plt.title('Trade no. (exit frame): ' + str(i+1) + ' (sell signal)')
        plt.legend(loc='best',prop={'size': 8})
        plt.xlabel('Days')
        plt.ylabel('Spread Value')
        plt.subplot(1,3,3)
        ema,std = EMA_STD(pd.DataFrame(df[i]),wind)
        plt.plot(df[i])
        plt.plot(ema)
        plt.plot(29,entry_exit[i][0],'og',label='Entry')
        plt.plot(29+index[i],entry_exit[i][1],'or',label='Exit')
        if status[i][0] == True:
            plt.plot(29,tpsl_abs[i][1],'^g',label='Take Profit')
            plt.plot(29,tpsl_abs[i][2],'^r',label='Stop Loss')
        else: 
            plt.plot(29,tpsl_abs[i][3],'^g',label='Take Profit')
            plt.plot(29,tpsl_abs[i][0],'^r',label='Stop Loss')
        plt.title('Outcome: ' + str(outcome[i]) + ', Exit: ' + str(extype[i]))
        plt.legend(loc='best',prop={'size': 8})
        plt.xlabel('Days')
        plt.ylabel('Spread Value')
        plt.tight_layout()
        plt.show()
        i = i + 1