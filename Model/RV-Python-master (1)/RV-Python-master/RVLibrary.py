import math
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from statsmodels.tsa.stattools import adfuller
from utils import *

def generate_trade(df,args,weightthres,three_leg,pca,weight_overide):
    '''GENERATE TRADE RECOMMENDATION FROM DATAFRAME'''
    # Expects df as dataframe, args in form of [Upper Z Score Entry, Lower Z Score Entry], three_leg = True/False
    UTH,LTH,MAXTH,lookback = args
    def cond_trade(df_today,EMA_today,std):
        std_dev = (df_today - EMA_today)/std
        if std_dev > UTH and std_dev < MAXTH:
            sell = True
            buy = False
        elif std_dev < LTH and std_dev > -MAXTH:
            buy = True
            sell = False
        else:
            buy = False
            sell = False
        return [buy,sell]
    if three_leg == True: 
        x,y,z = df
        if weight_overide == None:  
            df,x_w,y_w,z_w = ols_3(x,y,z)
        else: 
            df,x_w,y_w,z_w = weight_overide[0]*x - weight_overide[1]*y - weight_overide[2]*z,weight_overide[0],weight_overide[1],weight_overide[2]
        df_EMA,std = EMA_STD(df,lookback)
        adf_res = adfuller(df_EMA - df)
        df_today,EMA_today = df.iloc[-1],df_EMA.iloc[-1]
#         HL,r_sqr = half_life(df_EMA - df)
        freq = get_freq(df,3,1)
        HL = 1/(freq[1]*2)
        status = cond_trade(df_today,EMA_today,std)
        x_today,y_today,z_today = x.iloc[-1],y.iloc[-1],z.iloc[-1]
        if weight_threshold(weightthres,three_leg,y_w/x_w,z_w/x_w) == False:
            status = [False,False]
#         if r_sqr < 0.05 and True in status:
# #             warnings.warn('Halflife estimate inaccurate, time stop DISABLED. R-Squared:' + str('%.2f' % r_sqr))
#             HL = math.inf
        return status,[x_w,y_w,z_w],[df_today,(df_today - EMA_today)/std,std,HL],df,df_EMA,adf_res[1],[x_today,y_today,z_today]
    else:
        
        x,y = df
        if pca == False:
            if weight_overide == None:
                df,x_w,y_w = ols_2(x,y)
            else: 
                df,x_w,y_w = weight_overide[0]*x - weight_overide[1]*y,weight_overide[0],weight_overide[1]
        elif pca == True:
            if weight_overide == None:
                df,x_w,y_w = pca_2(x,y)
            else: 
                df,x_w,y_w = weight_overide[0]*x - weight_overide[1]*y,weight_overide[0],weight_overide[1]
        df_EMA,std = EMA_STD(df,lookback)
        adf_res = adfuller(df_EMA - df)
        df_today,EMA_today = df.iloc[-1],df_EMA.iloc[-1]
#         HL,r_sqr = half_life(df_EMA - df)
        freq = get_freq(df,3,1)
        HL = 1/(freq[1]*2)
        status = cond_trade(df_today,EMA_today,std)
        x_today,y_today = x.iloc[-1],y.iloc[-1]
        if weight_threshold(weightthres,three_leg,y_w/x_w) == False:
            status = [False,False]
#         if r_sqr < 0.05 and True in status:
# #             warnings.warn('Halflife estimate inaccurate, time stop DISABLED. R-Squared:' + str('%.2f' % r_sqr))
#             HL = math.inf
        return status,[x_w,y_w],[df_today,(df_today - EMA_today)/std,std,HL],df,df_EMA,adf_res[1],[x_today,y_today]

def track_trade(std_today,abs_today,trade_time,status,args,total_data,trade_loc):
    '''ITERATIVELY TRACK TRADE AND OUTPUTS EXIT SIGNAL, EXIT TYPE'''
    # Expects status in form of [buy=True/False,sell=True/False] and args in form of                                                   [UStop_std,UTP_std,LStop_std,LTP_std,UStop_abs,UTP_abs,LStop_abs,LTP_abs,TS_multipler,HalfLife]
    buy,sell = status
    UStop_std,UTP_std,LStop_std,LTP_std,UStop_abs,UTP_abs,LStop_abs,LTP_abs,TS_mult,hl = args
    args_std = [UStop_std,UTP_std,LStop_std,LTP_std,buy,sell]
    args_abs = [UStop_abs,UTP_abs,LStop_abs,LTP_abs,buy,sell]
    args_ts = [TS_mult,hl]
    assert buy == True or buy == False, 'Input not valid'
    assert sell == True or sell == False, 'Input not valid'
    if buy == False and sell == False:
        raise ValueError('No trade recommended')
    elif buy == True and sell == True:
        raise ValueError('buy, sell can not both equal True')
    def exitcond_std(std_today,args):
        UStop_std,UTP_std,LStop_std,LTP_std,buy,sell = args
        if buy == True and sell == False and std_today < LStop_std:
            exit = True
            extype = 'Stop Loss - Std'
        elif buy == True and sell == False and std_today > UTP_std:
            exit = True
            extype = 'Take Profit - Std'
        elif sell == True and buy == False and std_today > UStop_std:
            exit = True
            extype = 'Stop Loss - Std'
        elif sell == True and buy == False and std_today < LTP_std:
            exit = True
            extype = 'Take Profit - Std'
        else:
            exit = False
            extype = None
        return [exit,extype]
    
    def exitcond_abs(abs_today,args):
        UStop_abs,UTP_abs,LStop_abs,LTP_abs,buy,sell = args
        if buy == True and sell == False and abs_today < LStop_abs:
            exit = True
            extype = 'Stop Loss - Abs'
        elif buy == True and sell == False and abs_today > UTP_abs:
            exit = True
            extype = 'Take Profit - Abs'
        elif sell == True and buy == False and abs_today > UStop_abs:
            exit = True
            extype = 'Stop Loss - Abs'
        elif sell == True and buy == False and abs_today < LTP_abs:
            exit = True
            extype = 'Take Profit - Abs'
        else: 
            exit = False
            extype = None
        return [exit,extype]
    
    def exitcond_ts(trade_time,args):
        TS_mult,hl = args
        if trade_time >= hl * TS_mult: 
            exit = True
            extype = 'Time Stop '
        else:
            exit = False
            extype = None
        return [exit,extype]

#     def weight_stability(weight_change,weight_tolerance):
#         if weight_change > weight_tolerance:
#             exit = True
#             extype = 'Weight Stability'
#         else: 
#             exit = False
#             extype = None
#         return [exit,extype]
    
    def exitcond_nodata(trade_loc,total_data):
        if trade_loc == total_data:
            exit = True
            extype = 'No Data'
        else:
            exit = False
            extype = None
        return [exit,extype]
    return exitcond_std(std_today,args_std),exitcond_abs(abs_today,args_abs),exitcond_ts(trade_time,args_ts),exitcond_nodata(trade_loc,total_data)
    
def back_test_RV(date,x,y,*args,**kwargs):
    '''TEST TRADING STRATEGY AND OUTPUTS RESULTS'''
    # Expects inputs x,y as dataframes and args in form of [Lookback, Entry Z, Stop Loss Z, Take Profit Z]
    # Outputs Entry Price, Exit Price, Entry Z Score, Exit Z Score, PnL (%), PnL (abs), Duration (d), Outcome, Exit Type
    lookback = kwargs.get('com', None)
    entry_z = kwargs.get('entry_z', None)
    stoploss_z = kwargs.get('stoploss_z', None)
    takeprofit_z = kwargs.get('takeprofit_z',None)
    maxentry_z = kwargs.get('maxentry_z',None)
    weight_stability = kwargs.get('weight_stability',False)
    weightthres_1 = kwargs.get('weightthres_1',None)
    weightthres_2 = kwargs.get('weightthres_2',None)
    weight_override = kwargs.get('weight_override',None)
    window = kwargs.get('window',100)
    pca = kwargs.get('pca',False)
    instrument = kwargs.get('instrument',None)
    verbose = kwargs.get('verbose',False)
    if pca == False: 
        print('OLS weight: Enabled by default')
    if instrument == None: 
        instrument = 'future'
        print('No instrument type selected: Default value future used')
    if lookback == None: 
        lookback = 100
        print('No exponential MA input detected: Default value 100 used')
    if entry_z == None:
        entry_z = 1.5
        print('No entry_z input detected: Default value 1.5 used')
    if stoploss_z == None: 
        stoploss_z = 2.5
        print('No stoploss_z input detected: Default value 2.5 used')
    if takeprofit_z == None: 
        takeprofit_z = 0 
        print('No takeprofit_z input detected: Default value 0 used')
    if maxentry_z == None: 
        maxentry_z = math.inf 
        print('No maxentry_z input detected: Default value inf used')
    three_leg = False
    args_iter = 0
    for a in args:
        z = a
        three_leg = True
        args_iter += 1
    assert args_iter < 2, 'Too many inputs'
    if pca == True and three_leg == True: 
        warnings.warn('three_leg == True and pca == True, PCA not supported for three leg')
        pca = False
    elif pca == True:
        print('PCA weight: Enabled')
    if three_leg == True:
        if weightthres_1 == None:
            weightthres_1 = [-math.inf,math.inf]
            print ('Weight threshold 1: Disabled (to enable use weightthres_1 = [lower,upper])')
        if weightthres_2 == None:
            weightthres_2 = [-math.inf,math.inf]
            print ('Weight threshold 2: Disabled (to enable use weightthres_2 = [lower,upper])')
        if weight_stability == False: 
            print('Weight stability: Disabled (to enable use weight_stability = True)')
        weightthres = np.asarray([weightthres_1,weightthres_2])
        if weight_override == None:
            print('Weight override: Disabled (to enable use weight_override = [weight1,weight2,weight3])')
    elif three_leg == False:
        if weightthres_1 == None:
            weightthres_1 = [-math.inf,math.inf]
            print ('Weight threshold 1: Disabled (to enable use weightthres_1 = [lower,upper])')
        if weight_stability == False: 
            print('Weight stability: Disabled (to enable use weight_stability = True)')
        if weight_override == None:
            print('Weight override: Disabled (to enable use weight_override = [weight1,weight2])')
        weightthres = np.asarray([weightthres_1])
    print()
    print('Backtest Period: ' + str(date.iloc[0]) + ' - ' + str(date.iloc[-1]))
    args_gen = [entry_z,-entry_z,maxentry_z,lookback]
    if three_leg == True:
        i,entry_date,exit_date,entry_abs,exit_abs,entry_std,exit_std,extype,PnL_pc,PnL_abs,outcome,time_held,w_ratio_1,w_ratio_2,adf = 0,[],[],[],[],[],[],[],[],[],[],[],[],[],[]
    else: 
        i,entry_date,exit_date,entry_abs,exit_abs,entry_std,exit_std,extype,PnL_pc,PnL_abs,outcome,time_held,w_ratio_1,adf = 0,[],[],[],[],[],[],[],[],[],[],[],[],[]
    notional = 100000
    wind = window
    plt_wind = 30
    dataframeentry_save,emaentry_save,status_save,tpsl_save,tpslstd_save,std_save = [],[],[],[],[],[]
    dataframeexit_save,indexexit_save,entrycondexit_save,dataframeexit2_save,ema_exit = [],[],[],[],[]
    # Divide data into dataframes of 100 in length 
    for i in progressbar(range(30,len(x)-wind), "Progress: ", 40):
        if three_leg == True:
            df = [x[i:i+wind],y[i:i+wind],z[i:i+wind]]
        else:
            df = [x[i:i+wind],y[i:i+wind]]
        status,weights,entry_cond,df_save,ema,adf_res,entry_legabs = generate_trade(df,args_gen,weightthres,three_leg,pca,weight_override) 
        # status [buy=True/False,sell=True/False], weights [x_w,y_w], entry_cond [abs_value,std_value,std,halflife]
        # Trade generated - proceed to track outcome; else next dataframe
        if True in status:
            t = 1
            exit = False
            if three_leg == True: 
                w_ratio_1.append('%.2f' % (weights[1]/weights[0]))
                w_ratio_2.append('%.2f' % (weights[2]/weights[0]))
            else: 
                w_ratio_1.append('%.2f' % (weights[1]/weights[0]))
            entry_date.append(date.iloc[i+wind].replace(',',''))
            entry_abs.append('%.5f' % entry_cond[0])
            entry_std.append('%.2f' % entry_cond[1])
            adf.append('%.4f' % adf_res)
            UStop_std,UTP_std,LStop_std,LTP_std = (entry_cond[1]+stoploss_z),-takeprofit_z,(entry_cond[1]-stoploss_z),takeprofit_z
            UStop_abs,UTP_abs,LStop_abs,LTP_abs = (entry_cond[0]+stoploss_z*entry_cond[2]),(entry_cond[0]+    (abs(entry_cond[1])-takeprofit_z)*entry_cond[2]),(entry_cond[0]-stoploss_z*entry_cond[2]),(entry_cond[0]-(abs(entry_cond[1])-takeprofit_z)*entry_cond[2])
            if verbose == True:
                tpsl_save.append(np.asarray([UStop_abs,UTP_abs,LStop_abs,LTP_abs]))
                tpslstd_save.append(np.asarray([UStop_std,UTP_std,LStop_std,LTP_std]))
                std_save.append(entry_cond[2])
                status_save.append(np.asarray(status))
                emaentry_save.append(np.asarray(ema))
                dataframeentry_save.append(np.asarray(df_save))
            # Maintain entry weights
            while t < len(x[i:]) and exit == False:
                if three_leg == True:
                    spread = weights[0] * x[i+t:i+wind+t] - weights[1] * y[i+t:i+wind+t] - weights[2] * z[i+t:i+wind+t]
                    x_current,y_current,z_current = x[i+t:i+wind+t],y[i+t:i+wind+t],z[i+t:i+wind+t]
                    spread = spread.reset_index(drop=True)
                    spread_save = weights[0] * x[i-plt_wind+wind:i+t+plt_wind+wind] - weights[1] * y[i-plt_wind+wind:i+t+plt_wind+wind] - weights[2] * z[i-plt_wind+wind:i+t+plt_wind+wind]
                    spread_save = spread_save.reset_index(drop=True)
                else:
                    spread = weights[0] * x[i+t:i+wind+t] - weights[1] * y[i+t:i+wind+t]
                    x_current,y_current = x[i+t:i+wind+t],y[i+t:i+wind+t]
                    spread = spread.reset_index(drop=True)
                    spread_save = weights[0] * x[i-plt_wind+wind:i+t+plt_wind+wind] - weights[1] * y[i-plt_wind+wind:i+t+plt_wind+wind]
                    spread_save = spread_save.reset_index(drop=True)
                df_date = date.iloc[i+t:i+t+wind]
                EMA,std = EMA_STD(spread,args_gen[3])
                args_track = [UStop_std,UTP_std,LStop_std,LTP_std,UStop_abs,UTP_abs,LStop_abs,LTP_abs,1,entry_cond[3]]
                # Set tracking arguments
                st_dev = (spread.iloc[-1] - EMA.iloc[-1])/entry_cond[2]
                status_std,status_abs,status_ts,status_nodata = track_trade(st_dev,spread.iloc[-1],t,status,args_track,len(date)-1,i+t)
                # status_std [Exit=True/False,extype=None/...], status_abs [Exit=True/False,extype=None/...], status_ts                             [Exit=True/False,extype=None/...]
                # Check for exit signals and record results
                status_weight = [False,None]
                if True in status_std:
                    exit = status_std[0]
                    extype.append(status_std[1])
                    exit_date.append(df_date.iloc[-1].replace(',',''))
                    time_held.append(t+1)
                    if verbose == True: 
                        dataframeexit_save.append(np.asarray(spread_save))
                        dataframeexit2_save.append(np.asarray(spread.iloc[-100:]))
                        ema,_ = EMA_STD(spread,lookback)
                        ema_exit.append(np.asarray(ema))
                        indexexit_save.append(t)
                        entrycondexit_save.append(np.asarray([entry_cond[0],spread.iloc[-1]]))
                    if status[0] == True:
                        PnL_pc.append('%.3f' % ((spread.iloc[-1] - entry_cond[0])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (- z_current.iloc[-1] + entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (spread.iloc[-1] - entry_cond[0]))
                        if spread.iloc[-1] - entry_cond[0] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('win')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('loss')
                    elif status[1] == True:
                        PnL_pc.append('%.3f' % ((entry_cond[0] - spread.iloc[-1])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (z_current.iloc[-1] - entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (entry_cond[0] - spread.iloc[-1]))
                        if entry_cond[0] - spread.iloc[-1] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('win')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('loss')
                    break 
                elif True in status_abs:
                    exit = status_abs[0]
                    extype.append(status_abs[1])
                    exit_date.append(df_date.iloc[-1].replace(',',''))
                    time_held.append(t+1)
                    if verbose == True:
                        dataframeexit_save.append(np.asarray(spread_save))
                        dataframeexit2_save.append(np.asarray(spread.iloc[-100:]))
                        ema,_ = EMA_STD(spread,lookback)
                        ema_exit.append(np.asarray(ema))
                        indexexit_save.append(t)
                        entrycondexit_save.append(np.asarray([entry_cond[0],spread.iloc[-1]]))
                    if status[0] == True:
                        PnL_pc.append('%.3f' % ((spread.iloc[-1] - entry_cond[0])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (- z_current.iloc[-1] + entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (spread.iloc[-1] - entry_cond[0]))
                        if spread.iloc[-1] - entry_cond[0] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('win')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('loss')
                    elif status[1] == True:
                        PnL_pc.append('%.3f' % ((entry_cond[0] - spread.iloc[-1])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (z_current.iloc[-1] - entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (entry_cond[0] - spread.iloc[-1]))
                        if entry_cond[0] - spread.iloc[-1] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('win')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('loss')
                    break
                elif True in status_ts:
                    exit = status_ts[0]
                    extype.append(status_ts[1])
                    exit_date.append(df_date.iloc[-1].replace(',',''))
                    time_held.append(t+1)
                    if verbose == True:
                        dataframeexit_save.append(np.asarray(spread_save))
                        dataframeexit2_save.append(np.asarray(spread.iloc[-100:]))
                        ema,_ = EMA_STD(spread,lookback)
                        ema_exit.append(np.asarray(ema))
                        indexexit_save.append(t)
                        entrycondexit_save.append(np.asarray([entry_cond[0],spread.iloc[-1]]))
                    if status[0] == True:
                        PnL_pc.append('%.3f' % ((spread.iloc[-1] - entry_cond[0])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (- z_current.iloc[-1] + entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (spread.iloc[-1] - entry_cond[0]))
                        if spread.iloc[-1] - entry_cond[0] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('win')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('loss')
                    elif status[1] == True:
                        PnL_pc.append('%.3f' % ((entry_cond[0] - spread.iloc[-1])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (z_current.iloc[-1] - entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (entry_cond[0] - spread.iloc[-1]))
                        if entry_cond[0] - spread.iloc[-1] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('win')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('loss')
                    break
                elif True in status_weight:
                    exit = status_weight[0]
                    extype.append(status_weight[1])
                    exit_date.append(df_date.iloc[-1].replace(',',''))
                    time_held.append(t+1)
                    if verbose == True:
                        dataframeexit_save.append(np.asarray(spread_save))
                        dataframeexit2_save.append(np.asarray(spread.iloc[-100:]))
                        ema,_ = EMA_STD(spread,lookback)
                        ema_exit.append(np.asarray(ema))
                        indexexit_save.append(t)
                        entrycondexit_save.append(np.asarray([entry_cond[0],spread.iloc[-1]]))
                    if status[0] == True:
                        PnL_pc.append('%.3f' % ((spread.iloc[-1] - entry_cond[0])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (- z_current.iloc[-1] + entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (spread.iloc[-1] - entry_cond[0]))
                        if spread.iloc[-1] - entry_cond[0] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('win')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('loss')
                    elif status[1] == True:
                        PnL_pc.append('%.3f' % ((entry_cond[0] - spread.iloc[-1])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (z_current.iloc[-1] - entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (entry_cond[0] - spread.iloc[-1]))
                        if entry_cond[0] - spread.iloc[-1] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('win')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('loss')
                    break
                elif True in status_nodata:
                    exit = status_nodata[0]
                    extype.append(status_nodata[1])
                    exit_date.append(df_date.iloc[-1].replace(',',''))
                    time_held.append(t+1)
                    if verbose == True: 
                        dataframeexit_save.append(np.asarray(spread_save))
                        dataframeexit2_save.append(np.asarray(spread.iloc[-100:]))
                        ema,_ = EMA_STD(spread,lookback)
                        ema_exit.append(np.asarray(ema))
                        indexexit_save.append(t)
                        entrycondexit_save.append(np.asarray([entry_cond[0],spread.iloc[-1]]))
                    if status[0] == True:
                        PnL_pc.append('%.3f' % ((spread.iloc[-1] - entry_cond[0])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (- z_current.iloc[-1] + entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (x_current.iloc[-1] - entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (- y_current.iloc[-1] + entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (spread.iloc[-1] - entry_cond[0]))
                        if spread.iloc[-1] - entry_cond[0] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('N/A')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('N/A')
                    elif status[1] == True:
                        PnL_pc.append('%.3f' % ((entry_cond[0] - spread.iloc[-1])*100/abs(entry_cond[0])))
                        if instrument == 'future':
                            if three_leg == True:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_leg3 = (z_current.iloc[-1] - entry_legabs[2])*notional*weights[2]/entry_legabs[2]
                                PnL_sum = PnL_leg1 + PnL_leg2 + PnL_leg3
                                PnL_abs.append('%.5f' % PnL_sum)
                            else:
                                PnL_leg1 = (- x_current.iloc[-1] + entry_legabs[0])*notional*weights[0]/entry_legabs[0]
                                PnL_leg2 = (y_current.iloc[-1] - entry_legabs[1])*notional*weights[1]/entry_legabs[1]
                                PnL_sum = PnL_leg1 + PnL_leg2
                                PnL_abs.append('%.5f' % PnL_sum)
                        elif instrument == 'swap':
                            PnL_abs.append('%.5f' % (entry_cond[0] - spread.iloc[-1]))
                        if entry_cond[0] - spread.iloc[-1] > 0:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('N/A')
                        else:
                            exit_abs.append('%.5f' % spread.iloc[-1])
                            exit_std.append('%.2f' % st_dev)
                            outcome.append('N/A')
                    break
                else:
                    exit = False
                t += 1
        i += 1 
    print()
    print('Backtest Complete - See Statistics Below')
    trade_stats(PnL_pc,PnL_abs,time_held,outcome,extype,args,instrument)
    if three_leg == True:    
        oup = np.column_stack((entry_date,exit_date,entry_abs,exit_abs,entry_std,exit_std,time_held,w_ratio_1,w_ratio_2,adf,PnL_pc,PnL_abs,outcome,extype))
    else: 
        oup = np.column_stack((entry_date,exit_date,entry_abs,exit_abs,entry_std,exit_std,time_held,w_ratio_1,adf,PnL_pc,PnL_abs,outcome,extype))
    writedata(oup,three_leg)
    if three_leg == True:
        df = [x[-1-wind:-1],y[-1-wind:-1],z[-1-wind:-1]]
    else:
        df = [x[-1-wind:-1],y[-1-wind:-1]]
    status,weights,entry_cond,df_save,ema,adf_res,_ = generate_trade(df,args_gen,weightthres,three_leg,pca,weight_override)
    if True in status:
        plt.plot(df_save,label='Spread')
        plt.plot(ema,label='Exp MA')
        plt.legend(loc='best',prop={'size': 8})
        plt.xlabel('Days')
        plt.ylabel('Spread')
        plt.title('Trade Recommendation')
        plt.show()
        if status[0] == True:
            if three_leg == True:
                print('Recommendation: Buy, Z-Score: ' + str(round(entry_cond[1],2)) + ', Leg1 Weight: ' + str(1.00) + ', Leg2 Weight: ' + str(round(weights[1]/weights[0],2)) + ', Leg3 Weight: ' + str(round(weights[2]/weights[0],2)) + ', ADF: ' + str(round(adf_res,4)))
            else: 
                print('Recommendation: Buy, Z-Score: ' + str(round(entry_cond[1],2)) + ', Leg1 Weight: ' + str(1.00) + ', Leg2 Weight: ' + str(round(weights[1]/weights[0],2)) + ', ADF: ' + str(round(adf_res,4)))
        else: 
            if three_leg == True:
                print('Recommendation: Sell, Z-Score: ' + str(round(entry_cond[1],2)) + ', Leg1 Weight: ' + str(1.00) + ', Leg2 Weight: ' + str(round(weights[1]/weights[0],2)) + ', Leg3 Weight: ' + str(round(weights[2]/weights[0],2)) + ', ADF: ' + str(round(adf_res,4)))
            else: 
                print('Recommendation: Sell, Z-Score: ' + str(round(entry_cond[1],2)) + ', Leg1 Weight: ' + str(1.00) + ', Leg2 Weight: ' + str(round(weights[1]/weights[0],2)) + ', ADF: ' + str(round(adf_res,4)))
    trade_analytics(oup,instrument)
    if verbose == True: 
        
        plot_multiple(dataframeexit_save,entrycondexit_save,indexexit_save,dataframeentry_save,emaentry_save,status_save,tpsl_save,tpslstd_save,outcome,extype,dataframeexit2_save,ema_exit,std_save,wind)
    return oup
    
def trade_stats(PnL_pc,PnL_abs,time_held,outcome,extype,args,instrument):
    '''PRINTS TRADE STATISTICS'''
    total_num = len(PnL_pc) - outcome.count('N/A')
    if instrument == 'future':
        if total_num > 0:
            win_ratio = '{:.2%}'.format(outcome.count('win')/total_num)
            avg_time = '{:.2f}'.format(sum(time_held)/len(time_held))
            max_drawdown = '{:.2f}'.format(min(map(float, PnL_abs)))
            cum_pnl = '{:.2f}'.format(sum(map(float,PnL_abs)))
            avg_pnl = '{:.2f}'.format(sum(map(float,PnL_abs))/total_num)
            print('Total Number of Trades: ' + str(total_num))
            print('Win Rate: ' + str(win_ratio))
            print('Total PnL USD: ' + str(cum_pnl))
            print('Average PnL per Trade USD: ' + str(avg_pnl))
            print('Max Drawdown in Single Trade USD: ' + str(max_drawdown))
            print('Average Time of Trade: ' + str(avg_time) + ' Trading Days')
        else:
            print('No Trades Completed')
    else:
        if total_num > 0:
            win_ratio = '{:.2%}'.format(outcome.count('win')/total_num)
            avg_time = '{:.2f}'.format(sum(time_held)/len(time_held))
            max_drawdown = '{:.2f}'.format(min(map(float, PnL_abs))*100)
            cum_pnl = '{:.2f}'.format(sum(map(float,PnL_abs))*100)
            avg_pnl = '{:.2f}'.format(sum(map(float,PnL_abs))*100/total_num)
            print('Total Number of Trades: ' + str(total_num))
            print('Win Rate: ' + str(win_ratio))
            print('Total PnL bps: ' + str(cum_pnl))
            print('Average PnL per Trade bps: ' + str(avg_pnl))
            print('Max Drawdown in Single Trade bps: ' + str(max_drawdown))
            print('Average Time of Trade: ' + str(avg_time) + ' Trading Days')
        else:
            print('No Trades Completed')

def trade_analytics(x,instrument):
    if instrument == 'future':
        if x.shape[1] == 14:
            total_num_2020 = 0
            win_num_2020 = 0
            total_num_2019 = 0
            win_num_2019 = 0
            total_num_2018 = 0
            win_num_2018 = 0
            total_num_2017 = 0
            win_num_2017 = 0
            total_num_2016 = 0
            win_num_2016 = 0
            total_num_2015 = 0
            win_num_2015 = 0
            for i in np.arange(x.shape[0]):
                if '2020' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2020 += 1
                    else: 
                        total_num_2020 = total_num_2020
                    if x[i,12] == 'win':
                        win_num_2020 =+ 1
                    else:
                        win_num_2020 = win_num_2020
                elif '2019' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2019 += 1
                    else: 
                        total_num_2019 = total_num_2019
                    if x[i,12] == 'win':
                        win_num_2019 += 1
                    else:
                        win_num_2019 = win_num_2019
                elif '2018' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2018 += 1
                    else: 
                        total_num_2018 = total_num_2018
                    if x[i,12] == 'win':
                        win_num_2018 += 1
                    else:
                        win_num_2018 = win_num_2018
                elif '2017' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2017 += 1
                    else: 
                        total_num_2017 = total_num_2017
                    if x[i,12] == 'win':
                        win_num_2017 += 1
                    else:
                        win_num_2017 = win_num_2017
                elif '2016' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2016 += 1
                    else: 
                        total_num_2016 = total_num_2016
                    if x[i,12] == 'win':
                        win_num_2016 += 1
                    else:
                        win_num_2016 = win_num_2016
                elif '2015' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2015 += 1
                    else: 
                        total_num_2015 = total_num_2015
                    if x[i,12] == 'win':
                        win_num_2015 += 1
                    else:
                        win_num_2015 = win_num_2015
            try:
                win_rate_2015 = win_num_2015/total_num_2015
            except: 
                win_rate_2015 = 'no trade'
            try:
                win_rate_2016 = win_num_2016/total_num_2016
            except:
                win_rate_2016 = 'no trade'
            try:
                win_rate_2017 = win_num_2017/total_num_2017
            except: 
                win_rate_2017 = 'no trade'
            try:
                win_rate_2018 = win_num_2018/total_num_2018
            except:
                win_rate_2018 = 'no trade'
            try:
                win_rate_2019 = win_num_2019/total_num_2019
            except:
                win_rate_2019 = 'no trade'
            try:
                win_rate_2020 = win_num_2020/total_num_2020
            except:
                win_rate_2020 = 'no trade'
            plt.figure(figsize=(12.8,3.8))
            plt.subplot(1,3,1)
            sns.kdeplot(sorted(x[:,11]))
            plt.xlabel('USD PnL')
            plt.ylabel('Probability Density')
            plt.title('Probability Density Distribution of PnL')

            plt.subplot(1,3,2)
            plt.bar(['2015','2016','2017','2018','2019','2020'],[win_rate_2015,win_rate_2016,win_rate_2017,win_rate_2018,win_rate_2019,win_rate_2020])
            plt.xlabel('Year')
            plt.ylabel('Win Rate %')
            plt.title('Strategy Win Rate by Year')

            plt.subplot(1,3,3)
            plt.bar(['2015','2016','2017','2018','2019','2020'],[total_num_2015,total_num_2016,total_num_2017,total_num_2018,total_num_2019,total_num_2020])
            plt.xlabel('Year')
            plt.ylabel('Total Number of Trades')
            plt.title('Total Number of Trades by Year')
            plt.tight_layout()
            plt.show()
        elif x.shape[1] == 13:
            total_num_2020 = 0
            win_num_2020 = 0
            total_num_2019 = 0
            win_num_2019 = 0
            total_num_2018 = 0
            win_num_2018 = 0
            total_num_2017 = 0
            win_num_2017 = 0
            total_num_2016 = 0
            win_num_2016 = 0
            total_num_2015 = 0
            win_num_2015 = 0
            for i in np.arange(x.shape[0]):
                if '2020' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2020 += 1
                    else: 
                        total_num_2020 = total_num_2020
                    if x[i,11] == 'win':
                        win_num_2020 += 1
                    else:
                        win_num_2020 = win_num_2020
                elif '2019' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2019 += 1
                    else: 
                        total_num_2019 = total_num_2019
                    if x[i,11] == 'win':
                        win_num_2019 += 1
                    else:
                        win_num_2019 = win_num_2019
                elif '2018' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2018 += 1
                    else: 
                        total_num_2018 = total_num_2018
                    if x[i,11] == 'win':
                        win_num_2018 += 1
                    else:
                        win_num_2018 = win_num_2018
                elif '2017' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2017 += 1
                    else: 
                        total_num_2017 = total_num_2017
                    if x[i,11] == 'win':
                        win_num_2017 += 1
                    else:
                        win_num_2017 = win_num_2017
                elif '2016' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2016 += 1
                    else: 
                        total_num_2016 = total_num_2016
                    if x[i,11] == 'win':
                        win_num_2016 += 1
                    else:
                        win_num_2016 = win_num_2016
                elif '2015' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2015 += 1
                    else: 
                        total_num_2015 = total_num_2015
                    if x[i,11] == 'win':
                        win_num_2015 += 1
                    else:
                        win_num_2015 = win_num_2015
            try:
                win_rate_2015 = win_num_2015/total_num_2015
            except: 
                win_rate_2015 = 0
            try:
                win_rate_2016 = win_num_2016/total_num_2016
            except:
                win_rate_2016 = 0
            try:
                win_rate_2017 = win_num_2017/total_num_2017
            except: 
                win_rate_2017 = 0
            try:
                win_rate_2018 = win_num_2018/total_num_2018
            except:
                win_rate_2018 = 0
            try:
                win_rate_2019 = win_num_2019/total_num_2019
            except:
                win_rate_2019 = 0
            try:
                win_rate_2020 = win_num_2020/total_num_2020
            except:
                win_rate_2020 = 0
            plt.figure(figsize=(12.8,3.8))
            plt.subplot(1,3,1)
            sns.kdeplot(sorted(x[:,10]))
            plt.xlabel('USD PnL')
            plt.ylabel('Probability Density')
            plt.title('Probability Density Distribution of PnL')

            plt.subplot(1,3,2)
            plt.bar(['2015','2016','2017','2018','2019','2020'],[win_rate_2015,win_rate_2016,win_rate_2017,win_rate_2018,win_rate_2019,win_rate_2020])
            plt.xlabel('Year')
            plt.ylabel('Win Rate %')
            plt.title('Strategy Win Rate by Year')

            plt.subplot(1,3,3)
            plt.bar(['2015','2016','2017','2018','2019','2020'],[total_num_2015,total_num_2016,total_num_2017,total_num_2018,total_num_2019,total_num_2020])
            plt.xlabel('Year')
            plt.ylabel('Total Number of Trades')
            plt.title('Total Number of Trades by Year')
            plt.tight_layout()
            plt.show()
    elif instrument == 'swap':
        if x.shape[1] == 14:
            total_num_2020 = 0
            win_num_2020 = 0
            total_num_2019 = 0
            win_num_2019 = 0
            total_num_2018 = 0
            win_num_2018 = 0
            total_num_2017 = 0
            win_num_2017 = 0
            total_num_2016 = 0
            win_num_2016 = 0
            total_num_2015 = 0
            win_num_2015 = 0
            for i in np.arange(x.shape[0]):
                if '2020' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2020 += 1
                    else: 
                        total_num_2020 = total_num_2020
                    if x[i,12] == 'win':
                        win_num_2020 =+ 1
                    else:
                        win_num_2020 = win_num_2020
                elif '2019' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2019 += 1
                    else: 
                        total_num_2019 = total_num_2019
                    if x[i,12] == 'win':
                        win_num_2019 += 1
                    else:
                        win_num_2019 = win_num_2019
                elif '2018' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2018 += 1
                    else: 
                        total_num_2018 = total_num_2018
                    if x[i,12] == 'win':
                        win_num_2018 += 1
                    else:
                        win_num_2018 = win_num_2018
                elif '2017' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2017 += 1
                    else: 
                        total_num_2017 = total_num_2017
                    if x[i,12] == 'win':
                        win_num_2017 += 1
                    else:
                        win_num_2017 = win_num_2017
                elif '2016' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2016 += 1
                    else: 
                        total_num_2016 = total_num_2016
                    if x[i,12] == 'win':
                        win_num_2016 += 1
                    else:
                        win_num_2016 = win_num_2016
                elif '2015' in x[i,0]:
                    if x[i,12] == 'win' or x[i,12] == 'loss':
                        total_num_2015 += 1
                    else: 
                        total_num_2015 = total_num_2015
                    if x[i,12] == 'win':
                        win_num_2015 += 1
                    else:
                        win_num_2015 = win_num_2015
            try:
                win_rate_2015 = win_num_2015/total_num_2015
            except: 
                win_rate_2015 = 'no trade'
            try:
                win_rate_2016 = win_num_2016/total_num_2016
            except:
                win_rate_2016 = 'no trade'
            try:
                win_rate_2017 = win_num_2017/total_num_2017
            except: 
                win_rate_2017 = 'no trade'
            try:
                win_rate_2018 = win_num_2018/total_num_2018
            except:
                win_rate_2018 = 'no trade'
            try:
                win_rate_2019 = win_num_2019/total_num_2019
            except:
                win_rate_2019 = 'no trade'
            try:
                win_rate_2020 = win_num_2020/total_num_2020
            except:
                win_rate_2020 = 'no trade'
            plt.figure(figsize=(12.8,3.8))
            plt.subplot(1,3,1)
            sns.kdeplot(sorted(x[:,11]))
            plt.xlabel('Bps PnL')
            plt.ylabel('Probability Density')
            plt.title('Probability Density Distribution of PnL')

            plt.subplot(1,3,2)
            plt.bar(['2015','2016','2017','2018','2019','2020'],[win_rate_2015,win_rate_2016,win_rate_2017,win_rate_2018,win_rate_2019,win_rate_2020])
            plt.xlabel('Year')
            plt.ylabel('Win Rate %')
            plt.title('Strategy Win Rate by Year')

            plt.subplot(1,3,3)
            plt.bar(['2015','2016','2017','2018','2019','2020'],[total_num_2015,total_num_2016,total_num_2017,total_num_2018,total_num_2019,total_num_2020])
            plt.xlabel('Year')
            plt.ylabel('Total Number of Trades')
            plt.title('Total Number of Trades by Year')
            plt.tight_layout()
            plt.show()
        elif x.shape[1] == 13:
            total_num_2020 = 0
            win_num_2020 = 0
            total_num_2019 = 0
            win_num_2019 = 0
            total_num_2018 = 0
            win_num_2018 = 0
            total_num_2017 = 0
            win_num_2017 = 0
            total_num_2016 = 0
            win_num_2016 = 0
            total_num_2015 = 0
            win_num_2015 = 0
            for i in np.arange(x.shape[0]):
                if '2020' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2020 += 1
                    else: 
                        total_num_2020 = total_num_2020
                    if x[i,11] == 'win':
                        win_num_2020 += 1
                    else:
                        win_num_2020 = win_num_2020
                elif '2019' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2019 += 1
                    else: 
                        total_num_2019 = total_num_2019
                    if x[i,11] == 'win':
                        win_num_2019 += 1
                    else:
                        win_num_2019 = win_num_2019
                elif '2018' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2018 += 1
                    else: 
                        total_num_2018 = total_num_2018
                    if x[i,11] == 'win':
                        win_num_2018 += 1
                    else:
                        win_num_2018 = win_num_2018
                elif '2017' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2017 += 1
                    else: 
                        total_num_2017 = total_num_2017
                    if x[i,11] == 'win':
                        win_num_2017 += 1
                    else:
                        win_num_2017 = win_num_2017
                elif '2016' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2016 += 1
                    else: 
                        total_num_2016 = total_num_2016
                    if x[i,11] == 'win':
                        win_num_2016 += 1
                    else:
                        win_num_2016 = win_num_2016
                elif '2015' in x[i,0]:
                    if x[i,11] == 'win' or x[i,11] == 'loss':
                        total_num_2015 += 1
                    else: 
                        total_num_2015 = total_num_2015
                    if x[i,11] == 'win':
                        win_num_2015 += 1
                    else:
                        win_num_2015 = win_num_2015
            try:
                win_rate_2015 = win_num_2015/total_num_2015
            except: 
                win_rate_2015 = 0
            try:
                win_rate_2016 = win_num_2016/total_num_2016
            except:
                win_rate_2016 = 0
            try:
                win_rate_2017 = win_num_2017/total_num_2017
            except: 
                win_rate_2017 = 0
            try:
                win_rate_2018 = win_num_2018/total_num_2018
            except:
                win_rate_2018 = 0
            try:
                win_rate_2019 = win_num_2019/total_num_2019
            except:
                win_rate_2019 = 0
            try:
                win_rate_2020 = win_num_2020/total_num_2020
            except:
                win_rate_2020 = 0
            plt.figure(figsize=(12.8,3.8))
            plt.subplot(1,3,1)
            sns.kdeplot(sorted(x[:,10]))
            plt.xlabel('Bps PnL')
            plt.ylabel('Probability Density')
            plt.title('Probability Density Distribution of PnL')

            plt.subplot(1,3,2)
            plt.bar(['2015','2016','2017','2018','2019','2020'],[win_rate_2015,win_rate_2016,win_rate_2017,win_rate_2018,win_rate_2019,win_rate_2020])
            plt.xlabel('Year')
            plt.ylabel('Win Rate %')
            plt.ylim([0,1])
            plt.title('Strategy Win Rate by Year')

            plt.subplot(1,3,3)
            plt.bar(['2015','2016','2017','2018','2019','2020'],[total_num_2015,total_num_2016,total_num_2017,total_num_2018,total_num_2019,total_num_2020])
            plt.xlabel('Year')
            plt.ylabel('Total Number of Trades')
            plt.title('Total Number of Trades by Year')
            plt.tight_layout()
            plt.show()
        