import pandas as pd
import numpy as np
import math
import warnings
from utils import progressbar,readdata,writedata,pca_2,pca_3,EMA_STD,half_life

def generate_trade(df,args,three_leg):
    '''GENERATE TRADE RECOMMENDATION FROM DATAFRAME'''
    # Expects df as dataframe, args in form of [Upper Z Score Entry, Lower Z Score Entry], three_leg = True/False
    UTH,LTH,lookback = args
    def cond_trade(df_today,EMA_today,std):
        std_dev = (df_today - EMA_today)/std
        if std_dev > UTH:
            sell = True
            buy = False
        elif std_dev < LTH:
            buy = True
            sell = False
        else:
            buy = False
            sell = False
        return [buy,sell]
    if three_leg == True: 
        x,y,z = df
        df,x_w,y_w,z_w = pca_3(x,y,z)
        df_EMA,std = EMA_STD(df,lookback)
        df_today,EMA_today = df.iloc[-1],df_EMA.iloc[-1]
        HL,r_sqr = half_life(df_EMA - df)
        status = cond_trade(df_today,EMA_today,std)
        if r_sqr < 0.01 and True in status:
#             warnings.warn('Halflife estimate inaccurate, time stop DISABLED. R-Squared:' + str('%.2f' % r_sqr))
            HL = 50
        return status,[x_w,y_w,z_w],[df_today,(df_today - EMA_today)/std,std,HL]
    else:
        x,y = df
        df,x_w,y_w = pca_2(x,y)
        df_EMA,std = EMA_STD(df,lookback)
        df_today,EMA_today = df.iloc[-1],df_EMA.iloc[-1]
        HL,r_sqr = half_life(df_EMA - df)
        status = cond_trade(df_today,EMA_today,std)
        if r_sqr < 0.01 and True in status:
#             warnings.warn('Halflife estimate inaccurate, time stop DISABLED. R-Squared:' + str('%.2f' % r_sqr))
            HL = 50
        return status,[x_w,y_w],[df_today,(df_today - EMA_today)/std,std,HL]

def track_trade(std_today,abs_today,trade_time,status,args):
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
            extype = 'Stop Loss - Time'
        else:
            exit = False
            extype = None
        return [exit,extype]
    return exitcond_std(std_today,args_std),exitcond_abs(abs_today,args_abs),exitcond_ts(trade_time,args_ts)
    
def back_test(date,x,y,*args,**kwargs):
    '''TEST TRADING STRATEGY AND OUTPUTS RESULTS'''
    # Expects inputs x,y as dataframes and args in form of [Lookback, Entry Z, Stop Loss Z, Take Profit Z]
    # Outputs Entry Price, Exit Price, Entry Z Score, Exit Z Score, PnL (%), PnL (abs), Duration (d), Outcome, Exit Type
    lookback = kwargs.get('lookback', None)
    entry_z = kwargs.get('entry_z', None)
    stoploss_z = kwargs.get('stoploss_z', None)
    takeprofit_z = kwargs.get('takeprofit_z',None)
    if lookback == None: 
        lookback = 100
        print('No lookback input detected: Default value 100 used')
    if entry_z == None:
        entry_z = 1.5
        print('No entry_z input detected: Default value 1.5 used')
    if stoploss_z == None: 
        stoploss_z = 2.5
        print('No stoploss_z input detected: Default value 2.5 used')
    if takeprofit_z == None: 
        takeprofit_z = 0 
        print('No takeprofit_z input detected: Default value 0 used')
    three_leg = False
    args_iter = 0
    for a in args:
        z = a
        three_leg = True
        args_iter += 1
        print('Mode: three leg')
    assert args_iter < 2, 'Too many inputs'
    if three_leg == False:
        print('Mode: two leg')
    print('Backtest Period: ' + str(date.iloc[0]) + ' - ' + str(date.iloc[-1]))
    args_gen = [entry_z,-entry_z,lookback]
    i,entry_date,exit_date,entry_abs,exit_abs,entry_std,exit_std,extype,PnL_pc,PnL_abs,outcome,time_held = 0,[],[],[],[],[],[],[],[],[],[],[]
    wind = 100
    # Divide data into dataframes of 100 in length 
    for i in progressbar(range(len(x)-wind), "Progress: ", 40):
        if three_leg == True:
            df = [x[i:i+wind],y[i:i+wind],z[i:i+wind]]
        else:
            df = [x[i:i+wind],y[i:i+wind]]
        status,weights,entry_cond = generate_trade(df,args_gen,three_leg) 
        # status [buy=True/False,sell=True/False], weights [x_w,y_w], entry_cond [abs_value,std_value,std,halflife]
        # Trade generated - proceed to track outcome; else next dataframe
        if True in status:
            t = 1
            exit = False
            entry_date.append(date.iloc[i+wind].replace(',',''))
            entry_abs.append('%.5f' % entry_cond[0])
            entry_std.append('%.2f' % entry_cond[1])
            UStop_std,UTP_std,LStop_std,LTP_std = stoploss_z,-takeprofit_z,-stoploss_z,takeprofit_z
            UStop_abs,UTP_abs,LStop_abs,LTP_abs = (entry_cond[0]+abs(UStop_std*entry_cond[2])),(entry_cond[0]+                                                                     abs(entry_cond[1]*entry_cond[2])),(entry_cond[0]-abs(LStop_std*entry_cond[2])),                                                   (entry_cond[0]-abs(entry_cond[1]*entry_cond[2]))
            # Maintain entry weights
            while t < len(x[i:]-100) and exit == False:
                if three_leg == True: 
                    spread = weights[0] * x[i+t:i+wind+t] - weights[1] * y[i+t:i+wind+t] - weights[2] * z[i+t:i+wind+t]
                else:
                    spread = weights[0] * x[i+t:i+wind+t] - weights[1] * y[i+t:i+wind+t]
                df_date = date[i+t:i+wind+t]
                EMA,std = EMA_STD(spread,args_gen[2])
                args_track = [UStop_std,UTP_std,LStop_std,LTP_std,UStop_abs,UTP_abs,LStop_abs,LTP_abs,4,entry_cond[3]]
                # Set tracking arguments
                st_dev = (spread.iloc[-1] - EMA.iloc[-1])/std
                status_std,status_abs,status_ts = track_trade((spread.iloc[-1] -                                                                                                   EMA.iloc[-1])/std,spread.iloc[-1],t,status,args_track)
                # status_std [Exit=True/False,extype=None/...], status_abs [Exit=True/False,extype=None/...], status_ts                             [Exit=True/False,extype=None/...]
                # Check for exit signals and record results
                if True in status_std:
                    exit = status_std[0]
                    extype.append(status_std[1])
                    exit_date.append(df_date.iloc[-1].replace(',',''))
                    time_held.append(t+1)
                    if status[0] == True:
                        PnL_pc.append('%.3f' % ((spread.iloc[-1] - entry_cond[0])*100/abs(entry_cond[0])))
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
                    if status[0] == True:
                        PnL_pc.append('%.3f' % ((spread.iloc[-1] - entry_cond[0])*100/abs(entry_cond[0])))
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
                    if status[0] == True:
                        PnL_pc.append('%.3f' % ((spread.iloc[-1] - entry_cond[0])*100/abs(entry_cond[0])))
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
                else:
                    exit = False
                t += 1
        i += 1
    print('Backtest Complete - See Statistics Below')
    trade_stats(PnL_pc,PnL_abs,time_held,outcome,extype,args)
    oup = np.column_stack((entry_date,exit_date,entry_abs,exit_abs,entry_std,exit_std,time_held,PnL_pc,PnL_abs,outcome,extype))
    writedata(oup)
    return oup

def trade_stats(PnL_pc,PnL_abs,time_held,outcome,extype,args):
    '''PRINTS TRADE STATISTICS'''
    total_num = len(PnL_pc)
    win_ratio = '{:.2%}'.format(outcome.count('win')/total_num)
    avg_time = '{:.2f}'.format(sum(time_held)/len(time_held))
    max_drawdown = '{:.2%}'.format(min(map(float, PnL_pc))/100)
    cum_pnl = '{:.2%}'.format(sum(map(float,PnL_pc))/100)
    print('Total Number of Trades: ' + str(total_num))
    print('Win Ratio: ' + str(win_ratio))
    print('Total % PnL: ' + str(cum_pnl))
    print('Average Time of Trade: ' + str(avg_time) + ' Trading Days')
    print('Max Drawdown in Single Trade: ' + str(max_drawdown))
    
