def back_test_RV(date,x,y,*args,**kwargs):
    '''TEST TRADING STRATEGY AND OUTPUTS RESULTS'''
    # Expects inputs x,y as dataframes and args in form of [Lookback, Entry Z, Stop Loss Z, Take Profit Z]
    # Outputs Entry Price, Exit Price, Entry Z Score, Exit Z Score, PnL (%), PnL (abs), Duration (d), Outcome, Exit Type
    here = False
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
    if three_leg == True:
        df = [x[-1-wind:-1],y[-1-wind:-1],z[-1-wind:-1]]
    else:
        df = [x[-1-wind:-1],y[-1-wind:-1]]
    status,weights,entry_cond,df_save,ema,adf_res,_ = generate_trade(df,args_gen,weightthres,three_leg,pca,weight_override)
    if True in status:
        here = True
        plt.plot(df_save,label='Spread')
        plt.plot(ema,label='Exp MA')
        plt.legend(loc='best',prop={'size': 8})
        plt.xlabel('Days')
        plt.ylabel('Spread')
        plt.title('Trade Recommendation')
        plt.show()
        UStop_std,UTP_std,LStop_std,LTP_std = (entry_cond[1]+stoploss_z),-takeprofit_z,(entry_cond[1]-stoploss_z),takeprofit_z
        UStop_abs,UTP_abs,LStop_abs,LTP_abs = (entry_cond[0]+stoploss_z*entry_cond[2]),(entry_cond[0]+    (abs(entry_cond[1])-takeprofit_z)*entry_cond[2]),(entry_cond[0]-stoploss_z*entry_cond[2]),(entry_cond[0]-(abs(entry_cond[1])-takeprofit_z)*entry_cond[2])

        if status[0] == True:
            if three_leg == True:
                print('Recommendation: Buy, Z-Score: ' + str(round(entry_cond[1],2)) + ', Leg1 Weight: ' + str(1.00) + ', Leg2 Weight: ' + str(round(weights[1]/weights[0],2)) + ', Leg3 Weight: ' + str(round(weights[2]/weights[0],2)) + ', ADF: ' + str(round(adf_res,4)) + ', Time Stop: ' + str(entry_cond[3]) + ', Max PnL: ' + str(round(abs(entry_cond[0] - UTP_abs),4)))
            else: 
                print('Recommendation: Buy, Z-Score: ' + str(round(entry_cond[1],2)) + ', Leg1 Weight: ' + str(1.00) + ', Leg2 Weight: ' + str(round(weights[1]/weights[0],2)) + ', ADF: ' + str(round(adf_res,4)) + ', Time Stop: ' + str(entry_cond[3]) + ', Max PnL: ' + str(round(abs(entry_cond[0] - UTP_abs),4)))
        else: 
            if three_leg == True:
                print('Recommendation: Sell, Z-Score: ' + str(round(entry_cond[1],2)) + ', Leg1 Weight: ' + str(1.00) + ', Leg2 Weight: ' + str(round(weights[1]/weights[0],2)) + ', Leg3 Weight: ' + str(round(weights[2]/weights[0],2)) + ', ADF: ' + str(round(adf_res,4)) + ', Time Stop: ' + str(entry_cond[3]) + ', Max PnL: ' + str(round(abs(entry_cond[0] - LTP_abs),4)))
            else: 
                print('Recommendation: Sell, Z-Score: ' + str(round(entry_cond[1],2)) + ', Leg1 Weight: ' + str(1.00) + ', Leg2 Weight: ' + str(round(weights[1]/weights[0],2)) + ', ADF: ' + str(round(adf_res,4)) + ', Time Stop: ' + str(entry_cond[3]) + ', Max PnL: ' + str(round(abs(entry_cond[0] - LTP_abs),4)))
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
#                         spread = weights[0] * x[i+t:i+wind+t] - weights[1] * y[i+t:i+wind+t]
                        spread = 1 * x[i+t:i+wind+t] - weights[1]/weights[0] * y[i+t:i+wind+t]
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
        max_pnl = round(abs(entry_cond[0] - LTP_abs),4)
        end_of_spread = df_save.iloc[-1]
        end_of_exp = ema.iloc[-1]
        flag_ibg = end_of_spread > end_of_exp
        take_prof = end_of_spread - max_pnl if flag_ibg else end_of_spread + max_pnl
        total_num = len(PnL_pc) - outcome.count('N/A')
        win_ratio,avg_time, max_drawdown, cum_pnl, avg_pnl = trade_stats(PnL_pc,PnL_abs,time_held,outcome,extype,args,instrument)

        print('Backtest Complete - See Statistics Below')
        if three_leg == True:    
            oup = np.column_stack((entry_date,exit_date,entry_abs,exit_abs,entry_std,exit_std,time_held,w_ratio_1,w_ratio_2,adf,PnL_pc,PnL_abs,outcome,extype))
        else: 
            oup = np.column_stack((entry_date,exit_date,entry_abs,exit_abs,entry_std,exit_std,time_held,w_ratio_1,adf,PnL_pc,PnL_abs,outcome,extype))
        writedata(oup,three_leg)
    #     trade_analytics(oup,instrument)
        if verbose == True: 
            plot_multiple(dataframeexit_save,entrycondexit_save,indexexit_save,dataframeentry_save,emaentry_save,status_save,tpsl_save,tpslstd_save,outcome,extype,dataframeexit2_save,ema_exit,std_save,wind)
        return here, df_save, ema, flag_ibg, end_of_spread, end_of_exp, take_prof, total_num, win_ratio,avg_time, max_drawdown, cum_pnl, avg_pnl
    else:
        return False,False, False, False, False, False, False, False, False,False, False, False, False    

