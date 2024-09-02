

df = pd.DataFrame(columns=[
    'DataFrame1',
    'DataFrame2',
    'Below',
    'Ticker1',
    'CountryCode1',
    'Ticker2',
    'CountryCode2',
    'EntryLevel',
    'Fourier',
    'WinRate',
    'TotalPNLBps',
    'AvgPNLPerTradeBps',
    'MaxDrawdown',
    'AvgTime'
    ])
    
    a = 136
    b = 143
    try:
        date,x,y = readdata('DataFinal.csv',date=0,inp_1=a,inp_2=b) 
        print(str(crossrates.columns[a]) + ' ' + str(crossrates.columns[b]))
        print(str(crossrates.columns[1]))
        print(a,b) 
        print(l)
        print(len(l))
        
        ######CALIBRATION HIGH VOL: USE EITHER 120 day Z=3.5/TP=-3.5 or 120 day Z=3/TP=-3 
        ######CALIBRATION LOW VOL: USE EITHER 120 day Z=2.5/TP=-2.5 or 120 day Z=2/TP=-2
        df_save, ema, flag_ibg, end_of_spread, end_of_exp, take_prof, total_num, win_ratio,avg_time, max_drawdown, cum_pnl, avg_pnl = back_test_RV(date,x,y,window=30,com=30,entry_z=2.5,maxentry_z=7,stoploss_z=10,takeprofit_z=0,instrument='swap',pca=False,verbose=False, weightthres_1 = [0.3, 1.5], weight_override = [1,1]) #weightthres_1 = [minimum entry weight, maximum entry weight]  
        new_row = {
            'DataFrame1': df_save,
            'DataFrame2': ema,
            'Below': flag_ibg,
            'Ticker1': str(crossrates.columns[a]),
            'CountryCode1': 'US',
            'Ticker2': str(crossrates.columns[b]),
            'CountryCode2': 'US',
            'EntryLevel': end_of_spread,
            'Fourier': 0.9,
            'WinRate': win_ratio,
            'TotalPNLBps': 200,
            'AvgPNLPerTradeBps': avg_pnl,
            'MaxDrawdown': 0.3,
            'AvgTime': avg_time
        }

        # Append the new row to the DataFrame
        df = df.append(new_row, ignore_index=True)

crossrates = pd.read_excel('Cross_market_data_final_final_final_final.xlsx') crossrates.to_csv('DataFinal.csv', index=False)
