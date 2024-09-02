import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api
import statsmodels as sm
import datetime  
from statsmodels.tsa.seasonal import STL

def pnl_calculator(trade, data, day, type, start, end, verbose=False):
    print("here")
    pnls = []
    Index = data.index  
    Years = list(set(Index.year))
    Years.sort()
    Years = list(map(str, Years))

    df = data.loc[:, trade]
    res = STL(df).fit()
    starts = [str(i) + '-' + start for i in Years]
    ends = [str(i) + '-' + end for i in Years]

    if type == 'Pay':

        for i in range(len(Years)):
            try:

                Data_Seasonality = res.seasonal[str(Years[i])]
                Data_Seasonality = Data_Seasonality.loc[data[Years[i]].index >=starts[i]]
                Data_Seasonality = Data_Seasonality[Data_Seasonality.index <= ends[i]]
        #         print(Years[i])
        #         print(Data_Seasonality)
                #     i+=1
                min_date = Data_Seasonality.idxmin()
        #         print(min_date)
                start_date = min_date
                end_date = min_date + pd.DateOffset(days = day)
                mask = (Data_Seasonality.index >= start_date) & (Data_Seasonality.index <= end_date)
                Min_Months = Data_Seasonality.idxmin().month
                Max_Month = Data_Seasonality.loc[mask].idxmax().month
                Min_Day = Data_Seasonality.idxmin().day
                Max_Day = Data_Seasonality.loc[mask].idxmax().day

                if len(str(Min_Months)) < 2:
                    Min_Months = '0' + str(Min_Months)
                if len(str(Max_Month)) < 2:
                    Max_Month = '0' + str(Max_Month)

                try:
                    pnl = df[str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day)] - df[str(Years[i]) + '-' + str(Min_Months) + '-' + str(Min_Day)]
        #             print(pnl)
        #             l.append(pnl, year)
                    pnls.append((pnl, str(Years[i]) + '-' + str(Min_Months) + '-' + str(Min_Day), str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day)))
                except:
                    pass
            except:
                pass

    
    elif type == 'Receive':
        for i in range(len(Years)):
            try:

                Data_Seasonality = res.seasonal[str(Years[i])]
                Data_Seasonality = Data_Seasonality.loc[data[Years[i]].index >=starts[i]]
                Data_Seasonality = Data_Seasonality[Data_Seasonality.index <= ends[i]]
                max_date = Data_Seasonality.idxmax()
                start_date = max_date
                end_date = start_date + pd.DateOffset(days = day)
                mask = (Data_Seasonality.index >= start_date) & (Data_Seasonality.index <= end_date)
                Max_Month = Data_Seasonality.idxmax().month
                Min_Month = Data_Seasonality.loc[mask].idxmin().month
                Max_Day = Data_Seasonality.idxmax().day
                Min_Day = Data_Seasonality.loc[mask].idxmin().day

                if len(str(Max_Month)) < 2:
                    Max_Month = '0' + str(Max_Month)
                if len(str(Min_Month)) < 2:
                    Min_Month = '0' + str(Min_Month)

                try:
                    pnl = df[str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day)] - df[str(Years[i]) + '-' + str(Min_Month) + '-' + str(Min_Day)]
                    pnls.append((pnl, str(Years[i]) + '-' + str(Max_Month) + '-' + str(Max_Day), str(Years[i]) + '-' + str(Min_Month) + '-' + str(Min_Day)))
                except:
                    pass
            except:
                pass
    finaldf = pd.DataFrame(pnls, columns = ['PnL', 'Entry', 'Exit'])
    mean = finaldf['PnL'].mean()
    std = finaldf['PnL'].std()
    finaldf['Win Rate'] = ((finaldf['PnL'] > 0)*1)
    winrate = finaldf['Win Rate'].mean()
    finaldf.drop('Win Rate', axis = 1, inplace=True)

    if verbose == True:
      print(type, trade)
      print('Average PnL: ' + str(round(mean, 5)))
      print('Std: ' +  str(round(std, 5)))
      return finaldf
    return mean, std, trade, winrate

def pnl_scanner(data, day, type, start, end):
  print("here")
  pnl = []
  for column in data.columns:
    stats = pnl_calculator(column, data, day, type, start, end, verbose=False)
    pnl.append(stats)
  print(pnl)
  df = pd.DataFrame(pnl, columns = ['PnL', 'Std', ' Trade', 'Win Rate'])
  df['PnL/Std'] = df['PnL'] / df['Std']
  
  df = df.sort_values(by='PnL/Std', ascending = False)
  return df


data=pd.read_excel('data.xlsx', index_col=0, parse_dates = True) 
table = pnl_scanner(data, 20, 'Receive', 1, 30)


print(table)



