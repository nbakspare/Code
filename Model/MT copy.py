import pandas as pd
from RVLibrary import *
import itertools

"""
df = pd.read_excel("data.xlsx")
csv_file = 'data.csv'
df.to_csv(csv_file, index=False)
"""

def generate_combinations(start, end):
    # Generate a list of numbers from start to end
    numbers = list(range(start, end + 1))
    
    # Generate all combinations of two numbers
    combinations = list(itertools.combinations(numbers, 2))
    
    # Convert each combination (tuple) to a list
    combinations_as_lists = [list(combo) for combo in combinations]
    
    return combinations_as_lists

# Example usage:
def generate_combinations():
    combinations = []
    # Iterate over the range of numbers from 0 to 98 with a step of 9
    for start in range(0, 99, 9):
        end = min(start + 9, 99)
        # Generate the combinations within the current group
        for i in range(start, end):
            for j in range(i + 1, end):
                if i == 0:
                    continue
                else:
                    combinations.append([i, j])
    return combinations

# Usage
l = generate_combinations()
print(l)

def LT():    
    data = pd.read_csv('trades.csv')
    date_column = data.columns[0] 
    data['Dates'] = pd.to_datetime(data['Dates'])

    data.set_index('Dates', inplace=True)
    #data.index = pd.to_datetime(data.index)
    #data.set_index(date_column, inplace=True)

    data.to_csv('trades.csv')
    cols = data.columns
    #data = data.drop(data.index[0])
    cols = data.columns
    columns = ['Name', 'Blue', 'Spread', 'Flag', 'End_Spread', 'End_Exp',
           'Take_Profit', 'Total_Num', 'Win_Ratio', 'Avg_Time',
           'Max_Drawdown', 'Cum_PNL', 'Average_PNL','Max_PNL','Fourier', 'ttof']

    # Create an empty DataFrame
    main = pd.DataFrame(columns=columns)
    for i in l:
        a = i[0] #84
        b = i[1] #87
        print(a,b)
        date,x,y = readdata('trades.csv',date=0,inp_1=a,inp_2=b) 

        combined = pd.DataFrame({'x': x, 'y': y})

        # Drop rows where either x or y is NaN
        combined_cleaned = combined.dropna()

        # Separate them back into two Series
        x = combined_cleaned['x']
        y = combined_cleaned['y']
        #print(x)
        print(x)
        print(y)
        name = str(data.columns[a]) + ' ' + str(data.columns[b])
        name1 = str(data.columns[a])
        name2 = str(data.columns[b])
        
        here,df_save, ema, flag_ibg, end_of_spread, end_of_exp, take_prof, total_num, win_ratio,avg_time, max_drawdown, cum_pnl, avg_pnl,max_pnl = back_test_RV(date,x,y,window=120,com=30,entry_z=1,maxentry_z=7,stoploss_z=10,takeprofit_z=-3,instrument='swap',pca=False,verbose=False, weightthres_1 = [0.3, 1.5], weight_override = [1,1]) #weightthres_1 = [minimum entry weight, maximum entry weight]  
        
        if win_ratio!=False:
            win = float(win_ratio[:-1])
            if here and (win > 0.0):
                ema_list = ema.tolist()
                dfs = df_save.tolist()
                country_1 = "EU" #excel_country[a-1]
                country_2 = "EU" #excel_country[b-1]
                fname = ("Receive "+ name1 + ", Pay: " + name2) if flag_ibg else ("Receive "+ name2 + ", Pay: " + name1)
                fourier, ttof = "EU","EU" #find_nearest_business_day(back_time, country_1, country_2, math.ceil(float(avg_time)))

                data2 = {
                    'Name': fname,
                    'Win_Ratio': win_ratio,
                    'Blue': (dfs,),
                    'Spread': (ema_list,),
                    'Flag': flag_ibg,
                    'End_Spread': end_of_spread*100,
                    'End_Exp': end_of_exp,
                    'Take_Profit': take_prof,
                    'Total_Num': total_num,
                    'Avg_Time': avg_time,
                    'Max_Drawdown': float(max_drawdown)/100,
                    'Cum_PNL': float(cum_pnl)/100,
                    'Average_PNL': float(avg_pnl)/100, 
                    'Max_PNL':max_pnl, 
                    #'Time_Stop':time_stop,
                    'Fourier': fourier,
                    'ttof': ttof
                }
                
                main.loc[len(main.index)] = data2
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    return main


output = LT()
output.to_csv('LT.csv')
print(output)

"""
def MT():
    columns = ['Name', 'Blue', 'Spread', 'Flag', 'End_Spread', 'End_Exp',
           'Take_Profit', 'Total_Num', 'Win_Ratio', 'Avg_Time',
           'Max_Drawdown', 'Cum_PNL', 'Average_PNL', 'Max_PNL', 'Time_Stop', 'Fourier', 'ttof']

    main = pd.DataFrame(columns=columns)
    for i in l:
        a = i[0]
        b = i[1]
        try:
            date,x,y = readdata('DataFinal.csv',date=0,inp_1=a,inp_2=b) 
            name = str(crossrates.columns[a]) + ' ' + str(crossrates.columns[b])
            name1 = str(crossrates.columns[a])
            name2 = str(crossrates.columns[b])
            print(name)
            print(a,b)
            here,back_time, max_pnl, time_stop,df_save, ema, flag_ibg, end_of_spread, end_of_exp, take_prof, total_num, win_ratio,avg_time, max_drawdown, cum_pnl, avg_pnl = back_test_RV(date,x,y,window=90,com=30,entry_z=3,maxentry_z=7,stoploss_z=10,takeprofit_z=-3,instrument='swap',pca=False,verbose=False, weightthres_1 = [0.3, 1.5], weight_override = [1,1])
            if here and (win_ratio > 79.0):
                ema_list = ema.tolist()
                dfs = df_save.tolist()
                country_1 = excel_country[a-1]
                country_2 = excel_country[b-1]
                fname = ("Receive "+ name1 + ", Pay: " + name2) if flag_ibg else ("Receive "+ name2 + ", Pay: " + name1)
                fourier, ttof = find_nearest_business_day(back_time, country_1, country_2, math.ceil(float(avg_time)))
                data = {
                    'Name': fname,
                    'Win_Ratio': win_ratio,
                    'Blue': (dfs,),
                    'Spread': (ema_list,),
                    'Flag': flag_ibg,
                    'End_Spread': end_of_spread*100,
                    'End_Exp': end_of_exp,
                    'Take_Profit': take_prof*100,
                    'Total_Num': total_num,
                    'Avg_Time': avg_time,
                    'Max_Drawdown': max_drawdown,
                    'Cum_PNL': cum_pnl,
                    'Average_PNL': avg_pnl, 
                    'Max_PNL':max_pnl*100, 
                    'Time_Stop':time_stop,
                    'Fourier': fourier,
                    'ttof': ttof
                }
                main.loc[len(main.index)] = data
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        except:
            print('Error')
            print(a,b)
    
    return main
"""

"""
here,back_time, max_pnl, time_stop,df_save, ema, flag_ibg, end_of_spread, end_of_exp, take_prof, total_num, win_ratio,avg_time, max_drawdown, cum_pnl, avg_pnl
here,                              df_save, ema, flag_ibg, end_of_spread, end_of_exp, take_prof, total_num, win_ratio,avg_time, max_drawdown, cum_pnl, avg_pnl



back_time, max_pnl, time_stop
"""