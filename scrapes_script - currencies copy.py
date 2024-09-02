# %%
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# %% [markdown]
#  ## INTRODUCTION
#  All input and output can be found in  **Y:\Analytics and Tools\Swaps\Scrape Data**
#  
#  from SDR, download the inflation, OIS, Fwds, and Vanilla pages. Ensure you save the files once they open and rename them. the naming convention is **'__YYYYMMDD'** where the dash is **inf** for inflation, **ois** for OIS, **fra** for Fwds and **irs** for Vanilla
#  
#  Only proceed once all the input excel files have been downloaded, appropriately named and placed in the **Y:\Analytics and Tools\Swaps\Scrape Data**

# %%
import pandas as pd
from xbbg import blp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
from datetime import timedelta as td
import statsmodels.api as sm
from math import isnan
import re
from ipywidgets import interact, IntSlider, Checkbox, Dropdown, Output, HBox, VBox, interactive, interactive_output, ToggleButton,Text, Button, DatePicker, IntText, ToggleButtons, RadioButtons
from IPython.display import display, clear_output
import itertools
import decimal as dec
#import warnings
#warnings.filterwarnings("ignore")

fmc = pd.DataFrame({'Date': [ '06/12/2024','07/31/2024','09/18/2024','11/07/2024','12/18/2024']})#,'01/29/2025']})
ecb = pd.DataFrame({'Date': ['06/12/2024','07/24/2024','09/18/2024','10/23/2024','12/18/2024']})#,'10/23/2024','12/18/2024']})
mpc = pd.DataFrame({'Date': ['06/20/2024','08/01/2024','09/19/2024','11/07/2024','12/19/2024']})#,'11/07/2024','12/19/2024']})
cads = pd.DataFrame({'Date': ['06/06/2024','07/25/2024','09/05/2024','10/24/2024','12/12/2024']})
rba = pd.DataFrame({'Date': ['06/19/2024','08/07/2024','09/25/2024','11/06/2024','12/11/2024','02/19/2025','04/02/2025','05/21/2025']})
rbnz = pd.DataFrame({'Date': ['07/11/2024','08/15/2024','10/10/2024','11/28/2024','02/20/2025','04/10/2025','05/29/2025']})
#boj = pd.DataFrame({'Date': ['06/17/2024','08/01/2024','09/24/2024','11/01/2024','12/20/2024']})

fmc['Date'] = pd.to_datetime(fmc['Date'])
ecb['Date'] = pd.to_datetime(ecb['Date'])
mpc['Date'] = pd.to_datetime(mpc['Date'])
cads['Date'] = pd.to_datetime(cads['Date'])
rba['Date'] = pd.to_datetime(rba['Date'])
rbnz['Date'] = pd.to_datetime(rbnz['Date'])
#boj['Date'] = pd.to_datetime(boj['Date'])



# %%
def workday(n):    
    curr = dt.today()
    while n > 0:
        curr -= td(days = 1)
        if curr.weekday()>=5:
             continue
        n -= 1
    return curr


def last_5():
    l = [dt(year,month,d),workday(1),workday(2),workday(3),workday(4)]
    p = [i.strftime('%Y%m%d') for i in l]
    k = [i.strftime('%d %b') for i in l]
    for i in k:
        print(i)
    c = input('are these dates ok? (y/n): ')
    
    if c == 'y':
        return p
    else:
        e = []
        n =int(input('How many are incorrect?: '))
        for i in range(n):
            t = int(input('Which ones? t-'))
            e.append(t)
        for i in e:
            p[i-1] = input(f'What is the correct date code for t-{i}? ')
            
        return p
        

def monthdelta(date, delta):
    m, y = (date.month+delta) % 12, date.year + ((date.month)+delta-1) // 12
    if not m: m = 12
    d = min(date.day, [31,
        29 if y%4==0 and not y%400==0 else 28,31,30,31,30,31,31,30,31,30,31][m-1])
    return (date.replace(day=d,month=m, year=y))


def get_last_date_of_month(year, month):
    
    if month == 12:
        last_date = dt(year, month, 31)
    else:
        last_date = dt(year, month + 1, 1) + td(days=-1)
    
    return last_date


def normalise(amount, min=0, max=2):
    """
    Rounds to a variable number of decimal places - as few as necessary in the range [min,max]
    :param amount: A float, int, decimal.Decimal, or string.
    :param min: the minimum number of decimal places to keep
    :param max: the maximum number of decimal places to keep
    :return: string.
    """
    if not amount:
        return str(amount)

    # To Decimal, round to highest desired precision
    d = round(dec.Decimal(amount), max)
    s = str(d)

    # Truncate as many extra zeros as we are allowed to
    for i in range(max-min):
        if s[-1] == '0':
            s = s[:-1]

    # Lose a trailing decimal point.
    if s[-1] == '.':
        s = s[:-1]

    return s

def day(d,month,year):
    c = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
    if dt.today() - dt(year,month,d) < td(days = 2):
        return 'yesterday'
    else:
        return 'on '+c[dt(year,month,d).weekday()]

# %%
################################### INFLATION SCRAPE ############################################################# 
def clean_infl_df(filepath_inf):
    #print("filepath: ", filepath_inf)
    df = pd.read_excel(filepath_inf)
    df.fillna(0, inplace = True)
    df['Trade Time'] = pd.to_datetime(df['Trade Time'])
    df['Effective'] = pd.to_datetime(df['Effective'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])  
    df['Rate 2'] = [max(i,0) for i in df['Rate 2']]
    df.sort_values('Trade Time', inplace = True)
    df['french'] = [True if  i == 'FRC-EXT-CPI' or j =='FRC-EXT-CPI' else False for i,j in  zip(df['Underlying'],df['Underlying 2'])]
    ini = []
    for i in df.index:
        if df.loc[i,'Curr'] == 'USD' and df.loc[i,'T'] == '1Y':
            act_eff = monthdelta(df.loc[i,'Effective'],-3)
            eom = get_last_date_of_month(act_eff.year,act_eff.month)
            initial_index.index = pd.to_datetime(initial_index.index).normalize()
            eom_date = eom.date()

            # Convert initial_index index to a date for comparison
            initial_index_dates = initial_index.index.date
            #print("eom: ", eom)
            #print("initial: ",initial_index.index)
            #print(f"Adjusted Effective Date: {act_eff}, End of Month: {eom}")
            
            if eom_date in initial_index_dates:
                index_position = list(initial_index_dates).index(eom_date)  # Get the index position
                ini_value = initial_index.iloc[index_position]['initial']   # Retrieve the 'initial' value using the position
                ini.append(ini_value)
                #print("HERE!")
                #print("Initial value: ", ini_value)
            else:
                ini.append(0)
        else:
            ini.append(0)
        
    df['initial'] =ini
    #print(df)
    df.to_excel('C:\\BLP\\data\\check.xlsx')
    #print(df)
    return df

def produce_infl_excel(ecy,df):
    l ={'USA':['USD',False,-3,'$'],
    'Eurozone':['EUR',False,-3,'€'],
    'UK':['GBP',False,-2,'£'],
    'France':['EUR',True,-3,'€']}
    
    new_df = df[(df['Curr'] == l[ecy][0]) & (df['T'] == '1Y')  & (df['french'] == l[ecy][1])].copy()
    new_df.sort_values(by ='Maturity', inplace = True)    
    
    
    tot = [new_df.loc[i,'Trade Time'].strftime('%X') for i in new_df.index]
    fix = [f"{monthdelta(new_df.loc[i,'Effective'],l[ecy][2]).strftime('%b%y')}/{monthdelta(new_df.loc[i,'Maturity'],l[ecy][2]).strftime('%y')}" for i in new_df.index]
    amt = [f"{l[ecy][3]}{int(new_df.loc[i,'Not. 1']//1e6)}mm" for i in new_df.index]
    
    
    if ecy == 'USA':
        rat = [new_df.loc[i,'Rate'] + new_df.loc[i,'Rate 2'] for i in new_df.index]
        #a = rat
        #print(rat)
        #rat = []
        """
        for i, j in zip(rat, new_df.index):
            initial_value = new_df.loc[j, 'initial']
            modified_rate = (1 + (i / 100))
            result = modified_rate * initial_value
            rounded_result = round(result, 2)
            rat.append(rounded_result)
            print(f"Original Rate: {i}, Initial: {initial_value}, Modified Rate: {modified_rate}, Result: {result}, Rounded: {rounded_result}")
        """
        #rat = a
        rat = [round(((1+(i/100))*new_df.loc[j,'initial']), 2) for i,j in zip(rat, new_df.index)]
        #print(rat)
        final_result = pd.DataFrame({'ToT': tot, 'Fix': fix, 'Amount': amt, 'NSA Index': rat })
        
    else:
        rat = [new_df.loc[i,'Rate'] + new_df.loc[i,'Rate 2'] for i in new_df.index]
        rat = [f"{normalise(i,1,5)}%" for i in rat]
        final_result = pd.DataFrame({'ToT': tot, 'Fix': fix, 'Amount': amt, 'Rate': rat })

    # Regex to find numeric part in the amount string
    def extract_numeric_value(amount_str):
        numeric_part = re.search(r'\d+', amount_str)
        if numeric_part:
            return int(numeric_part.group())
        return 0  # Return 0 if no numeric part is found

    # Apply the function to create a new column for comparison
    final_result['NumericAmount'] = final_result['Amount'].apply(extract_numeric_value)

    # Filter DataFrame based on the numeric value
    final_result = final_result[final_result['NumericAmount'] >= 25]

    # Drop the auxiliary column if you don't need it anymore
    final_result = final_result.drop(columns=['NumericAmount'])
    #print(final_result)
    return final_result

# %%
################################# MEETING DATE SCRAPE ########################################################
def clean_md_df(CUR,filepath):
    df = pd.read_excel(filepath)
    df.fillna(0, inplace = True)
    df.drop(columns = ['Src'], inplace = True)
    df['Trade Time'] = pd.to_datetime(df['Trade Time'])
    df['Effective'] = pd.to_datetime(df['Effective'])
    df['Maturity'] = pd.to_datetime(df['Maturity']) 
    df.sort_values('Trade Time', inplace = True)
    df['Rate'] = df['Rate'] / 100
    
    return df[(df['Curr'] == CUR)].copy()


def trim_meeting_d(bank,ser, m):
    x = []
    mno = f'{meeting_info[bank][2]}{m}{meeting_info[bank][3]}'
    for i in ser:
        if i <= mds.loc[mno,'low'] - (1/2e4):
            x.append(mds.loc[mno,'low'])
        elif i >= mds.loc[mno,'high'] + (1/2e4):
            x.append(mds.loc[mno,'high'])
        else:
            x.append(i)
    return x 

def meeting_date_fix(bank,year, month, trim = False):
    meeting_dates = meeting_info[bank][0] 
    #print(meeting_dates['Date'])
    for i in meeting_dates['Date']:
        #print(i, bank)
        if i.month == month and i.year == year:
            start = i
        else:
            #print("flagging i", i, bank)
            continue            
    #print("start: ,", start)
    end = meeting_dates.loc[meeting_dates[meeting_dates['Date'] == start].index[0] +  1, 'Date']
    m = meeting_dates[meeting_dates['Date'] == start].index[0] + 1
    mno = mno = f'{meeting_info[bank][2]}{m}{meeting_info[bank][3]}'
    mask_us = (meeting_info[bank][4]['Effective'] == start) & (meeting_info[bank][4]['Maturity'] == end) & (meeting_info[bank][4]['Underlying'].str.contains('Federal'))
    mask_eg = (meeting_info[bank][4]['Effective'] == start) & (meeting_info[bank][4]['Maturity'] == end)
                
    if bank == 'fomc':
        new_df = meeting_info[bank][4][mask_us].copy()
    else:
        new_df = meeting_info[bank][4][mask_eg].copy()
        
    new_df.reset_index(inplace = True)
    
    new_df['fac'] = [1  if i <= (mds.loc[mno,'open'] +mds.loc[mno,'close'])/2 else -1 for i in new_df['Rate'] ]
    new_df['Adj_rate'] = [new_df.loc[i,'Rate'] + (new_df.loc[i,'fac'] *  new_df.loc[i,'Othr Pmnt'] / new_df.loc[i,'DV01']) for i in range(len(new_df['Rate']))]
    
    if trim:
        new_df['Adj_rate'] = trim_meeting_d(bank,new_df['Adj_rate'],m)
    new_df['Fee'] = ['No' if i == 0 else 'Yes' for i in new_df['Othr Pmnt'] ]
    
    return new_df


def md_summ(cb_key, mstr):
    iterator = meeting_info[cb_key][0]
    mstr = mstr.copy()
    dat = []
    amt = []
    hig = []
    low = []
    avg = []
    dv1 = []
    
    for i in iterator.loc[:len(iterator)-2,'Date']:
        df = meeting_date_fix(cb_key,i.year,i.month,True)
        dat.append(i)
        amt.append(df['Not.'].sum())
        dv1.append(df['DV01'].sum())
        hig.append(df['Adj_rate'].max())
        avg.append((df['Adj_rate'] * df['DV01']).sum() / df['DV01'].sum())
        low.append(df['Adj_rate'].min())
    
    summ = pd.DataFrame({'Meeting': dat, 'Notional': amt, 'DV01': dv1, 'Low': low, 'Average': avg, 'High': hig }, index = dat)
    
    imp = []

    for i in range(len(mstr['Adj_rate'])):
        if mstr.at[i,'Effective'] == iterator.loc[0,'Date']:
            imp.append(mstr.at[i,'Adj_rate'] - meeting_info[cb_key][5])
        else:
            index = iterator.loc[iterator[iterator['Date'] == mstr.at[i,'Effective']].index[0] - 1, 'Date']
            if isnan(summ.at[index, 'Average']):
                meet_tick = f"{meeting_info[cb_key][2]}{iterator[iterator['Date'] == mstr.at[i,'Effective']].index[0]}{meeting_info[cb_key][3]}"
                imp.append(mstr.at[i,'Adj_rate'] -  mds.at[meet_tick,'close'])
            else:
                imp.append(mstr.at[i,'Adj_rate'] - summ.at[index, 'Average'])

    mstr['Implied Action'] = imp
    
    impl = []
    impa = []
    imph = []

    for i in range(len(iterator['Date']) -1):
        df = mstr[mstr['Effective'] == iterator.at[i,'Date']]
        imph.append(df['Implied Action'].max())
        impa.append((df['Implied Action'] * df['DV01']).sum() / df['DV01'].sum())
        impl.append(df['Implied Action'].min())

    summ['Implied Low'] = impl
    summ['Implied Average'] = impa
    summ['Implied High'] = imph
            
    return summ

def weighted_avg(values, weights):
    total_weighted_value = sum(v * w for v, w in zip(values, weights))
    total_weight = sum(weights)
    return total_weighted_value / total_weight

def md_present(summ):
    summ = summ.dropna()
    present = pd.DataFrame({})
    present['Meeting'] = summ['Meeting'].dt.date
    present['Notional'] = summ['Notional'] / 1000000000
    present['DV01'] = summ['DV01'] / 1000
    present['Average'] = summ['Average']
    present['Avg implied'] = summ['Implied Average'] * 10000
    
    
    """
    print(present)
    # NEW
    total_notional = summ['Notional'].sum()
    print("Summ: ", summ)
    summ['Normalized Notional'] = summ['Notional'] / total_notional
    summ['Weighted Avg'] = summ.apply(lambda row: weighted_avg(
        [row['Low'], row['Average'], row['High']], 
        [row['Normalized Notional'], row['Normalized Notional'], row['Normalized Notional']]
    ), axis=1)

    present['Weighted Avg'] = summ['Weighted Avg']
    """
    present['Traded Range'] = [f"{round(i * 10000, 1)} - {round(j * 10000, 1)}" for i,j in zip(summ['Low'],summ['High'])]
    present['Implied Range'] = [f"{round(i * 10000, 1)} - {round(j * 10000, 1)}" for i,j in zip(summ['Implied Low'],summ['Implied High'])]
    present = present[present['DV01'] != 0]

    return present

def lipstick_md(present_us,present_eu,present_gb,present_cad,present_au,present_nz):
    #print("present cad: ",present_cad)
    (us_row, us_col) = present_us.shape
    (eu_row, eu_col) = present_eu.shape
    (gb_row, gb_col) = present_gb.shape
    (cad_row, cad_col) = present_cad.shape

    (au_row, au_col) = present_au.shape
    (nz_row, nz_col) = present_nz.shape
    #(jp_row, jp_col) = present_jp.shape
    
    print("Shapes: ")
    print(present_au)
    print(present_nz)
    #print(present_jp)

    writer = pd.ExcelWriter(location1, engine='xlsxwriter',datetime_format='hh:mm:ss', date_format='mmm yy')


    present_us.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 4, startcol = 3)

    present_eu.to_excel(writer, sheet_name='Sheet1', index = False, startrow = us_row+7, startcol = 3)

    present_gb.to_excel(writer, sheet_name='Sheet1', index = False, startrow = us_row +eu_row + 10, startcol = 3)

    present_cad.to_excel(writer, sheet_name='Sheet1', index = False, startrow = us_row +eu_row +gb_row+ 13, startcol = 3)

    present_au.to_excel(writer, sheet_name='Sheet1', index = False, startrow = us_row +eu_row +gb_row+cad_row+ 16, startcol = 3)

    present_nz.to_excel(writer, sheet_name='Sheet1', index = False, startrow = us_row +eu_row +gb_row+cad_row+au_row+19, startcol = 3)

    #present_jp.to_excel(writer, sheet_name='Sheet1', index = False, startrow = us_row +eu_row +gb_row+cad_row+au_row+nz_row+22, startcol = 3)

    wb = writer.book
    ws = writer.sheets['Sheet1']

    merge_format = wb.add_format({'bold': 1,'align': 'center','valign': 'vcenter'})
    f1 = wb.add_format({'bold':True,'font_color':'purple','bg_color':'#daeef3','num_format':'0.00%'})

    def f2(fl):
        if fl == 'first':
            return wb.add_format({'bold':True,'bg_color':'#d9d9d9','align': 'center','valign': 'vcenter','text_wrap':True,'top':2,'bottom':1,'left':2,'right':1})
        elif fl == 'last':
            return wb.add_format({'bold':True,'bg_color':'#d9d9d9','align': 'center','valign': 'vcenter','text_wrap':True,'top':2,'bottom':1,'left':1,'right':2})
        else:
            return wb.add_format({'bold':True,'bg_color':'#d9d9d9','align': 'center','valign': 'vcenter','text_wrap':True,'top':2,'bottom':1,'left':1,'right':1})

    def f3(fl):
        if fl == 'last':
            return wb.add_format({'num_format':'mmm yy','bg_color':'yellow','align': 'center','valign': 'vcenter','text_wrap':True,'top':1,'bottom':2,'left':2,'right':1})
        else:
            return wb.add_format({'num_format':'mmm yy','bg_color':'yellow','align': 'center','valign': 'vcenter','text_wrap':True,'top':1,'bottom':1,'left':2,'right':1})

    def f4(fl):
        if fl == 'last':
            return wb.add_format({'num_format':'"$"#,##0.0"bn"','align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'num_format':'"$"#,##0.0"bn"','align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':1})

    def f5(fl):
        if fl == 'last':
            return wb.add_format({'num_format':'"€"#,##0.0"bn"','align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'num_format':'"€"#,##0.0"bn"','align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':1})

    def f6(fl):
        if fl == 'last':
            return wb.add_format({'num_format':'"£"#,##0.0"bn"','align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'num_format':'"£"#,##0.0"bn"','align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':1})


    def f7(fl):
        if fl == 'last':
            return wb.add_format({'num_format':'"$"#,##0"k"','align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'num_format':'"$"#,##0"k"','align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':1})

    def f8(fl):
        if fl == 'last':
            return wb.add_format({'num_format':'"€"#,##0"k"','align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'num_format':'"€"#,##0"k"','align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':1})

    def f9(fl):
        if fl == 'last':
            return wb.add_format({'num_format':'"£"#,##0"k"','align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'num_format':'"£"#,##0"k"','align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':1})

    def f10(fl):
        if fl == 'last':
            return wb.add_format({'bold':True,'italic':True,'num_format':'0.000%','font_color':'#0070c0','align': 'center','valign': 'vcenter','text_wrap':True,'top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'bold':True,'italic':True,'num_format':'0.000%','font_color':'#0070c0','align': 'center','valign': 'vcenter','text_wrap':True,'top':1,'bottom':1,'left':1,'right':1})

    def f11(fl):
        if fl == 'last':
            return wb.add_format({'bold': True,'num_format':'0.0','align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'bold': True,'num_format':'0.0','align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':1})

    def f12(fl):
        if fl == 'last':
            return wb.add_format({'align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':1})
        else:
            return wb.add_format({'align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':1})

    def f13(fl):
        if fl == 'last':
            return wb.add_format({'align': 'center','valign': 'vcenter','top':1,'bottom':2,'left':1,'right':2})
        else:
            return wb.add_format({'align': 'center','valign': 'vcenter','top':1,'bottom':1,'left':1,'right':2})

    ws.conditional_format(5,4,5+us_row,4,{'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    ws.conditional_format(5,5,5+us_row,5, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})

    ws.conditional_format(us_row+8,4,us_row+eu_row+8,4, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    ws.conditional_format(us_row+8,5,us_row+eu_row+8,5, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})

    ws.conditional_format(us_row +eu_row + 11,4,us_row +eu_row + gb_row+ 11,4, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    ws.conditional_format(us_row +eu_row + 11,5,us_row +eu_row + gb_row+ 11,5, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})

    ws.conditional_format(us_row +eu_row +gb_row+ 14,4,us_row +eu_row + gb_row +cad_row+ 14,4, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    ws.conditional_format(us_row +eu_row +gb_row+ 14,5,us_row +eu_row + gb_row+cad_row+ 14,5, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    ########## NEW 3
    ws.conditional_format(us_row +eu_row +gb_row+cad_row+ 17,4,us_row +eu_row + gb_row +cad_row+au_row+ 17,4, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    ws.conditional_format(us_row +eu_row +gb_row+cad_row+ 17,5,us_row +eu_row + gb_row+cad_row+au_row+ 17,5, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})

    ws.conditional_format(us_row +eu_row +gb_row+cad_row+au_row+ 20,4,us_row +eu_row + gb_row +cad_row+au_row+nz_row+20,4, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    ws.conditional_format(us_row +eu_row +gb_row+cad_row+au_row+ 20,5,us_row +eu_row + gb_row+cad_row+au_row+nz_row+20,5, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    """
    ws.conditional_format(us_row +eu_row +gb_row+cad_row+au_row+nz_row+ 23,4,us_row +eu_row + gb_row +cad_row+au_row+nz_row+jp_row+23,4, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    ws.conditional_format(us_row +eu_row +gb_row+cad_row+au_row+nz_row+ 23,5,us_row +eu_row + gb_row+cad_row+au_row+nz_row+jp_row+23,5, {'type': '2_color_scale','min_color':'#e3e7ea', 'max_color':'#cadbec'})
    """
    t_minus_days[0].value
    ws.insert_image('A1', 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\logo.png')
    ws.merge_range('D2:J2','Snapshot of Meeting Dates traded on SDR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year),merge_format)
    ws.write('D4','FOMC',f1)
    ws.write('E4','FF',f1)
    ws.write('F4',fed_funds,f1)

    ws.write(6 + us_row,3,'ECB',f1)
    ws.write(6 + us_row,4,'ESTR',f1)
    ws.write(6 + us_row,5,estron,f1)

    ws.write(9 + us_row + eu_row,3,'MPC',f1)
    ws.write(9 + us_row + eu_row,4,'SONIA',f1)
    ws.write(9 + us_row + eu_row,5,sonia,f1)

    ws.write(12 + us_row + eu_row + gb_row,3,'BOC',f1)
    ws.write(12 + us_row + eu_row + gb_row,4,'CAONREPO',f1)
    ws.write(12 + us_row + eu_row + gb_row,5,caddy,f1)

    ws.write(15 + us_row + eu_row + gb_row + cad_row,3,'RBA',f1)
    ws.write(15 + us_row + eu_row + gb_row + cad_row,4,'RBACOR',f1)
    ws.write(15 + us_row + eu_row + gb_row + cad_row,5,aussie,f1)

    ws.write(18 + us_row + eu_row + gb_row + cad_row + au_row,3,'RBNZ',f1)
    ws.write(18 + us_row + eu_row + gb_row + cad_row + au_row,4,'NZOCRS',f1)
    ws.write(18 + us_row + eu_row + gb_row + cad_row + au_row,5,newz,f1)
    """
    ws.write(21 + us_row + eu_row + gb_row + cad_row + au_row + nz_row,3,'BOJ',f1)
    ws.write(21 + us_row + eu_row + gb_row + cad_row + au_row + nz_row,4,'MUTKCALM',f1)
    ws.write(21 + us_row + eu_row + gb_row + cad_row + au_row + nz_row,5,jpy,f1)
    """
    for i in range(6,3+us_col):
        ws.write(3,i,'',f1)
        ws.write(6 + us_row,i,'',f1)
        ws.write(9 + us_row + eu_row,i,'',f1)
        ws.write(12 + us_row + eu_row + gb_row,i,'',f1)
        ws.write(15 + us_row + eu_row + gb_row + cad_row,i,'',f1)
        ws.write(18 + us_row + eu_row + gb_row + cad_row + au_row,i,'',f1)
        #ws.write(21 + us_row + eu_row + gb_row + cad_row + au_row + nz_row,i,'',f1)

    for i in range(us_col):
        if i == 0:
            fl = 'first'
        elif i == us_col -1:
            fl = 'last'
        else:
            fl = ''
        ws.write(4,i+3,present_us.columns[i],f2(fl))
        ws.write(7 + us_row,i+3,present_us.columns[i],f2(fl))
        ws.write(10 + us_row + eu_row,i+3,present_us.columns[i],f2(fl))
        ws.write(13 + us_row + eu_row+gb_row,i+3,present_us.columns[i],f2(fl))
        ws.write(16 + us_row + eu_row+gb_row+cad_row,i+3,present_us.columns[i],f2(fl))
        ws.write(19 + us_row + eu_row+gb_row+cad_row+au_row,i+3,present_us.columns[i],f2(fl))
        #ws.write(22 + nz_row+us_row + eu_row+gb_row+cad_row+au_row,i+3,present_us.columns[i],f2(fl))

    for i in range(us_row):
        if i == us_row -1:
            fl = 'last'
        else:
            fl = ''
        ws.write(5+i,3,present_us.iat[i,0],f3(fl))
        ws.write(5+i,4,present_us.iat[i,1],f4(fl))
        ws.write(5+i,5,present_us.iat[i,2],f7(fl))
        ws.write(5+i,6,present_us.iat[i,3],f10(fl))
        ws.write(5+i,7,present_us.iat[i,4],f11(fl))
        ws.write(5+i,8,present_us.iat[i,5],f12(fl))
        ws.write(5+i,9,present_us.iat[i,6],f13(fl))
        #ws.write(5+i,10,present_us.iat[i,6],f13(fl))

    for i in range(eu_row):
        if i == eu_row -1:
            fl = 'last'
        else:
            fl = ''
        ws.write(8 + us_row+i,3,present_eu.iat[i,0],f3(fl))
        ws.write(8 + us_row+i,4,present_eu.iat[i,1],f5(fl))
        ws.write(8 + us_row+i,5,present_eu.iat[i,2],f8(fl))
        ws.write(8 + us_row+i,6,present_eu.iat[i,3],f10(fl))
        ws.write(8 + us_row+i,7,present_eu.iat[i,4],f11(fl))
        ws.write(8 + us_row+i,8,present_eu.iat[i,5],f12(fl))
        ws.write(8 + us_row+i,9,present_eu.iat[i,6],f13(fl))
        #ws.write(8 + us_row+i,10,present_eu.iat[i,6],f13(fl))

    for i in range(gb_row):
        if i == gb_row -1:
            fl = 'last'
        else:
            fl = ''
        ws.write(11 + us_row + eu_row+i,3,present_gb.iat[i,0],f3(fl))
        ws.write(11 + us_row + eu_row+i,4,present_gb.iat[i,1],f6(fl))
        ws.write(11 + us_row + eu_row+i,5,present_gb.iat[i,2],f9(fl))
        ws.write(11 + us_row + eu_row+i,6,present_gb.iat[i,3],f10(fl))
        ws.write(11 + us_row + eu_row+i,7,present_gb.iat[i,4],f11(fl))
        ws.write(11 + us_row + eu_row+i,8,present_gb.iat[i,5],f12(fl))
        ws.write(11 + us_row + eu_row+i,9,present_gb.iat[i,6],f13(fl))
        #ws.write(11 + us_row + eu_row+i,10,present_gb.iat[i,6],f13(fl))

    for i in range(cad_row):
        if i == cad_row -1:
            fl = 'last'
        else:
            fl = ''
        ws.write(14 + gb_row + us_row + eu_row+i,3,present_cad.iat[i,0],f3(fl))
        ws.write(14 + gb_row + us_row + eu_row+i,4,present_cad.iat[i,1],f6(fl))
        ws.write(14 + gb_row + us_row + eu_row+i,5,present_cad.iat[i,2],f9(fl))
        ws.write(14 + gb_row + us_row + eu_row+i,6,present_cad.iat[i,3],f10(fl))
        ws.write(14 + gb_row + us_row + eu_row+i,7,present_cad.iat[i,4],f11(fl))
        ws.write(14 + gb_row + us_row + eu_row+i,8,present_cad.iat[i,5],f12(fl))
        ws.write(14 + gb_row + us_row + eu_row+i,9,present_cad.iat[i,6],f13(fl))
        #ws.write(14 + gb_row + us_row + eu_row+i,10,present_cad.iat[i,6],f13(fl))

    for i in range(au_row):
        if i == au_row -1:
            fl = 'last'
        else:
            fl = ''
        ws.write(17 + gb_row + us_row + eu_row + cad_row +i,3,present_au.iat[i,0],f3(fl))
        ws.write(17 + gb_row + us_row + eu_row+ cad_row +i,4,present_au.iat[i,1],f6(fl))
        ws.write(17 + gb_row + us_row + eu_row+ cad_row +i,5,present_au.iat[i,2],f9(fl))
        ws.write(17 + gb_row + us_row + eu_row+ cad_row +i,6,present_au.iat[i,3],f10(fl))
        ws.write(17 + gb_row + us_row + eu_row+ cad_row +i,7,present_au.iat[i,4],f11(fl))
        ws.write(17 + gb_row + us_row + eu_row+ cad_row +i,8,present_au.iat[i,5],f12(fl))
        ws.write(17 + gb_row + us_row + eu_row+ cad_row +i,9,present_au.iat[i,6],f13(fl))
        #ws.write(17 + gb_row + us_row + eu_row+ cad_row +i,10,present_au.iat[i,6],f13(fl))

    for i in range(nz_row):
        if i == nz_row -1:
            fl = 'last'
        else:
            fl = ''
        ws.write(20 + gb_row + us_row + eu_row+ cad_row+au_row +i,3,present_nz.iat[i,0],f3(fl))
        ws.write(20 + gb_row + us_row + eu_row+ cad_row+au_row +i,4,present_nz.iat[i,1],f6(fl))
        ws.write(20 + gb_row + us_row + eu_row+ cad_row+au_row +i,5,present_nz.iat[i,2],f9(fl))
        ws.write(20 + gb_row + us_row + eu_row+ cad_row+au_row +i,6,present_nz.iat[i,3],f10(fl))
        ws.write(20 + gb_row + us_row + eu_row+ cad_row+au_row +i,7,present_nz.iat[i,4],f11(fl))
        ws.write(20 + gb_row + us_row + eu_row+ cad_row+au_row +i,8,present_nz.iat[i,5],f12(fl))
        ws.write(20 + gb_row + us_row + eu_row+ cad_row+au_row +i,9,present_nz.iat[i,6],f13(fl))
        #ws.write(20 + gb_row + us_row + eu_row+ cad_row+au_row +i,10,present_nz.iat[i,6],f13(fl))
    """
    for i in range(jp_row):
        if i == jp_row -1:
            fl = 'last'
        else:
            fl = ''
        ws.write(23 + gb_row + us_row + eu_row+ cad_row+au_row + nz_row+i,3,present_jp.iat[i,0],f3(fl))
        ws.write(23 + gb_row + us_row + eu_row+ cad_row+au_row + nz_row+i,4,present_jp.iat[i,1],f6(fl))
        ws.write(23 + gb_row + us_row + eu_row+ cad_row+au_row + nz_row+i,5,present_jp.iat[i,2],f9(fl))
        ws.write(23 + gb_row + us_row + eu_row+ cad_row+au_row + nz_row+i,6,present_jp.iat[i,3],f10(fl))
        ws.write(23 + gb_row + us_row + eu_row+ cad_row+au_row + nz_row+i,7,present_jp.iat[i,4],f11(fl))
        ws.write(23 + gb_row + us_row + eu_row+ cad_row+au_row + nz_row+i,8,present_jp.iat[i,5],f12(fl))
        ws.write(23 + gb_row + us_row + eu_row+ cad_row+au_row + nz_row+i,9,present_jp.iat[i,6],f13(fl))
    """

    ws.hide_gridlines(2)
    writer.close()

######################################## SARON AND SCANDI #################################################

def saron_scrape(filepath):
    window_l = ['0 days', '365 days','730 days','1825 days','3650 days']
    window_u = ['365 days','730 days','1825 days','3650 days','36500 days']
    df = pd.read_excel(filepath)
    #print(filepath)
    #print("df with error: ",df)
    df = df[df['Curr'] == 'CHF'].copy()
    df['Effective'] = pd.to_datetime(df['Effective'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    if df.shape[0] != 0:
        df['today'] = [dt.today()] * len(df['Maturity'])
        df['days'] = df['Maturity'] - df['today']
        subsets = [df[(i <= df['days']) & (df['days'] < j)] for i,j in zip(window_l,window_u)]
        dvo1 = [i['DV01'].sum() for i in subsets]
        return [round(i/1e3,0) for i in dvo1]
    else:
        return [0] * 5
    
    
def scandi_scrape(scand, filepath):
    window_l = ['0 days', '93 days','186 days', '279 days','365 days']
    window_u = [ '93 days','186 days', '279 days','365 days','36500 days']
    df = pd.read_excel(filepath)
    df = df[df['Curr'] == scand].copy()
    df['Effective'] = pd.to_datetime(df['Effective'])
    df['Maturity'] = pd.to_datetime(df['Maturity'])
    df['days'] = df['Maturity'] - df['Effective']
    df = df[(df['days'] > '75 days') & (df['days'] < '94 days') ]
    if df.shape[0] != 0:
        df['today'] = [dt.today() + td(1)] * len(df['Effective'])
        df['fwd'] = df['Effective'] - df['today']
        subsets = [df[(i <= df['fwd']) & (df['fwd'] < j)] for i,j in zip(window_l,window_u)]
        notl = [i['Not.'].sum() for i in subsets]
        return [round(i/1e6,0) for i in notl]
    else:
        return [0] * 5

def saron_scandi_lipstick(saron,nok,sek):
    
    writer = pd.ExcelWriter(location2, engine='xlsxwriter',datetime_format='hh:mm:ss', date_format='mmm yy')

    saron.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 3, startcol = 20)

    nok.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 10, startcol = 20)

    sek.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 17, startcol = 20)

    wb = writer.book
    ws = writer.sheets['Sheet1']

    f1 = wb.add_format({'italic': True, 'font_size': 10})

    chart1 = wb.add_chart({'type': 'column'})
    chart2 = wb.add_chart({'type': 'column'})
    chart3 = wb.add_chart({'type': 'column'})

    chart1l = wb.add_chart({'type': 'line'})
    chart2l = wb.add_chart({'type': 'line'})
    chart3l = wb.add_chart({'type': 'line'})

    chart1.add_series({'categories': ['Sheet1', 4, 20, 8,20 ],
                       'values': ['Sheet1', 4, 21, 8,21 ],
                       'fill':{'color': 'red'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart1l.add_series({'categories': ['Sheet1', 4, 20, 8,20 ],
                       'values': ['Sheet1', 4, 22, 8,22],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart1.combine(chart1l)
    chart1.set_title({'name':'OIS traded on SARON '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart1.set_chartarea({'border': {'none': True}})
    chart1.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart1.set_y_axis({'name':"DV01 ('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart1.set_legend({'none': True})

    chart2.add_series({'categories': ['Sheet1', 11, 20, 15,20 ],
                       'values': ['Sheet1', 11, 21, 15,21 ],
                       'fill': {'color': '#001F58'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart2l.add_series({'categories': ['Sheet1', 11, 20, 15,20 ],
                       'values': ['Sheet1', 11, 22, 15,22],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart2.combine(chart2l)
    chart2.set_title({'name':'FRAs traded on NOK-NIBOR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart2.set_chartarea({'border': {'none': True}})
    chart2.set_x_axis({'name':'Starting Months forward',
                      'major_gridlines': {'visible': False}
                      })
    chart2.set_y_axis({'name':'Notional (MM)',
                      'major_gridlines': {'visible': False}
                      })
    chart2.set_legend({'none': True})

    chart3.add_series({'categories': ['Sheet1', 18, 20, 22,20 ],
                       'values': ['Sheet1', 18, 21, 22,21 ],
                       'fill': {'color':'#F6C600'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart3l.add_series({'categories': ['Sheet1', 18, 20, 22,20 ],
                       'values': ['Sheet1', 18, 22, 22,22],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart3.combine(chart3l)
    chart3.set_title({'name':'FRAs traded on SEK-STIBOR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart3.set_chartarea({'border': {'none': True}})
    chart3.set_x_axis({'name':'Starting Months forward',
                      'major_gridlines': {'visible': False}
                      })
    chart3.set_y_axis({'name':'Notional (MM)',
                      'major_gridlines': {'visible': False}
                      })
    chart3.set_legend({'none': True})

    ws.write('O32','Markers indicate 5 day MA of DV01 or Notionals traded',f1)
    ws.insert_image('B2', 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\logo.png')
    ws.insert_chart('D2',chart1)
    ws.insert_chart('L2',chart2)
    ws.insert_chart('H17',chart3)


    ws.hide_gridlines(2)
    writer.close()

    
############################################## EM SCRAPES ###########################################################


def em_scrape(curr, filepath):
    window_l = ['0 days', '547 days','913 days','1278 days','2737 days']
    window_u = ['547 days','913 days','1278 days','2737 days','3832 days']
    df = pd.read_excel(filepath)
    df = df[df['Curr'] == curr].copy()
    if df.shape[0] == 0:
        return [0] * 5
    else:
        df['Effective'] = pd.to_datetime(df['Effective'])
        df['Maturity'] = pd.to_datetime(df['Maturity']) 
        df['today'] = [dt.today()] * len(df['Maturity'])
        df['days'] = df['Maturity'] - df['today']
        subsets = [df[(i <= df['days']) & (df['days'] < j)] for i,j in zip(window_l,window_u)]
        dol_dvo1 = [i['DV01 (USD)'].sum() for i in subsets]
        return [round(i/1e3,0) for i in dol_dvo1]
    
def latam_lipstick(cdi,tiie,icp,ibr):
    writer = pd.ExcelWriter(location3, engine='xlsxwriter',datetime_format='hh:mm:ss', date_format='mmm yy')

    tiie.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 1)
    cdi.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 4)
    icp.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 7)
    ibr.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 10)

    wb = writer.book
    ws = writer.sheets['Sheet1']
    f1 = wb.add_format({'italic': True, 'font_size': 10})
    chart1 = wb.add_chart({'type': 'column'})
    chart2 = wb.add_chart({'type': 'column'})
    chart3 = wb.add_chart({'type': 'column'})
    chart4 = wb.add_chart({'type': 'column'})


    chart1l = wb.add_chart({'type': 'line'})
    chart2l = wb.add_chart({'type': 'line'})
    chart3l = wb.add_chart({'type': 'line'})
    chart4l = wb.add_chart({'type': 'line'})


    chart1.add_series({'categories': ['Sheet1', 51, 1, 55, 1 ],
                       'values': ['Sheet1', 51, 2, 55,2],
                       'fill':{'color': '#CE1126'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart1l.add_series({'categories': ['Sheet1', 51, 1, 55, 1 ],
                       'values': ['Sheet1', 51, 3, 55,3],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart1.combine(chart1l)
    chart1.set_title({'name':'IRS traded on MXN-TIIE '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart1.set_chartarea({'border': {'none': True}})
    chart1.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart1.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart1.set_legend({'none': True})

    chart2.add_series({'categories': ['Sheet1', 51, 4, 55,4 ],
                       'values': ['Sheet1', 51, 5, 55,5 ],
                       'fill': {'color': '#009638'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart2l.add_series({'categories': ['Sheet1', 51, 4, 55,4 ],
                       'values': ['Sheet1', 51, 6, 55,6 ],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart2.combine(chart2l)
    chart2.set_title({'name':'IRS traded on BRL-CDI '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart2.set_chartarea({'border': {'none': True}})
    chart2.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart2.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart2.set_legend({'none': True})

    chart3.add_series({'categories': ['Sheet1', 51, 7, 55,7 ],
                       'values': ['Sheet1', 51, 8, 55,8 ],
                       'fill': {'color':'#0037A1'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart3l.add_series({'categories': ['Sheet1', 51, 7, 55, 7 ],
                       'values': ['Sheet1', 51, 9, 55,9],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart3.combine(chart3l)
    chart3.set_title({'name':'IRS traded on CLP-ICP '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart3.set_chartarea({'border': {'none': True}})
    chart3.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart3.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart3.set_legend({'none': True})

    chart4.add_series({'categories': ['Sheet1', 51, 10, 55, 10 ],
                       'values': ['Sheet1', 51, 11, 55,11],
                       'fill':{'color': '#F7C700'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart4l.add_series({'categories': ['Sheet1', 51, 10, 55, 10 ],
                       'values': ['Sheet1', 51, 12, 55,12],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart4.combine(chart4l)
    chart4.set_title({'name':'OIS traded on COP-IBR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart4.set_chartarea({'border': {'none': True}})
    chart4.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart4.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart4.set_legend({'none': True})




    ws.write('O32','Markers indicate the 5 day MA of $DV01 traded',f1)

    ws.insert_image('B2', 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\logo.png')
    ws.insert_chart('D2',chart1)
    ws.insert_chart('L2',chart2)
    ws.insert_chart('D17',chart3)
    ws.insert_chart('L17',chart4)

    ws.hide_gridlines(2)
    writer.close()


    
def ceemea_lipstick(pribor,wibor,bubor,telbor,jibar):
    writer = pd.ExcelWriter(location4, engine='xlsxwriter',datetime_format='hh:mm:ss', date_format='mmm yy')

    pribor.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 1)
    wibor.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 4)
    bubor.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 7)
    jibar.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 10)
    telbor.to_excel(writer, sheet_name='Sheet1', index = False, startrow = 50, startcol = 13)


    wb = writer.book
    ws = writer.sheets['Sheet1']
    f1 = wb.add_format({'italic': True, 'font_size': 10})

    chart1 = wb.add_chart({'type': 'column'})
    chart2 = wb.add_chart({'type': 'column'})
    chart3 = wb.add_chart({'type': 'column'})
    chart4 = wb.add_chart({'type': 'column'})
    chart5 = wb.add_chart({'type': 'column'})

    chart1l = wb.add_chart({'type': 'line'})
    chart2l = wb.add_chart({'type': 'line'})
    chart3l = wb.add_chart({'type': 'line'})
    chart4l = wb.add_chart({'type': 'line'})
    chart5l = wb.add_chart({'type': 'line'})



    chart1.add_series({'categories': ['Sheet1', 51, 1, 55, 1 ],
                       'values': ['Sheet1', 51, 2, 55,2],
                       'fill':{'color': '#11457E'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart1l.add_series({'categories': ['Sheet1', 51, 1, 55, 1 ],
                       'values': ['Sheet1', 51, 3, 55,3],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart1.combine(chart1l)
    chart1.set_title({'name':'IRS traded on CZK-PRIBOR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart1.set_chartarea({'border': {'none': True}})
    chart1.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart1.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart1.set_legend({'none': True})

    chart2.add_series({'categories': ['Sheet1', 51, 4, 55,4 ],
                       'values': ['Sheet1', 51, 5, 55,5 ],
                       'fill': {'color': '#D5133A'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart2l.add_series({'categories': ['Sheet1', 51, 4, 55,4 ],
                       'values': ['Sheet1', 51, 6, 55,6 ],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart2.combine(chart2l)
    chart2.set_title({'name':'IRS traded on PLN-WIBOR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart2.set_chartarea({'border': {'none': True}})
    chart2.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart2.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart2.set_legend({'none': True})

    chart3.add_series({'categories': ['Sheet1', 51, 7, 55,7 ],
                       'values': ['Sheet1', 51, 8, 55,8 ],
                       'fill': {'color':'#455C4D'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart3l.add_series({'categories': ['Sheet1', 51, 7, 55, 7 ],
                       'values': ['Sheet1', 51, 9, 55,9],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart3.combine(chart3l)
    chart3.set_title({'name':'IRS traded on HUF-BUBOR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart3.set_chartarea({'border': {'none': True}})
    chart3.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart3.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart3.set_legend({'none': True})

    chart4.add_series({'categories': ['Sheet1', 51, 10, 55, 10 ],
                       'values': ['Sheet1', 51, 11, 55,11],
                       'fill':{'color': '#00764B'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart4l.add_series({'categories': ['Sheet1', 51, 10, 55, 10 ],
                       'values': ['Sheet1', 51, 12, 55,12],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart4.combine(chart4l)
    chart4.set_title({'name':'IRS traded on ZAR-JIBAR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart4.set_chartarea({'border': {'none': True}})
    chart4.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart4.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart4.set_legend({'none': True})

    chart5.add_series({'categories': ['Sheet1', 51, 13, 55,13 ],
                       'values': ['Sheet1', 51, 14, 55,14 ],
                       'fill': {'color': '#0036B2'},
                       'border':{'color':'black'},
                       'data_labels':{'value':True}
                      })
    chart5l.add_series({'categories': ['Sheet1', 51, 13, 55,13 ],
                       'values': ['Sheet1', 51, 15, 55,15 ],
                       'fill':{'none': True},
                        'border': {'none': True},
                        'marker': {'type': 'diamond', 'size': 8,'border': {'color': 'white'},'fill':   {'color': 'black'}},
                       'data_labels':{'value':False}
                      })
    chart5.combine(chart5l)
    chart5.set_title({'name':'IRS traded on ILS-TELBOR '+day(t_minus_days[0].value.day,t_minus_days[0].value.month,t_minus_days[0].value.year)})
    chart5.set_chartarea({'border': {'none': True}})
    chart5.set_x_axis({'name':'Maturity',
                      'major_gridlines': {'visible': False}
                      })
    chart5.set_y_axis({'name':"$DV01('000)",
                      'major_gridlines': {'visible': False}
                      })
    chart5.set_legend({'none': True})




    ws.write('U31','Markers indicate the 5 day MA of $DV01 traded',f1)

    ws.insert_image('B2', 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\logo.png')
    ws.insert_chart('D2',chart1)
    ws.insert_chart('L2',chart2)
    ws.insert_chart('T2',chart3)
    ws.insert_chart('H17',chart4)
    ws.insert_chart('P17',chart5)


    ws.hide_gridlines(2)
    writer.close()

# %% [markdown]
# ## SET DATES
# - For t-1 to t-5, input the last 5 days you have data for (ensuring the input data is up-to-date)
# 
# - Since central banks have a maintainence period//BBG does not update central bank decisions instantaneously, the cells for rate decision allow you to adjust the floatinng index of meeting date swaps until such an update has occurred. On your launchpad, check if the indices (fed funds, estr and sonia) are up-to-date, otherwise input the latest central bank decision to adjust the index 
# 
# - once you press 'Commit', you should see a small table with the dates the sheet will use. Only click 'Run Scrapes' after you see that table  
# 
# - If you try to commit dates following a meeting, the sheet will prompt you to amend code to get rid of meeting dates. Follow the instructions, save the sheet and run all cells again.

# %%
t_minus_days = [DatePicker(value = workday(i), description = f't-{i}') for i in range(1,6)]

fomc_adjust = 0  # in basis points
ecb_adjust = 0
mpc_adjust = 0
cad_adjust = 0
au_adjust=0
nz_adjust=0
#jp_adjust=0

filepath1 = None
filepaths1 = None
filepaths2 = None
filepaths3 = None
location1 = None
location2 = None
location3 = None
location4 = None
fed_funds = None
caddy = None

aussie = None
newz = None
#jpy = None

estron = None
sonia = None
initial_index = None
g = None # 

metgn = [f'USSOFED{i} Curncy' for i in range(1,9)]+[f'EESF{i}A Curncy' for i in range(1,9)]+[f'GPSF{i}A Curncy' for i in range(1,9)]+[f'CDSF{i}A Curncy' for i in range(1,9)]+[f'ADSF{i}A Curncy' for i in range(1,9)]+[f'NDSF{i}A Curncy' for i in range(1,9)]#+[f'JYSOMPM{i} Curncy' for i in range(1,5)]
mds = pd.DataFrame({}, index=metgn, columns = ['open','high','low','close']) 
meeting_info = None    

def automate_execution():
    global filepath1,filepaths1,filepaths2,filepaths3, location1, location2, location3,location4
    global g, initial_index, fed_funds,estron,sonia,mds, meeting_info, caddy
    
        
    if t_minus_days[0].value.date()>fmc.loc[0,'Date'].date():
        print(f'The FOMC swaps have rolled. Please toggle the code on, and in the code cell below the INTRODUCTION\n please delete the date for the {fmc.loc[0,"Date"].strftime("%B")} meeting. Please Run All cells after that ')
    elif t_minus_days[0].value.date()>ecb.loc[0,'Date'].date() + td(-6):
        print(f'The ECB swaps have rolled. Please toggle the code on, and in the code cell below the INTRODUCTION\n please delete the date for the {ecb.loc[0,"Date"].strftime("%B")} meeting. Please Run All cells after that ')
    elif t_minus_days[0].value.date()>mpc.loc[0,'Date'].date():
        print(f'The MPC swaps have rolled. Please toggle the code on, and in the code cell below the INTRODUCTION\n please delete the date for the {mpc.loc[0,"Date"].strftime("%B")} meeting. Please Run All cells after that ')
    else:
        clear_output()

        filepath1 = 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\inf'+t_minus_days[0].value.strftime('%Y%m%d')+'.xlsx'

        filepaths1 = ['Y:\\Analytics and Tools\\Swaps\\Scrape Data\\irs'+t_minus_days[i].value.strftime('%Y%m%d')+'.xlsx' for i in range(5)]
        filepaths2 = ['Y:\\Analytics and Tools\\Swaps\\Scrape Data\\ois'+t_minus_days[i].value.strftime('%Y%m%d')+'.xlsx' for i in range(5)]
        filepaths3 = ['Y:\\Analytics and Tools\\Swaps\\Scrape Data\\fra'+t_minus_days[i].value.strftime('%Y%m%d')+'.xlsx' for i in range(5)]

        location1 = 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\Meeting Dates\\Meeting_dates_summary_'+ t_minus_days[0].value.strftime('%Y%m%d')+'.xlsx'
        location2 = 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\SARON and Scandi\\CHF_SEK_NOK'+ t_minus_days[0].value.strftime('%Y%m%d')+'.xlsx'
        location3 = 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\LATAM\\LATAM_'+ t_minus_days[0].value.strftime('%Y%m%d')+'.xlsx'
        location4 = 'Y:\\Analytics and Tools\\Swaps\\Scrape Data\\CEEMEA\\CEEMEA_'+ t_minus_days[0].value.strftime('%Y%m%d')+'.xlsx'

        g = t_minus_days[0].value.strftime('%d %b')+'\n'
        try:
            cpi_from = dt(t_minus_days[0].value.year-1,t_minus_days[0].value.month-1,t_minus_days[0].value.day)
        except:
            cpi_from = dt(t_minus_days[0].value.year-1,dt.today().month,dt.today().day)
        initial_index = blp.bdh('CPURNSA Index','px_last',cpi_from)
        initial_index.columns = ['initial']

        fed_funds = (fomc_adjust/1e4) + (blp.bdp('FEDL01 Index', 'px_last').values[0][0] / 100)
        estron = (ecb_adjust/1e4) + (blp.bdp('ESTRON Index', 'px_last').values[0][0] / 100) 
        sonia = (mpc_adjust/1e4) + (blp.bdp('SONIO/N Index', 'px_last').values[0][0] / 100 )
        caddy = (cad_adjust/1e4) + (blp.bdp('CAONREPO INDEX', 'px_last').values[0][0] / 100 )
        #print("caddy: ", caddy)

        aussie = (au_adjust/1e4) + (blp.bdp('RBACOR INDEX', 'px_last').values[0][0] / 100 )
        newz = (nz_adjust/1e4) + (blp.bdp('NZOCRS INDEX', 'px_last').values[0][0] / 100 )
        #jpy = (jp_adjust/1e4) + (blp.bdp('MUTKCALM INDEX', 'px_last').values[0][0] / 100 )

        #print("aussie: ,",aussie, " newz: ", newz, " jpy: ", jpy)

        base = blp.bdh(metgn,['px_low','px_high','px_open','px_last'],start_date=t_minus_days[0].value,end_date=t_minus_days[0].value)
        for i in metgn:
            mds.at[i,'open'] = base[i]['px_open'][0]/100
            mds.at[i,'high'] = base[i]['px_high'][0]/100
            mds.at[i,'low'] = base[i]['px_low'][0]/100
            mds.at[i,'close'] = base[i]['px_last'][0]/100

        #print("base: ", base)
        #print("mds: ", mds)
        print("clean aussie: ",clean_md_df('AU',filepaths2[0]))
        meeting_info = {
            'fomc':[fmc,1,'USSOFED',' Curncy',clean_md_df('USD',filepaths2[0]),fed_funds],
            'ecb':[ecb,5,'EESF','A Curncy',clean_md_df('EUR',filepaths2[0]),estron],
            'mpc':[mpc,0,'GPSF','A Curncy',clean_md_df('GBP',filepaths2[0]),sonia],
            'cady':[cads,0,'CDSF','A Curncy',clean_md_df('CAD',filepaths2[0]),caddy],
            'rba':[rba,0,'ADSF','A Curncy',clean_md_df('AUD',filepaths2[0]),aussie],
            'rbnz':[rbnz,0,'NDSF','A Curncy',clean_md_df('NZD',filepaths2[0]),newz],
            #'boj':[boj,0,'JYSOMPM','Curncy',clean_md_df('JPY',filepaths2[0]),jpy],
            }
        
        print("meeting_info: ", meeting_info)
        
automate_execution()  


def run():    
    #with output2:
    clear_output()
    df1 = clean_infl_df(filepath1)

    produce_infl_excel('USA',df1).to_excel('Y:\\Analytics and Tools\\Swaps\\Scrape Data\\INFL\\USDScrape.xlsx',index = False, startcol = 0, startrow = 1 )
    produce_infl_excel('UK',df1).to_excel('Y:\\Analytics and Tools\\Swaps\\Scrape Data\\INFL\\GBPScrape.xlsx',index = False, startcol = 0, startrow = 1 )
    produce_infl_excel('Eurozone',df1).to_excel('Y:\\Analytics and Tools\\Swaps\\Scrape Data\\INFL\\EURScrape.xlsx',index = False, startcol = 0, startrow = 1 ) 
    produce_infl_excel('France',df1).to_excel('Y:\\Analytics and Tools\\Swaps\\Scrape Data\\INFL\\FREScrape.xlsx',index = False, startcol = 0, startrow = 1 )
    h = open("Y:\\Analytics and Tools\\Swaps\\Scrape Data\\INFL\\headers.txt","w")
    L = ['{US} CPI fix trades from SDR '+g, '{GB} RPI fix trades from SDR '+g,'{EU} HICP fix trades from SDR '+g, '{FR} CPI fix trades from SDR '+g]
    h.writelines(L)
    h.close()
    print('Inflation Scrapes Done') 
    
    ###### ************************ #########################
    print("here!")
    print(meeting_date_fix('cady',i.year,i.month,True) for i in mpc.Date[:-1])
    master_us1 = pd.concat([meeting_date_fix('fomc',i.year,i.month,True)for i in fmc.Date[:-1]], ignore_index = True)
    master_eu1 = pd.concat([meeting_date_fix('ecb',i.year,i.month,True) for i in ecb.Date[:-1]], ignore_index = True)
    master_gb1 = pd.concat([meeting_date_fix('mpc',i.year,i.month,True) for i in mpc.Date[:-1]], ignore_index = True)
    master_cd1 = pd.concat([meeting_date_fix('cady',i.year,i.month,True) for i in cads.Date[:-1]], ignore_index = True)
    master_au1 = pd.concat([meeting_date_fix('rba',i.year,i.month,True) for i in rba.Date[:-1]], ignore_index = True)
    master_nz1 = pd.concat([meeting_date_fix('rbnz',i.year,i.month,True) for i in rbnz.Date[:-1]], ignore_index = True)
    #master_jp1 = pd.concat([meeting_date_fix('boj',i.year,i.month,True) for i in boj.Date[:-1]], ignore_index = True)

    print(master_au1)
    print(master_nz1)
    #print(master_jp1)

    master_us1.sort_values('Effective', inplace = True)
    master_eu1.sort_values('Effective', inplace = True)
    master_gb1.sort_values('Effective', inplace = True)
    master_cd1.sort_values('Effective', inplace = True)
    master_au1.sort_values('Effective', inplace = True)
    master_nz1.sort_values('Effective', inplace = True)
    #master_jp1.sort_values('Effective', inplace = True)

    us_summary = md_summ('fomc', master_us1)
    eu_summary = md_summ('ecb', master_eu1)
    uk_summary = md_summ('mpc', master_gb1)
    cd_summary = md_summ('cady', master_cd1)
    au_summary = md_summ('rba', master_au1)
    nz_summary = md_summ('rbnz', master_nz1)
    #jp_summary = md_summ('boj', master_jp1)

    
    present_us = md_present(us_summary)
    present_eu = md_present(eu_summary)
    present_gb = md_present(uk_summary)
    present_cad = md_present(cd_summary)
    present_au = md_present(au_summary)
    present_nz = md_present(nz_summary)
    #present_jp = md_present(jp_summary)
    
    lipstick_md(present_us,present_eu,present_gb,present_cad,present_au,present_nz)
    
    print('Meeting Date Scrapes Done')
    
    ###### ************************ #########################
    
    saron = pd.DataFrame({'Maturity': ['<1Y', '1Y-2Y','2Y-5Y','5Y-10Y','>10Y']})
    nok = pd.DataFrame({'Months forward': ['0-3','3-6','6-9','9-12', '>12']})
    sek = pd.DataFrame({'Months forward': ['0-3','3-6','6-9','9-12', '>12']})

    saron['DV01'] = saron_scrape(filepaths2[0])
    nok['Notional'] = scandi_scrape('NOK',filepaths3[0])
    sek['Notional'] = scandi_scrape('SEK',filepaths3[0])
    
    saron['5D'] = sum([np.array(saron_scrape(i)) for i in filepaths2]) / 5
    nok['5D'] = sum([np.array(scandi_scrape('NOK',i)) for i in filepaths3]) /5
    sek['5D'] = sum([np.array(scandi_scrape('SEK',i)) for i in filepaths3]) /5
    
    saron_scandi_lipstick(saron,nok,sek)
    
    print('SARON Scrape Done')
    print('Scandi Scrape Done')
    
    ###### ************************ ########################
    
    pribor = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    wibor = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    jibar = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    telbor = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    tiie = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    bubor = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    icp = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    cdi = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    ibr = pd.DataFrame({'Maturity': ['<1.5Y', '1.5Y-2.5Y','2.5Y-3.5Y','3.5Y-7.5Y','7.5Y-10.5Y']})
    
    
    pribor['$DV01'] = em_scrape('CZK',filepaths1[0])
    wibor['$DV01'] = em_scrape('PLN',filepaths1[0])
    jibar['$DV01'] = em_scrape('ZAR',filepaths1[0])
    telbor['$DV01'] = em_scrape('ILS',filepaths1[0])
    tiie['$DV01'] = em_scrape('MXN',filepaths1[0])
    bubor['$DV01'] = em_scrape('HUF',filepaths1[0])
    icp['$DV01'] = em_scrape('CLP',filepaths1[0])
    cdi['$DV01'] = em_scrape('BRL',filepaths1[0])
    ibr['$DV01'] = em_scrape('COP',filepaths2[0])
    
    
    pribor['5D'] = sum([np.array(em_scrape('CZK',i)) for i in filepaths1]) /5
    wibor['5D'] = sum([np.array(em_scrape('PLN',i)) for i in filepaths1]) /5
    jibar['5D'] = sum([np.array(em_scrape('ZAR',i)) for i in filepaths1]) /5
    telbor['5D'] = sum([np.array(em_scrape('ILS',i)) for i in filepaths1]) /5
    tiie['5D'] = sum([np.array(em_scrape('MXN',i)) for i in filepaths1]) /5
    bubor['5D'] = sum([np.array(em_scrape('HUF',i)) for i in filepaths1]) /5
    icp['5D'] = sum([np.array(em_scrape('CLP',i)) for i in filepaths1]) /5
    cdi['5D'] = sum([np.array(em_scrape('BRL',i)) for i in filepaths1]) /5
    ibr['5D'] = sum([np.array(em_scrape('COP',i)) for i in filepaths2]) /5
    
    latam_lipstick(cdi,tiie,icp,ibr)
    
    print('Latam Scrape Done')
    
    ceemea_lipstick(pribor,wibor,bubor,telbor,jibar)
    
    print('CEEMEA Scrape Done')
        
        
#scrape_run.on_click(on_button_clicked2)
run()

# %%
import sys
print(sys.executable)


